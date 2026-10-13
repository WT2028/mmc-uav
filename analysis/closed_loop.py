"""Recompute the reported closed-loop metrics from the supplied numeric records.

Run from the repository root: python3 -m analysis.closed_loop
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def cut(t, values, window):
    t, values = np.asarray(t), np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    lo, hi = window
    if t[0] > lo + 1e-9 or t[-1] < hi - 1e-9 or np.any(np.diff(t) <= 0):
        raise ValueError('Incomplete or non-increasing time history')
    inside = (t > lo) & (t < hi)
    endpoints = [[np.interp(s, t, column) for column in values.T] for s in (lo, hi)]
    return np.r_[lo, t[inside], hi], np.vstack([endpoints[0], values[inside], endpoints[1]])


def rms(t, values, window):
    stamps, v = cut(t, values, window)
    return float(np.sqrt(np.trapz(np.sum(v * v, axis=1), stamps) / np.ptp(stamps)))


def reference(t, protocol):
    phase = np.clip((t - protocol.get('movement_start_controller_s', 8.)) /
                    protocol['movement_duration_s'], 0., 1.)
    if protocol.get('polynomial_order', 5) == 7:
        blend = 35 * phase**4 - 84 * phase**5 + 70 * phase**6 - 20 * phase**7
    else:
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    a, b = np.asarray(protocol['target_A_m']), np.asarray(protocol['target_B_m'])
    return a + blend[:, None] * (b - a)


def continuous_fraction(t, values, threshold, window):
    """Integrate time above a threshold of a piecewise-linear magnitude."""
    stamps, v = cut(t, values, window)
    a, b = v[:-1, 0], v[1:, 0]
    both = (a >= threshold) & (b >= threshold)
    rise, fall = (a < threshold) & (b >= threshold), (a >= threshold) & (b < threshold)
    pieces = both.astype(float)
    pieces[rise] = (b[rise] - threshold) / (b[rise] - a[rise])
    pieces[fall] = (a[fall] - threshold) / (a[fall] - b[fall])
    return float(np.dot(pieces, np.diff(stamps)) / np.ptp(stamps))


def calculate(row, config):
    f = pd.read_csv(ROOT / row['flight']).sort_values('Time').drop_duplicates('Time', keep='last')
    t, window = f.Time.to_numpy(), row['window_controller_s']
    if config['family'] == 'coordination':
        return {}
    error = f[['X', 'Y', 'Z']].to_numpy() - reference(t, config['protocol'])
    _, selected = cut(t, error, window)
    result = {'position_rmse_m': rms(t, error, window),
              'position_peak_m': float(np.linalg.norm(selected, axis=1).max())}
    attitude = f[['Roll_deg','Pitch_deg','Yaw_deg']].to_numpy() - f[
        ['Inner_roll_ref_deg','Inner_pitch_ref_deg','Inner_yaw_ref_deg']].to_numpy()
    attitude = (attitude + 180) % 360 - 180
    if config['family'] == 'scan':
        result.update(attitude_rmse_deg=rms(t, attitude, window),
                      rho_Omega=float(cut(t, f.Rotor_speed_util_actual, window)[1].max()),
                      hold_rmse_m=rms(t, error, [23.,70.]))
    if row['profile'].startswith('efficiency'):
        # The fixed-yaw task uses zero yaw; the logged yaw reference is also zero.
        result.update(yaw_rmse_deg=rms(t, attitude[:,2], window),
                      peak_rotor_util_450=float(cut(t, f.Rotor_speed_util_actual, window)[1].max()))
    if 'servo' in row:
        servo = pd.read_csv(ROOT / row['servo'])
        offset = row.get('native_clock_offset_s')
        if offset is None:
            selected = f[(f.Time >= window[0]) & (f.Time <= window[1])]
            offset = float(np.median(selected.Joint_feedback_clock_s - selected.Time))
        if config['family'] == 'scan':
            maxima = servo.assign(position=servo.position.abs(), velocity=servo.velocity.abs(),
                                  force_N=servo.force_N.abs()).groupby('sim_time_s')[['position','velocity','force_N']].max()
            stamps = maxima.index.to_numpy() - offset
            result.update(rho_s=float(cut(stamps, maxima.position, window)[1].max() / .15),
                          velocity_boundary_fraction=continuous_fraction(stamps, maxima.velocity, .245, window),
                          force_boundary_fraction=continuous_fraction(stamps, maxima.force_N, 1.96, window))
        else:
            tracks = [g.sort_values('sim_time_s').drop_duplicates('sim_time_s',keep='last')
                      for _, g in servo.groupby('joint_name')]
            if len(tracks) != 4:
                raise ValueError('Expected four independently recorded rail states')
            lo, hi = max(g.sim_time_s.iloc[0] for g in tracks), min(g.sim_time_s.iloc[-1] for g in tracks)
            stamps = np.unique(np.concatenate([g.sim_time_s.to_numpy() for g in tracks]))
            stamps = stamps[(stamps >= lo) & (stamps <= hi)]
            for column, threshold, name in [('position', .98*.135, 'native_position_near_fraction'),
                                            ('velocity', .98*.25, 'native_speed_near_fraction')]:
                values = np.array([np.interp(stamps, g.sim_time_s, g[column]) for g in tracks])
                flags = (np.abs(values) >= threshold).any(axis=0).astype(float)
                clock, cut_flags = cut(stamps-offset, flags, window)
                result[name] = float(np.trapz(cut_flags[:,0], clock) / np.ptp(clock))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/closed_loop_metrics.json')
    parser.add_argument('--run-dir', type=Path, help='Analyze a new runner output instead of the supplied records')
    args = parser.parse_args()
    if args.run_dir is not None:
        folder=args.run_dir.resolve()
        config=json.loads((folder/'configuration.json').read_text())
        logs=list(folder.glob('mmc_flight*.csv'))
        if len(logs)!=1:
            parser.error('Expected one completed flight CSV')
        if config['family']=='coordination':
            parser.error('Use coordination_replay for the coordinated-motion sensor history')
        row={'profile':config['name'],'flight':str(logs[0]),
             'window_controller_s':config['protocol']['main_window_controller_s']}
        if config['family'] in ('scan','pressure'):
            row['servo']=str(folder/'raw_servo_state.csv.gz')
        values=calculate(row,config)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(values,indent=2)+'\n')
        print(json.dumps(values,indent=2))
        return
    rows, checks, groups = [], [], {}
    for row in json.loads((ROOT/'data/closed_loop/index.json').read_text()):
        for field in ('flight','servo'):
            if field in row and hashlib.sha256((ROOT/row[field]).read_bytes()).hexdigest() != row[field+'_sha256']:
                raise ValueError('Dataset checksum mismatch: '+row[field])
        config = json.loads((ROOT/'experiments/configs'/f"{row['profile']}.json").read_text())
        actual = calculate(row, config)
        for key, expected in row['reference_metrics'].items():
            checks.append({'profile':row['profile'],'repeat':row['repeat'],'metric':key,
                           'absolute_difference':abs(actual[key]-expected),
                           'passed':abs(actual[key]-expected)<1e-6})
        rows.append({'profile':row['profile'],'repeat':row['repeat'],'metrics':actual})
        groups.setdefault(row['profile'],[]).append(actual)
    summary = {name:{k:{'n':len(values),'mean':float(np.mean([r[k] for r in values])),
                        'min':float(min(r[k] for r in values)), 'max':float(max(r[k] for r in values))}
                    for k in values[0]} for name,values in groups.items()}
    result = {'all_passed':all(c['passed'] for c in checks),'checks':checks,'per_run':rows,'groups':summary}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    failed = [c for c in checks if not c['passed']]
    print(json.dumps({'runs':len(rows),'checks':len(checks),'failed':failed,'output':str(args.output)},indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
