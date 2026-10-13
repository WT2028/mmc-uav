#!/usr/bin/env python3
"""Physical step tests on either side of the response-time and speed criteria."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import numpy as np

from . import actuator_probe as instrument
from . import mechanism as mech
from .cad import write_json,write_csv,sha256
mech.write_json=write_json
mech.write_csv=write_csv
mech.sha=sha256

ROOT = Path(__file__).resolve().parents[2]
TOL = dict(instrument.TOL)
# Same model trajectory budget as the existing 10 mm check, frozen also for
# 50 mm comparisons. Do not relax it after looking at a response.
TOL['trajectory_rms_constrained_m'] = .0003
TOL['t90_constrained_relative_error'] = .05
TOL['command_clipping_error_m'] = 1e-12


def read_csv(path):
    return instrument.read_csv(path)


def probe_frequencies():
    c = mech.legacy.t90_dimensionless(mech.DESIGN['zeta'])
    _, cv = mech.legacy.peak_velocity_dimensionless(mech.DESIGN['zeta'])
    low = c / mech.DESIGN['actuator_t90_requirement_s']
    high = mech.DESIGN['velocity_limit_m_s'] / (cv * mech.DESIGN['fixed_actuator_step_m'])
    return {'wn10': 10., 'wn30': 30.,
            'time_below': .98*low, 'time_above': 1.02*low,
            'speed_below': .98*high, 'speed_above': 1.02*high}


def sequence(_case):
    amplitudes = [0., .01, 0., .05, 0., -.01, 0., -.05, 0.]
    return [{'requested': {c: a for c in instrument.COLORS},
             'hold_s': .5 if i == 0 else 1.5} for i, a in enumerate(amplitudes)]


def analyze_case(raw, out, qa, *, reused=False):
    if out.exists() and (out/'comparison_summary.json').exists():
        raise RuntimeError(f'Refusing to overwrite probe comparison: {out}')
    out.mkdir(parents=True, exist_ok=True); qa.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((raw/'config.json').read_text())
    events = json.loads((raw/'events.json').read_text())
    telemetry = {c: read_csv(raw/f'servo_{c}_raw.csv') for c in instrument.COLORS}
    wn = float(cfg['params']['wn'])
    checks, metrics, reference_rows = [], [], []
    def check(name, value, limit):
        ok = value is not None and value <= limit
        checks.append({'name':name,'value':value,'limit':limit,'passed':bool(ok)})
    for c, measured in telemetry.items():
        ts = np.array([r['sim_time_s'] for r in measured])
        check(c+'_max_telemetry_gap_s', float(np.max(np.diff(ts))), TOL['maximum_sample_gap_s'])
        check(c+'_peak_speed_m_s', max(abs(r['velocity_m_s']) for r in measured), TOL['peak_speed_m_s'])
        check(c+'_peak_position_m', max(abs(r['position_m']) for r in measured), TOL['peak_position_m'])
        check(c+'_peak_force_N', max(abs(r['applied_force_N']) for r in measured), TOL['effort_limit_N'])
        check(c+'_clipping_error_m', max(abs(r['target_m']-max(-.135,min(.135,r['requested_m']))) for r in measured), TOL['command_clipping_error_m'])
        for index, event in enumerate(events):
            if index == 0:
                continue
            requested = event['requested'][c]
            previous = max(-.135,min(.135,events[index-1]['requested'][c]))
            target = max(-.135,min(.135,requested)); delta = target-previous
            if not (math.isclose(abs(delta),.01,abs_tol=1e-8) or math.isclose(abs(delta),.05,abs_tol=1e-8)):
                continue
            start = event['sim_time_s']
            end = events[index+1]['sim_time_s'] if index+1<len(events) else start+event['hold_s']
            selected = [r for r in measured if start<=r['sim_time_s']<end and abs(r['requested_m']-requested)<1e-9]
            if not selected:
                checks.append({'name':f'{c}_{index}_received','passed':False}); continue
            received = selected[0]['sim_time_s']
            elapsed = np.array([r['sim_time_s']-received for r in selected])
            q = np.array([r['position_m'] for r in selected])
            v = np.array([r['velocity_m_s'] for r in selected])
            q0, v0 = float(q[0]), float(v[0])
            mass = float(cfg['params']['moving_mass'])
            models = {}
            for free in [False,True]:
                label='force_speed_free_pair_observer' if free else 'force_speed_fixed_base_zero_load'
                trajectory=mech.constrained_step(wn,mass,target,free_body=free,dt=.001,
                         duration=max(1.5,float(elapsed[-1])+.001),q0=q0,v0=v0)
                rt=np.array([r['time_s'] for r in trajectory])
                rq=np.array([r['position_m'] for r in trajectory])
                rv=np.array([r['velocity_m_s'] for r in trajectory])
                predicted_q=np.interp(elapsed,rt,rq)
                predicted_v=np.interp(elapsed,rt,rv)
                t90ref=mech.crossing(rt,rq,target,q0=previous)
                t90measured=mech.crossing(elapsed,q,target,q0=previous)
                rms=float(np.sqrt(np.mean((q-predicted_q)**2)))
                max_error=float(np.max(np.abs(q-predicted_q)))
                vrms=float(np.sqrt(np.mean((v-predicted_v)**2)))
                models[label]={'t90_reference_s':t90ref,'trajectory_rms_m':rms,
                               'trajectory_peak_error_m':max_error,'velocity_rms_m_s':vrms,
                               'model_peak_force_N':max(abs(r['applied_force_N']) for r in trajectory),
                               'model_peak_speed_m_s':max(abs(r['velocity_m_s']) for r in trajectory)}
                for t, qm, vm, qp, vp in zip(elapsed,q,v,predicted_q,predicted_v):
                    reference_rows.append({'case':raw.name,'segment':index,'joint':c,
                       'wn_rad_s':wn,'model':label,'relative_time_s':float(t),
                       'measured_position_m':float(qm),'measured_velocity_m_s':float(vm),
                       'reference_position_m':float(qp),'reference_velocity_m_s':float(vp)})
            physical_model=models['force_speed_free_pair_observer']
            check(f'{c}_{index}_constrained_trajectory_rms_m',physical_model['trajectory_rms_m'],TOL['trajectory_rms_constrained_m'])
            ref=physical_model['t90_reference_s']
            t90_error=abs(t90measured/ref-1) if t90measured is not None and ref is not None else None
            check(f'{c}_{index}_constrained_t90_relative_error',t90_error,TOL['t90_constrained_relative_error'])
            check(f'{c}_{index}_command_latency_s',received-start,.05)
            steady=np.mean(q[elapsed>=elapsed[-1]-.2])
            check(f'{c}_{index}_steady_error_m',float(abs(steady-target)),TOL['steady_error_m'])
            positive=delta>0
            ideal_force=mass*wn**2*abs(delta)
            force_active=sum(abs(r['applied_force_N']) >= mech.DESIGN['force_limit_N']-1e-6 for r in selected)
            sample_dt=float(np.median(np.diff(elapsed)))
            metrics.append({'case':raw.name,'segment':index,'joint':c,'wn_rad_s':wn,
                            'step_m':delta,'positive':positive,'received_time_s':received,
                            't90_measured_s':t90measured,
                            'ideal_t90_s':mech.legacy.t90_dimensionless(1.2)/wn,
                            't90_requirement_met':t90measured is not None and t90measured<=.35,
                            'ideal_speed_limit_inactive':mech.legacy.peak_velocity_dimensionless(1.2)[1]*wn*abs(delta)<=.25,
                            'ideal_initial_force_N':ideal_force,
                            'ideal_initial_force_screen_met':ideal_force<=2.0,
                            'measured_force_limit_time_s':force_active*sample_dt,
                            'measured_peak_force_N':max(abs(r['applied_force_N']) for r in selected),
                            'measured_peak_speed_m_s':float(max(abs(v))),
                            'models':models})
    mech.write_csv(out/'aligned_model_comparison.csv',reference_rows)
    flat=[]
    for row in metrics:
        base={k:v for k,v in row.items() if k!='models'}
        for label,m in row['models'].items():
            flat.append({**base,'reference_model':label,**m})
    mech.write_csv(out/'step_summary.csv',flat)
    result={'case':raw.name,'wn_rad_s':wn,'reused_immutable_run':reused,
            'raw_dir':str(raw),
            'raw_sha256':{p.name:mech.sha(p) for p in raw.iterdir() if p.name in ['config.json','events.json','joint_raw.csv','odometry_raw.csv'] or p.name.startswith('servo_') and p.suffix=='.csv'},
            'checks':checks,'passed':all(c['passed'] for c in checks),'step_metrics':metrics,
            'model_scope':'Free-floating pair scalar recoil with installed observer/force/speed/stopping law; complete CAD angular motion is validated by the actual plant, not reproduced by this scalar surrogate',
            'actual_states_not_replaced':True,'gravity':cfg.get('gravity'),'rotor_commands':cfg.get('rotor_commands'),
            'selection':'Only 10/50 mm benchmark steps; larger travel/out-of-range segments retained in raw files and included in resource checks',
            'acceptance_tolerances_frozen':TOL,'script_sha256':mech.sha(__file__),
            'surrogate_script_sha256':mech.sha(mech.__file__)}
    write_json(out/'comparison_summary.json',result)
    write_json(qa/'checks.json',{'passed':result['passed'],'checks':checks})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['all',*probe_frequencies()],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'results/bandwidth')
    parser.add_argument('--run',action='store_true',help='Collect new physical measurements')
    parser.add_argument('--timeout',type=float,default=100.)
    args=parser.parse_args()
    labels=list(probe_frequencies()) if args.case=='all' else [args.case]
    for label in labels:
        if args.run:
            raw=args.output/'physical'/label
            if raw.exists():raise FileExistsError(raw)
            instrument.PARAMS=dict(instrument.PARAMS,wn=probe_frequencies()[label])
            instrument.sequence=sequence
            instrument.run_case(raw,label,ROOT/'src/mmc_uav_description/urdf/mmc_uav.urdf',args.timeout)
        else:
            raw=ROOT/'data/bandwidth'/label
        result=analyze_case(raw,args.output/'comparison'/label,args.output/'checks'/label)
        print(json.dumps({'case':label,'passed':result['passed']}),flush=True)
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
