#!/usr/bin/env python3
"""Time-weighted metrics and observed plotting inputs for one recorded flight.

Controller time and stamped Gazebo sensor time are distinct. Their recorded
offset is explicitly reported; metrics use the recorded time alignment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


FIG = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut(t, y, start, end):
    if t[0] > start or t[-1] < end:
        raise ValueError('Metric window is not fully observed')
    mask = (t > start) & (t < end)
    return np.r_[start, t[mask], end], np.r_[np.interp(start, t, y), y[mask], np.interp(end, t, y)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    case, out = args.case_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((case/'runtime_summary.json').read_text())
    csv_path = case/runtime['controller_csv'][0]
    df = pd.read_csv(csv_path).sort_values('Time').drop_duplicates('Time', keep='last')
    t = df.Time.to_numpy()
    start, end = 8.0, 23.0  # Frozen 5 s takeoff + 3 s holding; 15 s main window.
    clock = df.Joint_feedback_clock_s.to_numpy()
    offset_values = (clock-t)[(t >= start) & (t <= end)]
    offset = float(np.median(offset_values))
    ep = df[['X','Y','Z']].to_numpy()-df[['X_ref','Y_ref','Z_ref']].to_numpy()
    ea = np.radians(df[['Roll_deg','Pitch_deg','Yaw_deg']].to_numpy()-df[['Roll_ref_deg','Pitch_ref_deg','Yaw_ref_deg']].to_numpy())
    ea[:,2] = (ea[:,2]+np.pi)%(2*np.pi)-np.pi
    norm_ep = np.linalg.norm(ep, axis=1)
    norm_ea = np.linalg.norm(ea, axis=1)
    rho_s = np.maximum(abs(df.Slider_X_actual.to_numpy()),abs(df.Slider_Y_actual.to_numpy()))/.15
    rho_v = np.maximum(abs(df.Slider_X_vel_actual.to_numpy()),abs(df.Slider_Y_vel_actual.to_numpy()))/.25
    clipped = abs(df.Tau_z_request_raw.to_numpy()-df.Tau_z_request_clipped.to_numpy()) > 1e-5
    def peak(y):
        return float(cut(t,np.asarray(y),start,end)[1].max())
    def rms(y):
        tt, yy = cut(t,np.asarray(y),start,end)
        return float(np.sqrt(np.trapz(yy**2,tt)/(end-start)))
    def fraction(y):
        tt, yy = cut(t,np.asarray(y,dtype=float),start,end)
        return float(np.trapz(yy,tt)/(end-start))
    force_on_controller = np.zeros(len(t))
    force_window_peak, rail_peaks, force_on_rate = 0., {}, {}
    servo = pd.read_csv(case/'raw_servo_state.csv.gz')
    for joint, group in servo.groupby('joint_name'):
        group=group.sort_values('sim_time_s').drop_duplicates('sim_time_s',keep='last')
        times=group.sim_time_s.to_numpy()-offset
        force=abs(group.force_N.to_numpy())
        force_on_controller=np.maximum(force_on_controller,np.interp(t,times,force))
        tt, ff=cut(times,force,start,end)
        force_window_peak=max(force_window_peak,float(ff.max()))
        rail_peaks[joint]={'position_m':float(cut(times,abs(group.position.to_numpy()),start,end)[1].max()),
                           'velocity_m_s':float(cut(times,abs(group.velocity.to_numpy()),start,end)[1].max()),
                           'force_N':float(ff.max())}
    # Actual servo limits from the physical implementation, not zero effort in JointState.
    force_mask=force_on_controller/2 >= .98
    plot=df.copy()
    plot['t_eval_s']=t-start
    plot['slider_pos_util']=rho_s
    plot['slider_vel_util']=rho_v
    plot['slider_force_util']=force_on_controller/2
    plot['position_error_m']=norm_ep
    plot['Tau_z_request_Nm']=df.Tau_z_request_raw
    plot['Tau_z_applied_Nm']=df.Tau_z_allocated
    plot['Tau_z_feasible_min_Nm']=df.Tau_z_feasible_min
    plot['Tau_z_feasible_max_Nm']=df.Tau_z_feasible_max
    plot[(t>=start)&(t<=end)].to_csv(out/'plotting.csv',index=False)
    imu=pd.read_csv(case/'raw_imu.csv.gz').sort_values('stamp_s').drop_duplicates('stamp_s',keep='last')
    it=imu.stamp_s.to_numpy()
    grid=np.arange(it[0],it[-1],.004)
    alpha=np.column_stack([savgol_filter(np.interp(grid,it,imu[k]),11,3,deriv=1,delta=.004,mode='interp') for k in ('p','q','r')])
    dyn={}
    m=(grid-offset>=start)&(grid-offset<=end)
    for j,name in enumerate(('p','q','r')):
        idx=np.where(m)[0][np.argmax(abs(alpha[m,j]))]
        dyn[name+'_dot']={'peak_abs_deg_s2':float(np.degrees(abs(alpha[idx,j]))),
                          'time_from_movement_start_s':float(grid[idx]-offset-start)}
    pd.DataFrame({'t_eval_s':grid[m]-offset-start,**{k+'_dot_deg_s2':np.degrees(alpha[m,j]) for j,k in enumerate(('p','q','r'))}}).to_csv(out/'angular_acceleration.csv',index=False)
    result={'data_origin':'Recorded coordinated translation and yaw experiment',
        'case':str(case),'regression_pass':runtime['regression_pass'],
        'window':{'controller_start_s':start,'controller_end_s':end,'duration_s':end-start},
        'clock_offset_s':offset,'clock_offset_range_s':[float(offset_values.min()),float(offset_values.max())],
        'main_window':{'position_rmse_m':rms(norm_ep),'position_peak_m':peak(norm_ep),
            'attitude_rmse_deg':float(np.degrees(rms(norm_ea))),
            'roll_peak_deg':peak(abs(df.Roll_deg)),'pitch_peak_deg':peak(abs(df.Pitch_deg)),
            'rho_s_peak':peak(rho_s),'rho_v_peak':peak(rho_v),
            'rho_Omega_peak':peak(df.Rotor_speed_util_actual),
            'rho_v_ge_098_time_fraction':fraction(rho_v>=.98),
            'yaw_request_clipped_time_fraction':fraction(clipped),
            'simultaneous_rate_boundary_and_yaw_clipping_fraction':fraction((rho_v>=.98)&clipped),
            'rail_force_peak_N':force_window_peak,'rail_peaks':rail_peaks,
            'tau_request_peak_Nm':peak(abs(df.Tau_z_request_raw)),
            'tau_clipping_peak_Nm':peak(abs(df.Tau_z_request_raw-df.Tau_z_request_clipped)),
            'angular_acceleration':dyn},
        'holding':runtime['tracking']['windows']['last_20_s'],
        'final_position_error_m':runtime['final_position_error_m'],
        'final_yaw_error_deg':runtime['tracking']['final_yaw_error_deg'],
        'method':{'rms':'Time-weighted trapezoidal integration with linearly interpolated exact window endpoints',
            'fractions':'Time-weighted binary activation flags, exact endpoints; clipping means request-clip >1e-5 N m',
            'rho_s_v':'Actual two paired-channel states; command and individual rail limits checked separately',
            'force':'Physical native SliderServo force telemetry; never JointState effort',
            'angular_acceleration':'Measured IMU body rates, uniform 4 ms interpolation, fixed 11-point cubic Savitzky-Golay derivative; p_dot/q_dot are body angular acceleration, not Euler second derivatives',
            'torque_plot':'MPC raw request versus allocator output; latter is not an independently measured torque',
            'time_alignment':'Median recorded Gazebo logging-clock minus controller Time; full offset range retained'},
        'provenance':{'script_sha256':digest(Path(__file__)),
            'raw_sha256':{p.name:digest(p) for p in (csv_path,case/'raw_servo_state.csv.gz',case/'raw_imu.csv.gz',case/'runtime_summary.json',case/'parameters.json')}}}
    assert all(np.isfinite(plot[k]).all() for k in ('t_eval_s','slider_pos_util','slider_vel_util','slider_force_util','Tau_z_request_Nm','Tau_z_applied_Nm'))
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    metadata={'evaluation_window_s':[0,15],'time_origin_s':0,
        'position_limit_m':.15,'velocity_limit_m_s':.25,'force_limit_N':2,
        'torque_output_meaning':'Allocated mixer target, not measured torque',
        'flight_fields':{},'replay_fields':{},'qa_fields':{},'plot_input_sha256':digest(out/'plotting.csv')}
    (out/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(result['main_window'],indent=2))


if __name__=='__main__':
    main()
