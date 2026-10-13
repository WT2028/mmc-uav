#!/usr/bin/env python3
"""Matched measured-input CAD diagnostics on recorded coordinated flights.

Only the initial body observation conditions a propagated model. Every model
shares all six measured joint histories. The actual rotating rotor geometry,
physical joint angular momentum and CAD force points remain explicit. Nominal
zero-wind drag and damping are evaluated from each propagated state. Published
ApplyLinkWrench pulses are archived inputs, never held as continuous forces.
No controller, URDF, simulation or manuscript is modified by this program.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
from . import coupling, cad, simulation
from .cad import CADAssembly,sha256,serializable

def existing(folder, name):
    p = Path(folder)/name
    return p if p.exists() else p.with_name(p.name+'.gz')

MODELS = coupling.MODELS
DP, DA, H = coupling.DP, coupling.DA, coupling.HORIZON
JOINTS = ("joint_slider_green", "joint_slider_purple", "joint_slider_blue", "joint_slider_red",
          "joint_rotor_upper", "joint_rotor_lower")
FIELDS = ("x", "y", "z", "vx_world", "vy_world", "vz_world", "phi", "theta", "psi", "p", "q", "r")
FLAGS = np.array([[name in omit for omit in (
    {"G1", "M0"}, {"G2", "M0"}, {"G3", "M0"},
    {"G4", "G4_projection", "M0"}, {"G4", "G4_ac", "M0"})] for name in MODELS])


def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        yield from csv.DictReader(stream)


def series(records, fields, stamp="stamp_s"):
    data = {float(r[stamp]): [float(r[k]) for k in fields] for r in records}
    t = np.array(sorted(data))
    values = np.array([data[x] for x in t])
    if len(t) < 3 or not np.isfinite(values).all() or not np.isfinite(t).all():
        raise ValueError("Insufficient or nonfinite observed history: " + str(fields))
    return t, values


def interpolate(t, values, at):
    at = np.asarray(at)
    if at.min() < t[0]-1e-9 or at.max() > t[-1]+1e-9:
        raise ValueError("Requested observation lies outside measured support")
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(at, t, col) for col in flat.T], axis=-1)
    return result.reshape(at.shape+shape)


def write_json(path, data):
    Path(path).write_text(json.dumps(serializable(data), indent=2, allow_nan=False) + "\n")


class Flight:
    def __init__(self, folder, body_rates="auto"):
        self.folder = Path(folder).resolve()
        models = list(self.folder.glob("effective_*.urdf"))
        if len(models) != 1:
            raise ValueError("Require one preserved effective flight URDF")
        self.urdf = models[0]
        self.cad = CADAssembly(self.urdf)
        self.I0 = self.cad.terms_from_joints()["I"]
        self.raw_files = [self.urdf, existing(self.folder,"raw_odometry.csv"), existing(self.folder,"raw_joint_states.csv")]
        self.odom_t, odom = series(rows(existing(self.folder,"raw_odometry.csv")),
            ("x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "p", "q", "r"))
        frames = {(r["frame_id"], r["child_frame_id"]) for r in rows(existing(self.folder,"raw_odometry.csv"))}
        if frames != {("mmc_world", "base_link")}:
            raise ValueError("Unestablished body-origin/frame transform: " + str(frames))
        config = next(p for p in self.cad.root.findall("gazebo/plugin") if "OdometryPublisher" in p.get("name", ""))
        if any(config.find(k) is not None for k in ("xyz_offset", "rpy_offset")):
            raise ValueError("Odometry offsets require an explicit transform")
        rotation = Rotation.from_quat(odom[:, 3:7])
        self.body_y = np.column_stack((odom[:, :3], rotation.apply(odom[:, 7:10]),
            np.unwrap(rotation.as_euler("xyz"), axis=0), odom[:, 10:13]))
        self.rate_t, self.rate_v = self.odom_t, odom[:, 10:13]
        self.rate_source = "Gazebo odometry rolling-mean body twist"
        imu_path = existing(self.folder,"raw_imu.csv")
        if body_rates != "odom" and imu_path.exists():
            self.rate_t, self.rate_v = series(rows(imu_path), ("p", "q", "r"))
            self.raw_files.append(imu_path)
            self.rate_source = "Measured IMU angular velocity; mounting rotation is zero"
            imu = next(s for s in self.cad.root.findall("gazebo/sensor") if s.get("type") == "imu")
            mount = np.fromstring(imu.findtext("pose", "0 0 0 0 0 0"), sep=" ")
            if len(mount) != 6 or not np.allclose(mount[3:], 0, atol=1e-12):
                raise ValueError("IMU mounting rotation requires a transform")
        elif body_rates == "imu":
            raise ValueError("IMU body rates requested but not observed")
        grouped = {name: {} for name in JOINTS}
        for r in rows(existing(self.folder,"raw_joint_states.csv")):
            if r["joint_name"] in grouped:
                grouped[r["joint_name"]][float(r["stamp_s"])] = (float(r["position"]), float(r["velocity"]))
        self.joints = {}
        for name, data in grouped.items():
            t = np.array(sorted(data)); value = np.array([data[x] for x in t])
            if len(t) < 3 or not np.isfinite(value).all():
                raise ValueError("Missing/nonfinite actual joint " + name)
            self.joints[name] = (t, value)
        del grouped
        logs = list(self.folder.glob("mmc_flight_log*.csv*"))
        if len(logs) != 1:
            raise ValueError("Require one controller CSV to establish task clock")
        self.controller = list(rows(logs[0])); self.raw_files.append(logs[0])
        offsets = [float(r["Joint_feedback_clock_s"])-float(r["Time"]) for r in self.controller]
        self.clock_offset = float(np.median(offsets))
        ros_parameters = self.folder/"ros_parameters.yaml"
        self.move_start = 8.0
        if ros_parameters.exists():
            import yaml
            dumped = yaml.safe_load(ros_parameters.read_text())
            node = next(k for k in dumped if k.rstrip("/").endswith("mmc_uav_controller"))
            p = dumped[node]["ros__parameters"]
            self.move_start = float(p["auto_scene_takeoff_transition_time"])+float(p["auto_scene_hover_hold_time"])
            self.raw_files.append(ros_parameters)
        wind = self.folder/"raw_wind.csv"
        if wind.exists():
            wt, wv = series(rows(wind), ("vx", "vy", "vz"), "sim_time_s")
            if np.max(np.linalg.norm(wv, axis=1)) > 1e-10:
                raise ValueError("This diagnostic is declared for zero-wind flights")
            self.raw_files.append(wind)
        for name in ("parameters.json", "runtime_summary.json", "raw_servo_state.csv", "raw_published_wrench.csv"):
            path = self.folder/name
            if path.exists(): self.raw_files.append(path)
        self.start = max(self.odom_t[0], self.rate_t[0], *(s[0][0] for s in self.joints.values()))
        self.end = min(self.odom_t[-1], self.rate_t[-1], *(s[0][-1] for s in self.joints.values()))
        self.qa = dict(body_rate_source=self.rate_source, clock_offset_s=self.clock_offset,
            clock_offset_max_abs_deviation_s=float(np.max(np.abs(np.asarray(offsets)-self.clock_offset))),
            move_start_controller_s=self.move_start,
            odometry_twist="Child-frame linear velocity transformed into world ENU; measured rolling-mean estimate, not engine instantaneous velocity",
            rotors="Measured joint geometry/physical angular speed for rigid-body terms; slowdown factor used only for aerodynamic motor thrust",
            maximum_sampling_gap_s=dict(odometry=float(np.diff(self.odom_t).max()), body_rates=float(np.diff(self.rate_t).max()),
                **{name: float(np.diff(t).max()) for name, (t,v) in self.joints.items()}))

    def body(self, at):
        result = interpolate(self.odom_t, self.body_y, at)
        result[..., 9:12] = interpolate(self.rate_t, self.rate_v, at)
        return result

    def prepare(self, start, dt=.001, derivative="central"):
        pad = .010
        a, b = max(self.start, start-pad), min(self.end, start+H+pad)
        if a > start or b < start+H:
            raise ValueError("Incomplete measured support for prediction window")
        self.grid = np.linspace(a, b, int(math.ceil((b-a)/dt))+1)
        spacing = self.grid[1]-self.grid[0]
        value = np.stack([interpolate(t,v,self.grid) for t,v in self.joints.values()], axis=1)
        self.q, self.qd = value[:, :, 0], value[:, :, 1]
        if derivative == "central":
            qdd = np.gradient(self.qd, spacing, axis=0, edge_order=2)
        elif derivative == "backward":
            qdd = np.vstack(((self.qd[1]-self.qd[0])/spacing, np.diff(self.qd,axis=0)/spacing))
        else: raise ValueError("Unsupported joint acceleration derivative")
        terms, moving, forces, moments = [], [], [], []
        for q,qd,qa in zip(self.q,self.qd,qdd):
            positions, velocities, accelerations = [dict(zip(JOINTS,v)) for v in (q,qd,qa)]
            full = self.cad.terms_from_joints(positions,velocities,accelerations)
            mv = self.cad.terms_from_joints(positions,
                {k:v for k,v in velocities.items() if k.startswith("joint_slider_")},
                {k:v for k,v in accelerations.items() if k.startswith("joint_slider_")})
            wrench = self.cad.rotor_wrench_from_speeds(velocities,positions,full["c"])
            terms.append(full); moving.append(mv); forces.append(wrench["force_body"]); moments.append(wrench["moment_body_origin"])
        self.terms = {k: np.array([v[k] for v in terms]) for k in terms[0]}
        # G3 omits the moving-mass derivative/internal momentum while retaining
        # actual rotor spin. The full-minus-held-rotor difference is explicit.
        self.rotor_terms = {k: self.terms[k]-np.array([v[k] for v in moving]) for k in ("I_dot","h_m","h_m_dot")}
        self.rotor_force = np.array(forces); self.rotor_moment = np.array(moments)
        self.input_qa = dict(derivative=derivative, interpolation_dt_s=float(spacing),
            measured_peak_joint_speed_rad_s=float(np.abs(self.qd[:,4:]).max()),
            measured_peak_slider_position_m=float(np.abs(self.q[:,:4]).max()),
            measured_peak_slider_speed_m_s=float(np.abs(self.qd[:,:4]).max()),
            peak_rigid_rotor_h_norm=float(np.linalg.norm(self.rotor_terms["h_m"],axis=1).max()),
            peak_rigid_rotor_hdot_norm=float(np.linalg.norm(self.rotor_terms["h_m_dot"],axis=1).max()))

    def inputs(self, at):
        terms = {k: interpolate(self.grid,v,at) for k,v in self.terms.items()}
        rotor = {k: interpolate(self.grid,v,at) for k,v in self.rotor_terms.items()}
        return terms, rotor, interpolate(self.grid,self.rotor_force,at), interpolate(self.grid,self.rotor_moment,at)


def rhs(Y, terms, rotor, force, moment, cad, I0, psi0, flags=FLAGS):
    flags = flags[:len(Y)]
    c,cd,cdd = (terms[k] for k in ("c","c_dot","c_ddot"))
    R = coupling.rotations(Y[:,6:9]); omega = Y[:,9:12]
    vb = np.einsum("nji,nj->ni",R,Y[:,3:6])
    drag = -.5*1.225*np.array((1.2*.072,1.2*.072,1.3*.16))*vb*np.abs(vb)
    damping = -np.array((.04,.04,.02))*omega
    I = np.where(flags[:,1,None,None],I0,terms["I"])
    Id,h,hd = [np.where(flags[:,2].reshape((-1,)+(1,)*np.asarray(terms[k]).ndim),rotor[k],terms[k])
               for k in ("I_dot","h_m","h_m_dot")]
    eccentric_force = np.broadcast_to(force,(len(Y),3)).copy()
    eccentric_force[:,2] -= np.where(flags[:,0],force[2]-cad.M*9.81,0.)
    torque = moment+damping-np.cross(c,eccentric_force+drag)
    balance = torque-np.einsum("nij,nj->ni",Id,omega)
    balance -= np.cross(omega,np.einsum("nij,nj->ni",I,omega)+h)+hd
    alpha = np.linalg.solve(I,balance[...,None])[...,0]
    ac = cdd+2*np.cross(omega,cd)+np.cross(alpha,c)+np.cross(omega,np.cross(omega,c))
    first = np.stack((Y[:,7],-Y[:,6],np.ones(len(Y))),axis=1)
    Rz = Rotation.from_euler("z",psi0).as_matrix()
    projection = np.where(flags[:,3,None],first@Rz.T,R[:,:,2])
    acceleration = projection*force[2]/cad.M
    other_force = np.broadcast_to(force,(len(Y),3)).copy();other_force[:,2]=0.
    acceleration += np.einsum("nij,nj->ni",R,drag+other_force)/cad.M
    acceleration -= np.einsum("nij,nj->ni",R,np.where(flags[:,4,None],0.,ac))
    acceleration[:,2] -= 9.81
    phi,theta = Y[:,6],Y[:,7];p,q,r = omega.T
    rates = np.stack((p+(r*np.cos(phi)+q*np.sin(phi))*np.tan(theta),
                     q*np.cos(phi)-r*np.sin(phi),(r*np.cos(phi)+q*np.sin(phi))/np.cos(theta)),axis=1)
    return np.column_stack((Y[:,3:6],acceleration,rates,alpha))


def replay(flight,start,dt=.001,derivative="central"):
    flight.prepare(start,dt,derivative)
    count=int(round(H/dt)); step=H/count;t=start+np.arange(count+1)*step
    initial=flight.body(start);psi0=float(initial[8])
    Y=np.repeat(initial[None,:],len(MODELS),axis=0);history=np.empty((count+1,len(MODELS),12));history[0]=Y
    for k,tk in enumerate(t[:-1]):
        beginning=flight.inputs(tk);middle=flight.inputs(tk+.5*step);end=flight.inputs(tk+step)
        k1=rhs(Y,*beginning,flight.cad,flight.I0,psi0)
        k2=rhs(Y+.5*step*k1,*middle,flight.cad,flight.I0,psi0)
        k3=rhs(Y+.5*step*k2,*middle,flight.cad,flight.I0,psi0)
        k4=rhs(Y+step*k3,*end,flight.cad,flight.I0,psi0)
        Y+=step*(k1+2*k2+2*k3+k4)/6;history[k+1]=Y
    return t,history,flight.body(t)


def errors(a,b):
    ep=float(np.linalg.norm(a[:,:3]-b[:,:3],axis=1).max())
    ea=float(coupling.rotation_errors(a,b).max())
    return dict(position_peak_m=ep,rotation_peak_deg=math.degrees(ea),J=max(ep/DP,ea/DA))


def summarize(flight,run):
    t,Y,plant=run
    residual=errors(plant,Y[:,0]);omissions={name:errors(Y[:,0],Y[:,i]) for i,name in enumerate(MODELS) if i}
    J=[omissions[f"G{i}"]["J"] for i in range(1,5)];B=max(omissions["M0"]["J"],*J)
    return dict(start_sim_time_s=float(t[0]),start_controller_time_s=float(t[0]-flight.clock_offset),
        end_sim_time_s=float(t[-1]),frozen_projection_yaw_rad=float(plant[0,8]),MF_plant=residual,
        MF_plant_budget_pass=residual["J"]<=1.,omissions_vs_MF=omissions,B=B,weak=B<=1.,
        significant=[f"G{i+1}" for i,j in enumerate(J) if j>1],
        dominant="G0" if B<=1 else f"G{np.argmax(J)+1}" if max(J)>1 else "cumulative",
        cumulative=omissions["M0"]["J"]>1 and max(J)<=1,
        cancellation=omissions["M0"]["J"]<=1 and max(J)>1,
        input_sampling=flight.input_qa,
        prediction_evidence_eligible=residual["J"]<=1.,
        interpretation="Matched measured-input omission diagnostic, not a different-vehicle closed-loop ablation or an exact location in the prescribed S-Tcmd map slice")


def self_test():
    case=coupling.family_case(.075,.15);engine=coupling.Engine(case)
    Y=np.repeat(coupling.base.state_vector(case.initial())[None,:],len(MODELS),axis=0)
    rng=np.random.default_rng(2805);Y[:,3:12]+=rng.uniform(-.025,.025,(len(Y),9))
    u,_=case.input(.075,engine.P);expected,aux=engine.rhs(Y,u,.001)
    actual=rhs(Y[:,:12],aux["terms"],{k:np.zeros_like(aux["terms"][k]) for k in ("I_dot","h_m","h_m_dot")},
        np.array((0.,0.,u.thrust)),np.array((0.,0.,u.tau_z)),engine.cad,engine.I0,case.psi0)
    error=float(np.abs(expected[:,:12]-actual).max())
    if error>2e-12:raise AssertionError(("CAD nominal CAD RHS differs",error))
    # Independent finite differences check all six rotating/prismatic coordinates.
    q=dict(zip(JOINTS,(.013,.011,-.02,-.019,.8,-1.4)))
    v=dict(zip(JOINTS,(.05,.04,-.025,-.021,30.,-28.)))
    a=dict(zip(JOINTS,(.2,.15,-.1,-.09,.4,-.3)))
    terms=engine.cad.terms_from_joints(q,v,a);eps=1e-6
    sides=[]
    for sign in (-1,1):
        dt=sign*eps
        sides.append(engine.cad.terms_from_joints({k:q[k]+dt*v[k]+.5*dt**2*a[k] for k in q},
            {k:v[k]+dt*a[k] for k in v},a))
    derivative_errors={name:float(np.abs((sides[1][name]-sides[0][name])/(2*eps)-terms[target]).max())
        for name,target in (("c","c_dot"),("I","I_dot"),("h_m","h_m_dot"))}
    if max(derivative_errors.values())>2e-9:raise AssertionError(derivative_errors)
    return dict(passed=True,locked_rotor_CAD_equivalence_max_abs_error=error,
        six_actual_joint_mass_property_derivative_errors=derivative_errors,
        projection_heading="Frozen separately to each window's initial observed yaw",
        no_observed_future_body_state_enters_rhs=True,
        physical_rotor_slowdown_not_applied_to_rigid_joint_momentum=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--starts-controller-s",type=float,nargs="+")
    parser.add_argument("--stride",type=float,default=1.)
    parser.add_argument("--dt",type=float,default=.001)
    parser.add_argument("--derivative",choices=("central","backward"),default="central")
    parser.add_argument("--body-rates",choices=("auto","odom","imu"),default="auto")
    args=parser.parse_args()
    qa=self_test()
    if args.self_test and args.case_dir is None:
        out=ROOT/"results/coordination_self_test";out.mkdir(parents=True,exist_ok=True)
        write_json(out/"self_test.json",qa);print(json.dumps(qa));return
    if args.case_dir is None:parser.error("--case-dir required")
    flight=Flight(args.case_dir,args.body_rates)
    if args.dt<=0 or args.stride<=0:parser.error("Positive step and stride required")
    out=args.output_dir or ROOT/"results/coordination_replay"/flight.folder.name
    out.mkdir(parents=True,exist_ok=False)
    source_paths=(SCRIPT,Path(coupling.__file__),Path(cad.__file__),Path(simulation.__file__))
    snapshot=out/"source_snapshot";snapshot.mkdir()
    for path in source_paths:(snapshot/path.name).write_bytes(path.read_bytes())
    starts=args.starts_controller_s or np.arange(flight.move_start,flight.move_start+15-H,args.stride)
    results=[];failed=[]
    with gzip.open(out/"states.csv.gz","wt",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=("window","stamp_s","phase_s","model",*FIELDS));writer.writeheader()
        for i,start in enumerate(starts):
            run=replay(flight,float(start)+flight.clock_offset,args.dt,args.derivative)
            result=summarize(flight,run);result["window"]=i;results.append(result)
            if not result["MF_plant_budget_pass"]:failed.append(i)
            t,Y,plant=run
            for k,stamp in enumerate(t):
                for j,name in enumerate(MODELS):
                    writer.writerow(dict(window=i,stamp_s=stamp,phase_s=stamp-t[0],model=name,**dict(zip(FIELDS,Y[k,j]))))
                writer.writerow(dict(window=i,stamp_s=stamp,phase_s=stamp-t[0],model="plant",**dict(zip(FIELDS,plant[k]))))
            print(json.dumps(dict(window=i,controller_start_s=float(start),MF_plant=result["MF_plant"],significant=result["significant"])),flush=True)
    result=dict(case=str(flight.folder),data_origin="CAD analysis of recorded coordinated flight",
        horizon_s=H,budgets=dict(position_m=DP,rotation_deg=math.degrees(DA)),dt_s=args.dt,
        derivative=args.derivative,aerodynamic_policy="Unchanged zero-wind nominal law evaluated from each model's own propagated body state",
        published_wrench_policy="Published 1 ms pulse is not held or treated as measured continuous applied load",
        measured_history="All four physical rail positions/rates and both rotor positions/rates; measured rates differentiated; actual rotor force points and rigid spin retained",
        body_future_use="Used only for independent error comparison, never to update a propagated body state",
        G3_policy="Omit moving-mass derivative/internal momentum, retain independent physical rotor-spin contributions",
        self_test=qa,acquisition=flight.qa,windows=results,failed_windows=failed,
        independently_eligible_windows=[i for i,r in enumerate(results) if r["prediction_evidence_eligible"]],
        provenance=dict(raw_sha256={str(p):sha256(p) for p in flight.raw_files},
            source_sha256={str(p):sha256(p) for p in source_paths},
            command=vars(args)))
    write_json(out/"summary.json",result)
    with (out/"windows.csv").open("w",newline="") as stream:
        fields=("window","t_eval_s","window_start_controller_s","window_end_controller_s",
                "MF_plant_position_peak_m","MF_plant_rotation_peak_deg","MF_plant_budget_pass",
                "J_G1","J_G2","J_G3","J_G4","J_M0","B","weak","dominant")
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow(dict(window=r["window"],t_eval_s=r["start_controller_time_s"]-flight.move_start,
                window_start_controller_s=r["start_controller_time_s"],window_end_controller_s=r["start_controller_time_s"]+H,
                MF_plant_position_peak_m=r["MF_plant"]["position_peak_m"],MF_plant_rotation_peak_deg=r["MF_plant"]["rotation_peak_deg"],
                MF_plant_budget_pass=r["MF_plant_budget_pass"],**{f"J_{name}":r["omissions_vs_MF"][name]["J"] for name in ("G1","G2","G3","G4","M0")},
                B=r["B"],weak=r["weak"],dominant=r["dominant"]))
    print(json.dumps(dict(output=str(out),windows=len(results),eligible=result["independently_eligible_windows"],failed=failed)))


if __name__=="__main__":main()
