#!/usr/bin/env python3
"""Replay marked, measured CAD probes without hiding MF/plant residuals.

The full and seven omission models share all four observed rail histories and
the recorded body-origin wrench. Each starts from the same first post-initial
Physics body state. The full model is independently compared with subsequent
Physics observations; omission sensitivity is separately measured against MF.
This conditional replay is not a force-servo or closed-loop flight validation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from . import coupling
from .cad import CADAssembly, sha256, write_csv, write_json

SCRIPT = Path(__file__).resolve()
MODELS = coupling.MODELS
DP, DA, H = coupling.DP, coupling.DA, coupling.HORIZON
JOINTS = ("joint_slider_green", "joint_slider_purple", "joint_slider_blue", "joint_slider_red",
          "joint_rotor_upper", "joint_rotor_lower")
FIELDS = ("x", "y", "z", "vx_world", "vy_world", "vz_world", "phi", "theta", "psi", "p", "q", "r")
FLAGS = np.array([[name in omit for omit in (
    {"G1", "M0"}, {"G2", "M0"}, {"G3", "M0"},
    {"G4", "G4_projection", "M0"}, {"G4", "G4_ac", "M0"})] for name in MODELS])


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def series(rows, fields):
    samples = {float(r["stamp_s"]): [float(r[f]) for f in fields] for r in rows}
    t = np.array(sorted(samples))
    v = np.array([samples[s] for s in t])
    if len(t) < 3 or not np.isfinite(v).all():
        raise ValueError("Insufficient or nonfinite physical observations")
    return t, v


def interpolate(t, values, at):
    a = np.asarray(at)
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(a, t, col) for col in flat.T], axis=-1)
    return result.reshape(a.shape+shape)


class PhysicalHistory:
    def __init__(self, case_dir, derivative="central"):
        self.folder = Path(case_dir)
        self.protocol = json.loads((self.folder/"protocol.json").read_text())
        self.runtime = json.loads((self.folder/"runtime_summary.json").read_text())
        if not self.runtime["completed"] or self.runtime["initialization_marker_count"] != 1:
            raise ValueError("Physical initialization or acquisition did not complete")
        if not self.runtime["default_urdf_unchanged"]:
            raise ValueError("The default model changed during acquisition")
        for name in ("raw_physical_body.csv", "raw_joint_states.csv", "raw_wrench.csv", "aligned_source.urdf"):
            expected = self.runtime["files_sha256"][name]
            if sha256(self.folder/name) != expected:
                raise ValueError("Immutable physical source changed: " + name)
        self.cad = CADAssembly(self.folder/"aligned_source.urdf")
        bt, bv = series(read(self.folder/"raw_physical_body.csv"),
                        ("x", "y", "z", "qx", "qy", "qz", "qw", "vx_world", "vy_world", "vz_world", "p", "q", "r"))
        angles = np.unwrap(Rotation.from_quat(bv[:,3:7]).as_euler("xyz"), axis=0)
        self.body_t = bt
        self.body_y = np.column_stack((bv[:,:3], bv[:,7:10], angles, bv[:,10:13]))
        rows = read(self.folder/"raw_joint_states.csv")
        self.joints = {name: series([r for r in rows if r["joint_name"] == name], ("position", "velocity")) for name in JOINTS}
        wt, wv = series(read(self.folder/"raw_wrench.csv"),
                        ("thrust_N", "force_body_x_N", "force_body_y_N", "force_body_z_N",
                         "moment_body_x_Nm", "moment_body_y_Nm", "moment_body_z_Nm"))
        # PreUpdate t is the end stamp of the physical interval (t-dt,t].
        self.physics_dt = .001
        self.wrench_t, self.wrench_v = wt-self.physics_dt, wv
        start = max(bt[0], self.wrench_t[0], *(t[0] for t,v in self.joints.values()))
        end = min(bt[-1], self.wrench_t[-1], *(t[-1] for t,v in self.joints.values()))
        marker = float(read(self.folder/"initialization_markers.csv")[0]["stamp_s"])
        if abs(start-marker) > 1.5*self.physics_dt:
            raise ValueError("The first matched state does not immediately follow the marked initialization")
        self.start, self.end = start, end
        self.grid = np.arange(start, end+.5*self.physics_dt, self.physics_dt)
        j = np.stack([interpolate(t,v,self.grid) for t,v in self.joints.values()], axis=1)
        self.q, self.qd = j[:,:,0], j[:,:,1]
        if derivative == "central":
            self.qdd = np.gradient(self.qd, self.physics_dt, axis=0, edge_order=2)
        elif derivative == "backward":
            self.qdd = np.vstack(((self.qd[1]-self.qd[0])/self.physics_dt, np.diff(self.qd,axis=0)/self.physics_dt))
        else:
            raise ValueError("Unsupported derivative method")
        self.derivative = derivative
        terms = [self.cad.terms_from_joints(dict(zip(JOINTS,q)), dict(zip(JOINTS,qd)), dict(zip(JOINTS,qdd)))
                 for q,qd,qdd in zip(self.q,self.qd,self.qdd)]
        self.term_values = {key: np.array([r[key] for r in terms]) for key in terms[0]}
        self.I0 = self.cad.terms_from_joints()["I"]
        gaps = {"body":float(np.max(np.diff(bt))), "wrench":float(np.max(np.diff(wt))),
                **{name:float(np.max(np.diff(t))) for name,(t,v) in self.joints.items()}}
        if max(gaps.values()) > .00201:
            raise ValueError("Physical sampling gap exceeds two 1 ms steps: " + str(gaps))
        rotor_max = float(np.max(np.abs(j[:,4:,:])))
        if rotor_max > 1e-10:
            raise ValueError("Locked-rotor protocol contains rotor motion")
        self.qa = dict(max_sampling_gaps_s=gaps, measured_start_s=start, initialization_marker_s=marker,
                       initial_state12=self.body(start).tolist(), rotor_peak_abs_position_or_rate=rotor_max,
                       derivative_method=derivative, physics_dt_s=self.physics_dt,
                       peak_measured_stroke_m=float(np.max(np.abs(self.q[:,:4]))),
                       peak_measured_rate_m_s=float(np.max(np.abs(self.qd[:,:4]))))

    def body(self, at):
        return interpolate(self.body_t, self.body_y, at)

    def inputs(self, at):
        terms = {key:interpolate(self.grid,values,at) for key,values in self.term_values.items()}
        # The actual applied force is held over each physical integration step.
        index = min(len(self.wrench_t)-1, max(0, int(np.searchsorted(self.wrench_t,at+1e-10,side="right")-1)))
        return terms, self.wrench_v[index]


def rhs(Y, terms, wrench, cad, I0, psi0, aerodynamic="recorded"):
    n = len(Y)
    flags = FLAGS[:n]
    c, cd, cdd = (terms[key] for key in ("c", "c_dot", "c_ddot"))
    R = coupling.rotations(Y[:,6:9])
    omega = Y[:,9:12]
    I = np.where(flags[:,1,None,None], I0, terms["I"])
    Id = np.where(flags[:,2,None,None], 0., terms["I_dot"])
    h = np.where(flags[:,2,None], 0., terms["h_m"])
    hd = np.where(flags[:,2,None], 0., terms["h_m_dot"])
    thrust = wrench[0]
    if aerodynamic == "nominal":
        vb=np.einsum("nji,nj->ni",R,Y[:,3:6])
        drag=-.5*1.225*np.array((1.2*.072,1.2*.072,1.3*.16))*vb*np.abs(vb)
        moment=-np.array((.04,.04,.02))*omega
        F=drag+np.array((0.,0.,thrust))
    else:
        F=np.broadcast_to(wrench[1:4],(n,3))
        moment=np.broadcast_to(wrench[4:7],(n,3))
        drag=F-np.array((0.,0.,thrust))
    eccentric = F.copy()
    eccentric[:,2] -= np.where(flags[:,0], thrust-cad.M*9.81, 0.)
    torque = moment-np.cross(c,eccentric)
    angular_rhs = torque-np.einsum("nij,nj->ni",Id,omega)
    angular_rhs -= np.cross(omega,np.einsum("nij,nj->ni",I,omega)+h)+hd
    alpha = np.linalg.solve(I,angular_rhs[...,None])[...,0]
    ac = cdd+2*np.cross(omega,cd)+np.cross(alpha,c)+np.cross(omega,np.cross(omega,c))
    first = np.stack((Y[:,7],-Y[:,6],np.ones(n)),axis=1)
    Rz = Rotation.from_euler("z",psi0).as_matrix()
    projection = np.where(flags[:,3,None], first@Rz.T, R[:,:,2])
    acceleration = projection*thrust/cad.M+np.einsum("nij,nj->ni",R,drag)/cad.M
    acceleration -= np.einsum("nij,nj->ni",R,np.where(flags[:,4,None],0.,ac))
    acceleration[:,2] -= 9.81
    phi,theta = Y[:,6],Y[:,7]
    p,q,r = omega.T
    rates = np.stack((p+(r*np.cos(phi)+q*np.sin(phi))*np.tan(theta),
                      q*np.cos(phi)-r*np.sin(phi),
                      (r*np.cos(phi)+q*np.sin(phi))/np.cos(theta)),axis=1)
    result = np.column_stack((Y[:,3:6],acceleration,rates,alpha))
    return result


def replay(history, dt=.001, horizon=H, start=None, aerodynamic="recorded"):
    start = history.start if start is None else start
    if start+horizon > history.end+1e-10:
        raise ValueError("Incomplete physical window")
    if aerodynamic=="nominal" and not history.protocol["enable_nominal_aero"]:
        raise ValueError("Nominal-aero map replay requires a matching aero-enabled physical probe")
    count = int(round(horizon/dt))
    dt = horizon/count
    t = start+np.arange(count+1)*dt
    initial = history.body(start)
    # The G4 projection keeps the protocol's frozen heading reference. The
    # measured post-initial yaw still initializes every propagated model.
    psi0 = history.protocol["psi_initial_rad"]
    Y = np.repeat(initial[None,:],len(MODELS),axis=0)
    result = np.empty((count+1,len(MODELS),12)); result[0]=Y
    for k, tk in enumerate(t[:-1]):
        terms, force = history.inputs(tk)
        middle, _ = history.inputs(tk+.5*dt)
        end, _ = history.inputs(tk+dt)
        # Keep the sampled physical wrench over this whole 1 ms interval.
        k1 = rhs(Y,terms,force,history.cad,history.I0,psi0,aerodynamic)
        k2 = rhs(Y+.5*dt*k1,middle,force,history.cad,history.I0,psi0,aerodynamic)
        k3 = rhs(Y+.5*dt*k2,middle,force,history.cad,history.I0,psi0,aerodynamic)
        k4 = rhs(Y+dt*k3,end,force,history.cad,history.I0,psi0,aerodynamic)
        Y = Y+dt*(k1+2*k2+2*k3+k4)/6
        result[k+1]=Y
    return t,result,history.body(t)


def errors(reference, prediction):
    ep = float(np.max(np.linalg.norm(reference[:,:3]-prediction[:,:3],axis=1)))
    ea = float(np.max(coupling.rotation_errors(reference,prediction)))
    return dict(Ep_m=ep,Ea_deg=math.degrees(ea),J=max(ep/DP,ea/DA))


def physical_geometry(history, t):
    """Independent mass/inertia sums from all actual Physics link poses."""
    path=history.folder/"raw_physical_links.csv"
    if not path.exists():
        return dict(performed=False,reason="Preliminary probe predates seven-link telemetry")
    rows=read(path)
    plant=ET.parse(history.folder/"probe_world.sdf").getroot().find("world/model")
    observed=history.body(t)
    R=coupling.rotations(observed[:,6:9])
    M=0.; centers=[]; inertia_world=[]; masses=[]
    for link in plant.findall("link"):
        inertial=link.find("inertial")
        mass=float(inertial.findtext("mass"));M+=mass;masses.append(mass)
        pose=np.fromstring(inertial.findtext("pose","0 0 0 0 0 0"),sep=" ")
        com=pose[:3];Ri=Rotation.from_euler("xyz",pose[3:]).as_matrix()
        node=inertial.find("inertia")
        xx,yy,zz,xy,xz,yz=[float(node.findtext(k)) for k in ("ixx","iyy","izz","ixy","ixz","iyz")]
        own=Ri@np.array(((xx,xy,xz),(xy,yy,yz),(xz,yz,zz)))@Ri.T
        lt,lv=series([r for r in rows if r["link_name"]==link.get("name")],("x","y","z","qx","qy","qz","qw"))
        value=interpolate(lt,lv,t)
        RL=Rotation.from_quat(value[:,3:7]).as_matrix()
        centers.append(value[:,:3]+np.einsum("nij,j->ni",RL,com))
        inertia_world.append(RL@own@np.transpose(RL,(0,2,1)))
    cg=sum(m*p for m,p in zip(masses,centers))/M
    I=np.zeros((len(t),3,3))
    for m,p,own in zip(masses,centers,inertia_world):
        rel=p-cg
        I+=own+m*(np.einsum("ni,ni->n",rel,rel)[:,None,None]*np.eye(3)-np.einsum("ni,nj->nij",rel,rel))
    c_body=np.einsum("nji,nj->ni",R,cg-observed[:,:3])
    I_body=np.transpose(R,(0,2,1))@I@R
    cad_c=interpolate(history.grid,history.term_values["c"],t)
    cad_I=interpolate(history.grid,history.term_values["I"],t)
    ce=float(np.max(np.linalg.norm(c_body-cad_c,axis=1)))
    ie=float(np.max(np.abs(I_body-cad_I)))
    return dict(performed=True,total_mass_kg=M,link_count=len(masses),
                first_measured_cog_body_m=c_body[0].tolist(),
                first_measured_inertia_cog_body_kg_m2=I_body[0].tolist(),
                peak_CoG_mapping_error_m=ce,peak_inertia_mapping_error_kg_m2=ie,
                geometry_alignment_pass=ce<1e-9 and ie<1e-10,
                measurement="All seven observed world link poses with inertial mass/own-I read from the actual probe_world.sdf; independent of joint-coordinate CoG/I calculation")


def physical_resources(history,t):
    rows=[]
    force_by_joint={}
    body_t,body_acc=series(read(history.folder/"raw_physical_body.csv"),
                         ("ax_world","ay_world","az_world","alpha_body_x","alpha_body_y","alpha_body_z"))
    observed=history.body(t);R=coupling.rotations(observed[:,6:9]);acc=interpolate(body_t,body_acc,t)
    q=interpolate(history.grid,history.q,t);qd=interpolate(history.grid,history.qd,t)
    # Interval acceleration from observed velocity increments, independent of
    # the smoother central derivative used for the continuous replay.
    qdd_all=np.vstack(((history.qd[1]-history.qd[0])/history.physics_dt,
                      np.diff(history.qd,axis=0)/history.physics_dt))
    qdd=interpolate(history.grid,qdd_all,t)
    demands=np.array([list(history.cad.slider_required_forces(dict(zip(JOINTS,qi)),dict(zip(JOINTS,vi)),
                     dict(zip(JOINTS,ai)),bw,Ri,omega,alpha).values())
                     for qi,vi,ai,bw,Ri,omega,alpha in zip(q,qd,qdd,acc[:,:3],R,observed[:,9:12],acc[:,3:6])])
    for color in ("green","purple","blue","red"):
        measured=read(history.folder/f"raw_servo_{color}.csv")
        selected=[r for r in measured if t[0]-1e-10<=float(r["stamp_s"])<=t[-1]+1e-10]
        force_by_joint["joint_slider_"+color]=max(abs(float(r["applied_force_N"])) for r in selected)
    peak=max(force_by_joint.values())
    stroke=float(np.max(np.abs(q[:,:4])))
    rate=float(np.max(np.abs(qd[:,:4])))
    return dict(peak_applied_rail_force_N=peak,peak_applied_force_by_joint_N=force_by_joint,
                peak_complete_inverse_rail_demand_N=float(np.max(np.abs(demands))),
                peak_actual_rate_m_s=rate,peak_actual_stroke_m=stroke,
                resource_window_s=float(t[-1]-t[0]),
                acquisition_peak_actual_rate_m_s=history.qa["peak_measured_rate_m_s"],
                acquisition_peak_actual_stroke_m=history.qa["peak_measured_stroke_m"],
                command_stroke_limit_m=.135,mechanical_stroke_limit_m=.15,
                actual_command_envelope_overshoot_m=max(0.,stroke-.135),
                force_limit_N=2.,rate_limit_m_s=.25,
                force_and_mechanical_limits_pass=peak<=2.+1e-9 and stroke<=.15+1e-9,
                rate_excess_m_s=max(0.,rate-.25),
                torque_requirement=history.protocol.get("task_alpha_requirement_rad_s2"))


def summarize(history, run, dt,aerodynamic="recorded"):
    t,Y,plant = run
    residual = errors(plant,Y[:,0])
    omissions = {name:errors(Y[:,0],Y[:,i]) for i,name in enumerate(MODELS) if i}
    J = [omissions[f"G{i}"]["J"] for i in range(1,5)]
    B = max(omissions["M0"]["J"],*J)
    return dict(key=history.protocol.get("key",history.folder.name), dt_s=dt,horizon_s=H,
                start_stamp_s=float(t[0]),end_stamp_s=float(t[-1]),
                frozen_projection_heading_rad=history.protocol["psi_initial_rad"],
                MF_plant=residual, MF_plant_budget_pass=residual["J"]<=1.,
                omissions_vs_MF=omissions,B=B,weak=B<=1.,
                significant=[f"G{i+1}" for i,j in enumerate(J) if j>1.],
                dominant="G0" if B<=1 else f"G{int(np.argmax(J))+1}" if max(J)>1 else "cumulative",
                cumulative=omissions["M0"]["J"]>1 and max(J)<=1,
                cancellation=omissions["M0"]["J"]<=1 and max(J)>1,
                model_plant_residual_independent=True,
                aerodynamic_policy=aerodynamic,
                conditional_history=("All four actual rail positions/rates/acceleration derivatives and measured body-origin wrench shared by all models; raw plant pose/velocity only conditions the initial state." if aerodynamic=="recorded" else "All four actual rail histories and recorded applied thrust shared by all models; unchanged nominal drag/rotational damping evaluated from each model state, matching the map definition. Plant body history only conditions the initial state."),
                experimental_scope="Locked rotors, prescribed base-origin wrench; internal CAD mass coupling, not closed-loop flight or rotor dynamics",
                initialization=history.qa,
                physical_geometry=physical_geometry(history,t),
                physical_resources=physical_resources(history,t),
                provenance=dict(protocol_sha256=sha256(history.folder/"protocol.json"),
                                plant_urdf_sha256=sha256(history.folder/"aligned_source.urdf"),
                                replay_sha256=sha256(SCRIPT),
                                alignment_sha256=sha256(SCRIPT.with_name("cad.py")),
                                coupling_engine_sha256=sha256(Path(coupling.__file__))),
                tolerance=dict(position_m=DP,attitude_deg=math.degrees(DA)))


def self_test():
    from .cases import Case
    case = coupling.family_case(.075,.15)
    engine = coupling.Engine(case)
    u,_ = case.input(.075,engine.P)
    Y = np.repeat(coupling.base.state_vector(case.initial())[None,:],len(MODELS),axis=0)
    expected, aux = engine.rhs(Y,u,.001)
    terms = aux["terms"]
    # All states here are identical, so the original engine's state-dependent
    # aerodynamic load is exactly the same single observed body wrench.
    R = aux["R"][0]; vb=R.T@Y[0,3:6]
    P=engine.P
    drag = -.5*P.rho*np.array((P.Cd_side*P.S_side,P.Cd_side*P.S_side,P.Cd_top*P.S_top))*vb*np.abs(vb)
    moment = -np.array((P.kp,P.kq,P.kr))*Y[0,9:12]
    actual = rhs(Y[:,:12],terms,np.r_[u.thrust,drag+np.array((0,0,u.thrust)),moment],engine.cad,engine.I0,case.psi0)
    error = float(np.max(np.abs(actual-expected[:,:12])))
    if error > 2e-12: raise AssertionError(error)
    rng=np.random.default_rng(351023)
    Y[:,3:12]+=rng.uniform(-.025,.025,size=(len(Y),9))
    expected,aux=engine.rhs(Y,u,.001)
    actual=rhs(Y[:,:12],aux["terms"],np.r_[u.thrust,0,0,u.thrust,0,0,0],engine.cad,engine.I0,case.psi0,"nominal")
    nominal_error=float(np.max(np.abs(actual-expected[:,:12])))
    if nominal_error>2e-12:raise AssertionError(nominal_error)
    return dict(shared_wrench_matches_root_engine_identical_states=True,max_abs_rhs_error=error,
                map_consistent_aero_matches_root_engine_distinct_model_states=True,
                max_abs_nominal_rhs_error=nominal_error)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir",type=Path)
    ap.add_argument("--dt",type=float,default=.001)
    ap.add_argument("--derivative",choices=("central","backward"),default="central")
    ap.add_argument("--output-dir",type=Path)
    ap.add_argument("--aerodynamic-policy",choices=("recorded","nominal"),default="recorded")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        if args.case_dir is None: return
    if args.case_dir is None: ap.error("--case-dir required")
    history=PhysicalHistory(args.case_dir,args.derivative)
    run=replay(history,args.dt,aerodynamic=args.aerodynamic_policy)
    result=summarize(history,run,args.dt,args.aerodynamic_policy)
    out=args.output_dir or args.case_dir/("replay_"+args.aerodynamic_policy+"_"+args.derivative+"_dt"+str(args.dt).replace(".","p"))
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"summary.json",result)
    t,Y,plant=run
    with gzip.open(out/"states.csv.gz","wt",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=("stamp_s","phase_s","model",*FIELDS));writer.writeheader()
        for k,stamp in enumerate(t):
            for i,name in enumerate(MODELS):
                writer.writerow(dict(stamp_s=stamp,phase_s=stamp-t[0],model=name,**dict(zip(FIELDS,Y[k,i]))))
            writer.writerow(dict(stamp_s=stamp,phase_s=stamp-t[0],model="plant",**dict(zip(FIELDS,plant[k]))))
    print(json.dumps({k:result[k] for k in ("key","MF_plant","MF_plant_budget_pass","B","significant")}))


if __name__=="__main__":main()
