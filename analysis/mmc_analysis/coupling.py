"""Finite-horizon position/attitude error caused by removing each coupling term."""
from __future__ import annotations
import argparse,copy,csv,dataclasses,functools,gzip,hashlib,json,math,sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable,Dict,Iterable,List,Mapping,Optional,Sequence,Tuple
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[2]
from . import simulation as base
from .cases import Case
from .cad import make_cad_assembly,joint_coordinates,serializable
QA = ROOT / 'results/analysis_checks'

MODELS = base.MODEL_NAMES

DP, DA, HORIZON = .002, math.radians(.2), .35

MU = .05/1.45

def family_case(S, T):
    """One explicitly frozen map protocol: each coordinate uniquely sets a case."""
    k = S/.135
    return Case(key=f'S{S:.9f}_T{T:.9f}', family='unified_S_T', mu=MU,
                sx0=.75*S, sy0=S, sx1=-.25*S, sy1=.075*S,
                delay_y=.25*T, transition=T, thrust_scale1=1.1,
                phi0=-.15*k, theta0=.1*k, p0=1.5*k, q0=-.75*k)

def rotations(angles):
    phi, theta, psi = np.asarray(angles).T
    cr, sr, cp, sp, cy, sy = (np.cos(phi), np.sin(phi), np.cos(theta),
                             np.sin(theta), np.cos(psi), np.sin(psi))
    R = np.empty((len(phi), 3, 3))
    R[:, 0, :] = np.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr), axis=1)
    R[:, 1, :] = np.stack((sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), axis=1)
    R[:, 2, :] = np.stack((-sp, cp*sr, cp*cr), axis=1)
    return R

def rotation_errors(a, b):
    A, B = rotations(a[:, 6:9]), rotations(b[:, 6:9])
    # atan2 avoids losing relative-angle precision for almost identical models.
    Q = np.einsum('nji,njk->nik', A, B)
    sine = .5*np.linalg.norm(np.stack((Q[:, 2, 1]-Q[:, 1, 2],
                                      Q[:, 0, 2]-Q[:, 2, 0],
                                      Q[:, 1, 0]-Q[:, 0, 1]), axis=1), axis=1)
    cosine = np.clip((np.trace(Q, axis1=1, axis2=2)-1)/2, -1, 1)
    return np.arctan2(sine, cosine)

class Engine:
    def __init__(self, case, geometry='cad', aerodynamic=True):
        self.case = case
        self.P = base.make_params(case.mu, case.wn)
        self.P.u2_lim = .135
        self.cad = make_cad_assembly(case.mu)
        self.geometry = geometry
        self.aerodynamic = aerodynamic
        self.I0 = self.terms(np.zeros(4), np.zeros(2))['I']
        self.flags = {name: np.array([name in modes for modes in (
            {'G1', 'M0'}, {'G2', 'M0'}, {'G3', 'M0'},
            {'G4', 'G4_projection', 'M0'}, {'G4', 'G4_ac', 'M0'})])
            for name in MODELS}
        self.nominal = self.cad.link_states()
        self.slider_records = [r for r in self.nominal if r['joint_name'] in joint_coordinates(0, 0)]

    def terms(self, z, acc):
        sx, vx, sy, vy = z
        ax, ay = acc
        if self.geometry == 'cad':
            return self.cad.terms(sx, sy, vx, vy, ax, ay)
        P = self.P
        I = base.cal_inertia(P, base.cal_sigma(P, sx, sy, vx, vy))
        return dict(c=2*P.mu*np.array((sx, sy, 0)),
                    c_dot=2*P.mu*np.array((vx, vy, 0)),
                    c_ddot=2*P.mu*np.array((ax, ay, 0)),
                    I=base._inertia_matrix(I),
                    I_dot=np.array(((I.dIxx,I.dIxy,0),(I.dIxy,I.dIyy,0),(0,0,I.dIzz))),
                    h_m=np.array((0,0,4*P.m*P.mu*(sy*vx-sx*vy))),
                    h_m_dot=np.array((0,0,4*P.m*P.mu*(sy*ax-sx*ay))))

    def rhs(self, Y, u, horizon=0.):
        """All models share one actuator trajectory; only named balances differ."""
        P = self.P
        z = Y[0, 12:16]
        ax, _ = base.bounded_acceleration(z[0], z[1], u.chi_cmd, P, horizon)
        ay, _ = base.bounded_acceleration(z[2], z[3], u.ups_cmd, P, horizon)
        terms = self.terms(z, (ax, ay))
        c, cd, cdd = (terms[k] for k in ('c', 'c_dot', 'c_ddot'))
        R = rotations(Y[:, 6:9])
        omega = Y[:, 9:12]
        n = len(Y)
        body_velocity = np.einsum('nji,nj->ni', R, Y[:, 3:6])
        drag = -.5*P.rho*np.array((P.Cd_side*P.S_side, P.Cd_side*P.S_side, P.Cd_top*P.S_top))*body_velocity*np.abs(body_velocity)
        damping = -np.array((P.kp, P.kq, P.kr))*omega
        if not self.aerodynamic:
            drag[:] = 0; damping[:] = 0
        flags = np.array([self.flags[name] for name in MODELS[:n]])
        I = np.where(flags[:, 1, None, None], self.I0, terms['I'])
        Id = np.where(flags[:, 2, None, None], 0, terms['I_dot'])
        h = np.where(flags[:, 2, None], 0, terms['h_m'])
        hd = np.where(flags[:, 2, None], 0, terms['h_m_dot'])
        eccentric_thrust = np.where(flags[:, 0], P.M*P.g, u.thrust)
        F = np.zeros((n, 3)); F[:, 2] = eccentric_thrust
        torque = -np.cross(c, F) + damping - np.cross(c, drag)
        torque[:, 2] += u.tau_z
        rhs = torque - np.einsum('nij,nj->ni', Id, omega)
        rhs -= np.cross(omega, np.einsum('nij,nj->ni', I, omega)+h)+hd
        od = np.linalg.solve(I, rhs[..., None])[..., 0]
        ac = cdd+2*np.cross(omega, cd)+np.cross(od, c)+np.cross(omega, np.cross(omega, c))
        first = np.stack((Y[:, 7], -Y[:, 6], np.ones(n)), axis=1)
        psi = self.case.psi0
        Rz = np.array(((math.cos(psi),-math.sin(psi),0),(math.sin(psi),math.cos(psi),0),(0,0,1)))
        projection = np.where(flags[:, 3, None], first@Rz.T, R[:, :, 2])
        acceleration = projection*(u.thrust/P.M)
        acceleration += np.einsum('nij,nj->ni', R, drag)/P.M
        acceleration -= np.einsum('nij,nj->ni', R, np.where(flags[:, 4, None], 0, ac))
        acceleration[:, 2] -= P.g
        phi, theta = Y[:, 6], Y[:, 7]
        p, q, r = omega.T
        rates = np.stack((p+(r*np.cos(phi)+q*np.sin(phi))*np.tan(theta),
                          q*np.cos(phi)-r*np.sin(phi),
                          (r*np.cos(phi)+q*np.sin(phi))/np.cos(theta)), axis=1)
        dY = np.zeros_like(Y)
        dY[:, :3] = Y[:, 3:6]; dY[:, 3:6] = acceleration
        dY[:, 6:9] = rates; dY[:, 9:12] = od
        dY[:, 12:16] = (z[1], ax, z[3], ay)
        return dY, dict(R=R, acceleration=acceleration, alpha=od, omega=omega,
                         terms=terms, accelerations=(ax, ay), ac=ac)

    def project(self, Y):
        Y[:, (12,14)] = np.clip(Y[:, (12,14)], -.135, .135)
        Y[:, (13,15)] = np.clip(Y[:, (13,15)], -.25, .25)
        return Y

    def forces(self, Y, aux):
        q = joint_coordinates(Y[0,12], Y[0,14])
        qd = joint_coordinates(Y[0,13], Y[0,15])
        qdd = joint_coordinates(*aux['accelerations'])
        specific = aux['R'][0].T @ (aux['acceleration'][0]+np.array((0,0,self.P.g)))
        omega, alpha = aux['omega'][0], aux['alpha'][0]
        vals = []
        for record in self.slider_records:
            name, axis = record['joint_name'], record['joint_axis_body']
            p = record['p']+axis*q[name]
            absolute = specific+axis*qdd[name]+2*np.cross(omega,axis*qd[name])
            absolute += np.cross(alpha,p)+np.cross(omega,np.cross(omega,p))
            vals.append(record['mass']*(axis@absolute))
        return np.array(vals)

def simulate(case, dt=.001, geometry='cad', aerodynamic=True):
    engine = Engine(case, geometry, aerodynamic)
    n = int(round(HORIZON/dt)); dt = HORIZON/n
    t = np.linspace(0,HORIZON,n+1)
    Y = np.repeat(base.state_vector(case.initial())[None, :], len(MODELS), axis=0)
    history = np.empty((n+1, len(MODELS), 16)); history[0] = Y
    forces, inputs = [], []
    for k, tk in enumerate(t):
        u, inp = case.input(float(tk), engine.P)
        k1, aux = engine.rhs(Y, u, dt)
        forces.append(engine.forces(Y, aux)); inputs.append(inp)
        if k == n: break
        k2, _ = engine.rhs(engine.project(Y+.5*dt*k1), u, .5*dt)
        k3, _ = engine.rhs(engine.project(Y+.5*dt*k2), u, .5*dt)
        k4, _ = engine.rhs(engine.project(Y+dt*k3), u, dt)
        Y = engine.project(Y+dt*(k1+2*k2+2*k3+k4)/6)
        history[k+1] = Y
    return t, history, np.asarray(forces), inputs

def summarize(case, run, geometry, dt):
    t, history, forces, inputs = run
    full = history[:,0,:]
    row = dict(key=case.key, S_m=case.sy0, T_cmd_s=case.transition,
               geometry=geometry, dt_s=dt, horizon_s=HORIZON)
    indices = {}
    for i, name in enumerate(MODELS[1:],1):
        ep = float(np.max(np.linalg.norm(full[:,:3]-history[:,i,:3],axis=1)))
        ea = float(np.max(rotation_errors(full,history[:,i,:])))
        value = max(ep/DP,ea/DA)
        indices[name] = value
        row.update({f'{name}_J':value, f'{name}_Ep_m':ep, f'{name}_Ea_deg':math.degrees(ea)})
    J = [indices[f'G{i}'] for i in range(1,5)]
    B = max(indices['M0'],*J)
    row.update(B=B, weak=B<=1, significant=';'.join(f'G{i+1}' for i,j in enumerate(J) if j>1),
               dominant='G0' if B<=1 else f'G{int(np.argmax(J))+1}' if max(J)>1 else 'cumulative',
               cumulative=indices['M0']>1 and max(J)<=1,
               cancellation=indices['M0']<=1 and max(J)>1,
               peak_force_N=float(np.max(np.abs(forces))),
               peak_stroke_m=float(np.max(np.abs(full[:,(12,14)]))),
               peak_rate_m_s=float(np.max(np.abs(full[:,(13,15)]))),
               peak_attitude_deg=float(np.max(np.abs(full[:,6:8]))*180/math.pi),
               peak_omega_rad_s=float(np.max(np.linalg.norm(full[:,9:12],axis=1))))
    P = base.make_params(case.mu, case.wn)
    thrust = max(inp['thrust'] for inp in inputs)
    # Actual URDF thrust coefficients and ceiling, no clipped motor request.
    alloc = make_cad_assembly(case.mu).rotor_wrench_from_inputs(thrust,case.tau_z)
    row['peak_rotor_rad_s'] = max(alloc['motor_commands_rad_s'].values())
    row['resource_feasible'] = (row['peak_force_N']<=2+1e-8 and row['peak_rate_m_s']<=.25+1e-9
                                and row['peak_stroke_m']<=.135+1e-9 and alloc['feasible'])
    return row

def self_test():
    checks=[]
    case=family_case(.075,.15)
    engine=Engine(case,'ideal')
    rng=np.random.default_rng(3510)
    for k in range(12):
        x=base.State6(phi=rng.uniform(-.2,.2),theta=rng.uniform(-.2,.2),psi=rng.uniform(-.3,.3),
                      p=rng.uniform(-1,1),q=rng.uniform(-1,1),r=rng.uniform(-1,1),
                      vx=rng.uniform(-.5,.5),vy=rng.uniform(-.5,.5),vz=rng.uniform(-.5,.5),
                      chi=rng.uniform(-.1,.1),ups=rng.uniform(-.1,.1),
                      chi_d=rng.uniform(-.15,.15),ups_d=rng.uniform(-.15,.15))
        Y=np.repeat(base.state_vector(x)[None,:],len(MODELS),axis=0)
        u,_=case.input(.075,engine.P)
        actual,_=engine.rhs(Y,u,.001)
        expected=np.array([base.state_vector(base.offline_rhs(engine.P,x,u,m,case.psi0,
                                        bounded=True,actuator_horizon=.001)[0]) for m in MODELS])
        error=float(np.max(np.abs(actual-expected)))
        checks.append(dict(name=f'production_ideal_rhs_{k}',max_abs_error=error,passed=error<2e-12))
    t,Y,F,inp=simulate(case,.002)
    history_equal=float(np.max(np.abs(Y[:,:,12:16]-Y[:,0:1,12:16])))
    checks.append(dict(name='same_actuator_history_all_models',max_abs_error=history_equal,passed=history_equal==0))
    engine=Engine(case)
    for k in (0,20,80,150):
        u,_=case.input(float(t[k]),engine.P);_,aux=engine.rhs(Y[k],u,.002)
        direct=engine.cad.slider_required_forces(joint_coordinates(Y[k,0,12],Y[k,0,14]),
                    joint_coordinates(Y[k,0,13],Y[k,0,15]),joint_coordinates(*aux['accelerations']),
                    aux['acceleration'][0],aux['R'][0],aux['omega'][0],aux['alpha'][0])
        error=float(np.max(np.abs(engine.forces(Y[k],aux)-np.array(list(direct.values())))))
        checks.append(dict(name=f'four_link_complete_force_{k}',max_abs_error=error,passed=error<1e-12))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks)
    QA.mkdir(parents=True,exist_ok=True)
    (QA/'engine_self_test.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']: raise AssertionError(result)
    print(json.dumps(dict(passed=True,checks=len(checks))))
