"""Common integration, bounded actuator response and explicit model omissions."""
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
from .dynamics import *
STATE_FIELDS = (
    "x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi",
    "p", "q", "r", "chi", "chi_d", "ups", "ups_d",
)

MODEL_NAMES = (
    "MF", "G1", "G2", "G3", "G4", "G4_projection", "G4_ac", "M0",
)

def state_vector(x: State6) -> np.ndarray:
    return np.asarray([float(getattr(x, name)) for name in STATE_FIELDS], dtype=float)

def clone_state(x: State6) -> State6:
    return State6(**{name: float(getattr(x, name)) for name in STATE_FIELDS})

def quintic(alpha: float) -> float:
    a = min(max(float(alpha), 0.0), 1.0)
    return a * a * a * (10.0 + a * (-15.0 + 6.0 * a))

def bounded_state(x: State6, P: Params) -> State6:
    """Project only the physical actuator state, with position/rate together."""
    y = clone_state(x)
    y.chi = float(np.clip(y.chi, -P.u2_lim, P.u2_lim))
    y.ups = float(np.clip(y.ups, -P.u2_lim, P.u2_lim))
    y.chi_d = float(np.clip(y.chi_d, -P.slider_vel_max, P.slider_vel_max))
    y.ups_d = float(np.clip(y.ups_d, -P.slider_vel_max, P.slider_vel_max))
    return y

def bounded_acceleration(position: float, velocity: float, command: float, P: Params, horizon: float = 0.0) -> Tuple[float, bool]:
    raw = P.wn_mass * P.wn_mass * (command - position) - 2.0 * P.zeta_mass * P.wn_mass * velocity
    vmax = float(P.slider_vel_max)
    at_upper = velocity >= vmax - 1.0e-10 and raw > 0.0
    at_lower = velocity <= -vmax + 1.0e-10 and raw < 0.0
    if at_upper or at_lower:
        return 0.0, True
    # Anticipate a crossing within the RK4 substep.  This makes the applied
    # acceleration consistent with the rate state instead of integrating an
    # unconstrained ddot and silently projecting only the final velocity.
    if horizon > 0.0:
        predicted = velocity + raw * horizon
        if predicted > vmax and raw > 0.0:
            return (vmax - velocity) / horizon, True
        if predicted < -vmax and raw < 0.0:
            return (-vmax - velocity) / horizon, True
    return float(raw), False

def nominal_inertia(P: Params):
    return cal_inertia(P, cal_sigma(P, 0.0, 0.0, 0.0, 0.0))

def _inertia_matrix(I) -> np.ndarray:
    return np.array([[I.Ixx, I.Ixy, 0.0], [I.Ixy, I.Iyy, 0.0], [0.0, 0.0, I.Izz]], dtype=float)

@dataclasses.dataclass
class DynamicsAux:
    chi_dd: float
    ups_dd: float
    chi_limited: bool
    ups_limited: bool
    cog_acc_body: np.ndarray
    inertia: np.ndarray
    inertia_dot: np.ndarray
    h_m: np.ndarray
    h_m_dot: np.ndarray
    tau_I: np.ndarray
    tau_mm: np.ndarray
    force_drag_world: np.ndarray
    torque_aero_body: np.ndarray
    thrust_projection_world: np.ndarray

def offline_rhs(
    P: Params,
    x: State6,
    u: Input6,
    model: str,
    psi0: float,
    *,
    bounded: bool,
    actuator_horizon: float = 0.0,
) -> Tuple[State6, DynamicsAux]:
    """Production-equation RHS plus named MMC model switches.

    The MF branch is algebraically the current ``update_6dof_model`` branch,
    including full aerodynamic loads.  G4 uses the explicitly frozen
    yaw-aware first-order vector ``Rz(psi0) [theta, -phi, 1]``.
    """
    if model not in MODEL_NAMES:
        raise ValueError(model)
    xx = bounded_state(x, P) if bounded else clone_state(x)
    chi_dd, chi_limited = bounded_acceleration(xx.chi, xx.chi_d, u.chi_cmd, P, actuator_horizon) if bounded else (
        P.wn_mass * P.wn_mass * (u.chi_cmd - xx.chi) - 2.0 * P.zeta_mass * P.wn_mass * xx.chi_d,
        False,
    )
    ups_dd, ups_limited = bounded_acceleration(xx.ups, xx.ups_d, u.ups_cmd, P, actuator_horizon) if bounded else (
        P.wn_mass * P.wn_mass * (u.ups_cmd - xx.ups) - 2.0 * P.zeta_mass * P.wn_mass * xx.ups_d,
        False,
    )

    loads = calculate_aerodynamic_loads(xx, P, np.zeros(3, dtype=float))
    Tb_v, _ = T_transformation(xx)
    omega = np.array([xx.p, xx.q, xx.r], dtype=float)
    S = cal_sigma(P, xx.chi, xx.ups, xx.chi_d, xx.ups_d)
    I_full_obj = cal_inertia(P, S)
    I_full = _inertia_matrix(I_full_obj)
    I0 = _inertia_matrix(nominal_inertia(P))
    I_dot = np.array([
        [I_full_obj.dIxx, I_full_obj.dIxy, 0.0],
        [I_full_obj.dIxy, I_full_obj.dIyy, 0.0],
        [0.0, 0.0, I_full_obj.dIzz],
    ])
    m, mu = P.m, P.mu
    h_m = np.array([0.0, 0.0, 4.0 * m * mu * (xx.ups * xx.chi_d - xx.chi * xx.ups_d)])
    h_m_dot = np.array([0.0, 0.0, 4.0 * m * mu * (xx.ups * chi_dd - xx.chi * ups_dd)])
    cog = 2.0 * mu * np.array([xx.chi, xx.ups, 0.0])
    cog_dot = 2.0 * mu * np.array([xx.chi_d, xx.ups_d, 0.0])
    cog_ddot = 2.0 * mu * np.array([chi_dd, ups_dd, 0.0])

    # G2 isolates the instantaneous-inertia/rigid-gyro path.  G3 removes the
    # remaining moving-mass dynamic terms, leaving its instantaneous inertia.
    use_I0 = model in {"G2", "M0"}
    remove_dynamic = model in {"G3", "M0"}
    I_used = I0 if use_I0 else I_full
    I_dot_used = np.zeros((3, 3)) if remove_dynamic else I_dot
    h_used = np.zeros(3) if remove_dynamic else h_m
    h_dot_used = np.zeros(3) if remove_dynamic else h_m_dot

    thrust_for_eccentric = P.M * P.g if model in {"G1", "M0"} else float(u.thrust)
    thrust_body = np.array([0.0, 0.0, float(u.thrust)])
    torque = -np.cross(cog, np.array([0.0, 0.0, thrust_for_eccentric]))
    torque += np.array([0.0, 0.0, float(u.tau_z)]) + loads.moment_cog_body
    tau_I = I_dot @ omega
    tau_mm = np.cross(omega, h_m) + h_m_dot
    angular_rhs = torque - I_dot_used @ omega - np.cross(omega, I_used @ omega + h_used) - h_dot_used
    try:
        omega_dot = np.linalg.solve(I_used, angular_rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive only
        raise RuntimeError(f"singular inertia in {model}") from exc

    ac_full = (cog_ddot + 2.0 * np.cross(omega, cog_dot)
               + np.cross(omega_dot, cog)
               + np.cross(omega, np.cross(omega, cog)))
    first_order = np.array([xx.theta, -xx.phi, 1.0], dtype=float)
    rz0 = np.array([[math.cos(psi0), -math.sin(psi0), 0.0],
                    [math.sin(psi0), math.cos(psi0), 0.0],
                    [0.0, 0.0, 1.0]])
    projection_exact = Tb_v @ np.array([0.0, 0.0, 1.0])
    use_projection_approx = model in {"G4", "G4_projection", "M0"}
    omit_ac = model in {"G4", "G4_ac", "M0"}
    projection_world = rz0 @ first_order if use_projection_approx else projection_exact
    ac_used = np.zeros(3) if omit_ac else ac_full
    acceleration = projection_world * (float(u.thrust) / P.M) - Tb_v @ ac_used
    acceleration += loads.force_world / P.M + np.array([0.0, 0.0, -P.g])

    phid, thetad, psid = attitude_kinematics(xx.p, xx.q, xx.r, xx.phi, xx.theta)
    dx = State6(
        x=xx.vx, y=xx.vy, z=xx.vz,
        vx=float(acceleration[0]), vy=float(acceleration[1]), vz=float(acceleration[2]),
        phi=phid, theta=thetad, psi=psid,
        p=float(omega_dot[0]), q=float(omega_dot[1]), r=float(omega_dot[2]),
        chi=xx.chi_d, chi_d=float(chi_dd), ups=xx.ups_d, ups_d=float(ups_dd),
    )
    aux = DynamicsAux(
        chi_dd=float(chi_dd), ups_dd=float(ups_dd),
        chi_limited=bool(chi_limited), ups_limited=bool(ups_limited),
        cog_acc_body=ac_full, inertia=I_full, inertia_dot=I_dot,
        h_m=h_m, h_m_dot=h_m_dot, tau_I=tau_I, tau_mm=tau_mm,
        force_drag_world=loads.force_world, torque_aero_body=loads.moment_cog_body,
        thrust_projection_world=projection_exact,
    )
    return dx, aux

def make_params(mu: float, wn: float) -> Params:
    M = 1.45
    m = float(mu) * M
    mb = M - 4.0 * m
    return Params(
        m_b=mb, m=m,
        Ib_x=0.0585 * mb / 1.25,
        Ib_y=0.0585 * mb / 1.25,
        Ib_z=0.06 * mb / 1.25,
        Im_x=0.0006 * m / 0.05,
        Im_y=0.0006 * m / 0.05,
        Im_z=0.0012 * m / 0.05,
        wn_mass=float(wn), zeta_mass=1.2,
        dt=0.001, T=0.35,
    )
