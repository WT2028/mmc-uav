"""Idealized fixed-total-mass design relations and second-order step response."""
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
M_TOTAL = 1.45

M0 = 0.05

MB0 = 1.25

L = 0.40

G = 9.81

IB0_X = 0.0585

IB0_Y = 0.0585

IB0_Z = 0.0600

IM0_X = 0.0006

IM0_Y = 0.0006

IM0_Z = 0.0012

S_COMMAND = 0.135

S_PHYSICAL = 0.150

COG_SHOWCASE = 0.010

TASK_STEP = 0.050

WN_CASES = (10.0, 20.0, 30.0)

MU_LOWER = 1.0e-8

MU_UPPER = 0.24999999

def mass_family(mu: float) -> Dict[str, float]:
    """Return the fixed-total-mass analytical family at one mass ratio."""

    mu = float(mu)
    if not 0.0 <= mu < 0.25:
        raise ValueError(f"mu must be in [0, 0.25), got {mu}")
    m = mu * M_TOTAL
    mb = M_TOTAL - 4.0 * m
    return {
        "mu": mu,
        "m_kg": m,
        "mb_kg": mb,
        "M_kg": M_TOTAL,
        "Ib_x_kg_m2": IB0_X * mb / MB0,
        "Ib_y_kg_m2": IB0_Y * mb / MB0,
        "Ib_z_kg_m2": IB0_Z * mb / MB0,
        "Im_x_kg_m2": IM0_X * m / M0,
        "Im_y_kg_m2": IM0_Y * m / M0,
        "Im_z_kg_m2": IM0_Z * m / M0,
    }

def effective_inertia_axis(mu: float, s: float) -> float:
    """Effective roll/pitch inertia for a single displaced axis.

    For ``s_x=s`` and ``s_y=0`` this is ``I_yy`` from ``cal_inertia`` in
    ``mmc_uav.py``; for ``s_y=s`` and ``s_x=0`` it is the equal ``I_xx`` for
    the present symmetric nominal family.  The fixed-geometry parallel-axis
    contribution includes the ``s6`` term (the transverse half-span) and the
    ``s4`` term (the moving-mass contribution to the instantaneous CoG shift).
    With ``mb=M-4m`` those terms simplify to
    ``m*l^2/2 + (2*m - 4*m*mu)*s^2``.
    """

    family = mass_family(mu)
    m = family["m_kg"]
    mb = family["mb_kg"]
    position_term = (
        m * L * L / 2.0
        + 2.0 * m * s * s
        + 16.0 * m * mu * mu * s * s
        + 4.0 * mb * mu * mu * s * s
        - 8.0 * m * mu * s * s
    )
    return (
        family["Ib_y_kg_m2"]
        + 4.0 * family["Im_y_kg_m2"]
        + position_term
    )

def effective_inertia_axis_simplified(mu: float, s: float) -> float:
    """The fixed-total-mass simplification used as an independent check."""

    family = mass_family(mu)
    m = family["m_kg"]
    return (
        family["Ib_y_kg_m2"]
        + 4.0 * family["Im_y_kg_m2"]
        + m * L * L / 2.0
        + (2.0 * m - 4.0 * m * mu) * s * s
    )

def hover_torque(mu: float, s: float) -> float:
    """Magnitude of the local eccentric-thrust torque at hover thrust Mg."""

    return 2.0 * mu * M_TOTAL * G * abs(float(s))

def angular_acceleration_capacity(mu: float, s: float) -> float:
    return hover_torque(mu, s) / effective_inertia_axis(mu, abs(float(s)))

def initial_cog_extra_acceleration(mu: float, wn: float, delta_s: float = TASK_STEP) -> float:
    """Initial ``|a_c^0(0+)|`` for a zero-state position step."""

    return 2.0 * mu * wn * wn * abs(delta_s)

def _bisect_root(function: Callable[[float], float], left: float, right: float) -> float:
    f_left = float(function(left))
    f_right = float(function(right))
    if abs(f_left) <= 1.0e-15:
        return left
    if abs(f_right) <= 1.0e-15:
        return right
    if f_left * f_right > 0.0:
        raise ValueError(f"root is not bracketed: f({left})={f_left}, f({right})={f_right}")
    for _ in range(100):
        middle = 0.5 * (left + right)
        f_middle = float(function(middle))
        if abs(f_middle) < 1.0e-14 or abs(right - left) < 1.0e-13:
            return middle
        if f_left * f_middle <= 0.0:
            right, f_right = middle, f_middle
        else:
            left, f_left = middle, f_middle
    return 0.5 * (left + right)

def step_response_dimensionless(tau: float, zeta: float) -> float:
    """Exact unit-step position response at ``tau=wn*t``."""

    tau = max(0.0, float(tau))
    zeta = float(zeta)
    if zeta < 0.0:
        raise ValueError("zeta must be nonnegative")
    if zeta < 1.0 - 1.0e-10:
        beta = math.sqrt(max(0.0, 1.0 - zeta * zeta))
        if beta == 0.0:
            return 1.0 - (1.0 + tau) * math.exp(-tau)
        return 1.0 - math.exp(-zeta * tau) * (
            math.cos(beta * tau) + zeta / beta * math.sin(beta * tau)
        )
    if abs(zeta - 1.0) <= 1.0e-10:
        return 1.0 - (1.0 + tau) * math.exp(-tau)
    root = math.sqrt(zeta * zeta - 1.0)
    slow = zeta - root
    fast = zeta + root
    return 1.0 - (
        fast * math.exp(-slow * tau) - slow * math.exp(-fast * tau)
    ) / (fast - slow)

def step_velocity_dimensionless(tau: float, zeta: float) -> float:
    """Exact derivative ``d(position/step)/d(wn*tau)``."""

    tau = max(0.0, float(tau))
    zeta = float(zeta)
    if zeta < 0.0:
        raise ValueError("zeta must be nonnegative")
    if zeta < 1.0 - 1.0e-10:
        beta = math.sqrt(max(0.0, 1.0 - zeta * zeta))
        return math.exp(-zeta * tau) * math.sin(beta * tau) / beta
    if abs(zeta - 1.0) <= 1.0e-10:
        return tau * math.exp(-tau)
    root = math.sqrt(zeta * zeta - 1.0)
    slow = zeta - root
    fast = zeta + root
    return (math.exp(-slow * tau) - math.exp(-fast * tau)) / (fast - slow)

def _t90_scan_limit(zeta: float) -> float:
    if zeta > 1.0:
        slow = zeta - math.sqrt(zeta * zeta - 1.0)
        return max(20.0, 10.0 / max(slow, 1.0e-12))
    return max(20.0, 20.0 / max(zeta, 0.05))

def t90_dimensionless(zeta: float) -> float:
    """First upward crossing of 90% from the exact general-zeta response."""

    limit = _t90_scan_limit(zeta)
    samples = 50000
    tau_prev = 0.0
    value_prev = step_response_dimensionless(0.0, zeta) - 0.9
    for index in range(1, samples + 1):
        tau_now = limit * index / samples
        value_now = step_response_dimensionless(tau_now, zeta) - 0.9
        if value_now >= 0.0 and value_prev < 0.0:
            return _bisect_root(
                lambda tau: step_response_dimensionless(tau, zeta) - 0.9,
                tau_prev,
                tau_now,
            )
        tau_prev, value_prev = tau_now, value_now
    raise RuntimeError(f"could not find 90% crossing for zeta={zeta}")

def peak_velocity_dimensionless(zeta: float) -> Tuple[float, float]:
    """Return ``(tau_peak, max absolute dimensionless velocity)``."""

    if zeta < 0.0:
        raise ValueError("zeta must be nonnegative")
    if zeta < 1.0 - 1.0e-10:
        beta = math.sqrt(max(0.0, 1.0 - zeta * zeta))
        tau_peak = math.atan2(beta, zeta) / beta
    elif abs(zeta - 1.0) <= 1.0e-10:
        tau_peak = 1.0
    else:
        root = math.sqrt(zeta * zeta - 1.0)
        slow = zeta - root
        fast = zeta + root
        tau_peak = math.log(fast / slow) / (fast - slow)
    return tau_peak, abs(step_velocity_dimensionless(tau_peak, zeta))

def _rhs(s: float, v: float, command: float, wn: float, zeta: float, vmax: float | None) -> Tuple[float, float]:
    acceleration = wn * wn * (command - s) - 2.0 * zeta * wn * v
    if vmax is not None:
        if v >= vmax and acceleration > 0.0:
            acceleration = 0.0
        elif v <= -vmax and acceleration < 0.0:
            acceleration = 0.0
    return v, acceleration

def _rk4_step(
    s: float,
    v: float,
    dt: float,
    command: float,
    wn: float,
    zeta: float,
    vmax: float | None,
) -> Tuple[float, float]:
    k1s, k1v = _rhs(s, v, command, wn, zeta, vmax)
    k2s, k2v = _rhs(s + 0.5 * dt * k1s, v + 0.5 * dt * k1v, command, wn, zeta, vmax)
    k3s, k3v = _rhs(s + 0.5 * dt * k2s, v + 0.5 * dt * k2v, command, wn, zeta, vmax)
    k4s, k4v = _rhs(s + dt * k3s, v + dt * k3v, command, wn, zeta, vmax)
    next_s = s + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0
    next_v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    if vmax is not None:
        next_v = max(-vmax, min(vmax, next_v))
    return next_s, next_v

def integrate_step(
    wn: float,
    zeta: float,
    command: float = TASK_STEP,
    duration: float = 1.2,
    dt: float = 1.0e-4,
    vmax: float | None = None,
) -> List[Dict[str, float]]:
    """Fixed-step RK4 response; ``vmax`` enables the diagnostic speed limit."""

    samples: List[Dict[str, float]] = []
    s = 0.0
    v = 0.0
    count = int(round(duration / dt))
    for index in range(count + 1):
        t = index * dt
        _, acceleration = _rhs(s, v, command, wn, zeta, vmax)
        samples.append(
            {
                "time_s": t,
                "position_m": s,
                "velocity_m_s": v,
                "acceleration_m_s2": acceleration,
                "speed_limit_active": 1.0 if vmax is not None and abs(v) >= vmax - 1.0e-9 else 0.0,
            }
        )
        if index < count:
            s, v = _rk4_step(s, v, dt, command, wn, zeta, vmax)
    return samples

def first_crossing(samples: Sequence[Dict[str, float]], fraction: float, command: float) -> float:
    target = abs(command) * fraction
    previous = samples[0]
    previous_error = abs(previous["position_m"]) - target
    if previous_error >= 0.0:
        return previous["time_s"]
    for current in samples[1:]:
        current_error = abs(current["position_m"]) - target
        if current_error >= 0.0:
            span = current["time_s"] - previous["time_s"]
            if abs(current_error - previous_error) < 1.0e-15:
                return current["time_s"]
            ratio = -previous_error / (current_error - previous_error)
            return previous["time_s"] + ratio * span
        previous, previous_error = current, current_error
    raise RuntimeError("no response crossing in numerical trajectory")

def authority_summary(mu: float, alpha_required: float) -> Dict[str, float]:
    family = mass_family(mu)
    row: Dict[str, float] = {
        "mu": mu,
        "m_kg": family["m_kg"],
        "mb_kg": family["mb_kg"],
        "r_c_command_m": 2.0 * mu * S_COMMAND,
        "r_c_physical_m": 2.0 * mu * S_PHYSICAL,
        "s_req_for_10mm_m": COG_SHOWCASE / (2.0 * mu),
        "I_axis_at_zero_kg_m2": effective_inertia_axis(mu, 0.0),
        "I_axis_at_task_step_kg_m2": effective_inertia_axis(mu, TASK_STEP),
        "I_axis_at_command_kg_m2": effective_inertia_axis(mu, S_COMMAND),
        "I_axis_at_physical_kg_m2": effective_inertia_axis(mu, S_PHYSICAL),
        "tau_task_step_Nm": hover_torque(mu, TASK_STEP),
        "alpha_task_step_rad_s2": angular_acceleration_capacity(mu, TASK_STEP),
        "tau_command_Nm": hover_torque(mu, S_COMMAND),
        "tau_physical_Nm": hover_torque(mu, S_PHYSICAL),
        "alpha_command_rad_s2": angular_acceleration_capacity(mu, S_COMMAND),
        "alpha_physical_rad_s2": angular_acceleration_capacity(mu, S_PHYSICAL),
        "authority_required_rad_s2": alpha_required,
        "authority_margin_command_Nm": hover_torque(mu, S_COMMAND) - alpha_required * effective_inertia_axis(mu, S_COMMAND),
        "authority_margin_physical_Nm": hover_torque(mu, S_PHYSICAL) - alpha_required * effective_inertia_axis(mu, S_PHYSICAL),
    }
    for wn in WN_CASES:
        row[f"initial_extra_accel_{int(wn)}rad_s_m_s2"] = initial_cog_extra_acceleration(mu, wn)
    return row

def boundary_rows(
    boundary_id: str,
    condition: str,
    mu_boundary: float,
    s_allow: float,
    alpha_required: float,
) -> List[Dict[str, object]]:
    delta = max(1.0e-5, 0.01 * mu_boundary)
    values = [
        ("below_1pct", max(MU_LOWER, mu_boundary - delta)),
        ("at", mu_boundary),
        ("above_1pct", min(MU_UPPER, mu_boundary + delta)),
    ]
    rows: List[Dict[str, object]] = []
    for side, mu in values:
        family = mass_family(mu)
        inertia = effective_inertia_axis(mu, s_allow)
        tau = hover_torque(mu, s_allow)
        margin = tau - alpha_required * inertia
        rows.append(
            {
                "boundary_id": boundary_id,
                "condition": condition,
                "side": side,
                "s_allow_m": s_allow,
                "mu": mu,
                "m_kg": family["m_kg"],
                "mb_kg": family["mb_kg"],
                "r_c_m": 2.0 * mu * s_allow,
                "r_c_goal_margin_m": 2.0 * mu * s_allow - COG_SHOWCASE,
                "I_axis_kg_m2": inertia,
                "tau_capacity_Nm": tau,
                "alpha_capacity_rad_s2": tau / inertia,
                "authority_margin_Nm": margin,
                "geometry_pass": (2.0 * mu * s_allow >= COG_SHOWCASE),
                "authority_pass": (margin >= 0.0),
                "initial_extra_accel_10rad_s_m_s2": initial_cog_extra_acceleration(mu, 10.0),
                "initial_extra_accel_20rad_s_m_s2": initial_cog_extra_acceleration(mu, 20.0),
                "initial_extra_accel_30rad_s_m_s2": initial_cog_extra_acceleration(mu, 30.0),
            }
        )
    return rows
