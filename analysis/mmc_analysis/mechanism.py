"""Positioning requirements with force, speed and travel limits."""
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
from . import analytical as legacy
DESIGN = {
    'M_kg': 1.45, 'mu0': .05 / 1.45, 'g_m_s2': 9.81,
    'attitude_step_deg': 5.0, 'attitude_build_s': 1.0,
    'cog_target_m': .010, 'fixed_actuator_step_m': .050,
    'actuator_t90_requirement_s': .350, 'zeta': 1.2,
    'velocity_limit_m_s': .250, 'force_limit_N': 2.0,
    'command_limit_m': .135, 'mechanical_limit_m': .150,
    'load_observer_tau_s': .020, 'reserve_torque_Nm': 0.0,
    'gravity_in_probe_m_s2': 0.0, 'rotor_commands_in_probe': False,
    'boundary_bracket_fraction': .02,
}

def attitude_task():
    angle = math.radians(DESIGN['attitude_step_deg'])
    duration = DESIGN['attitude_build_s']
    return {'angle_rad': angle, 'duration_s': duration,
            'profile': 'eta(t)=Delta_eta*(10*u^3-15*u^4+6*u^5), u=t/T',
            'peak_rate_rad_s': 15 / 8 * angle / duration,
            'peak_acceleration_rad_s2': 10 * math.sqrt(3) / 3 * angle / duration**2,
            'peak_acceleration_normalized_times': [(3 - math.sqrt(3)) / 6, (3 + math.sqrt(3)) / 6],
            'basis': 'Prescribed 5 deg / 1 s smooth attitude-building design example, independent of nominal mass capacity',
            'tracking_guarantee': False}

def crossing(times, positions, target, q0=0.0, fraction=.9):
    distance = target - q0
    if abs(distance) < 1e-15:
        return 0.0
    previous = fraction * abs(distance)
    for index, (t, q) in enumerate(zip(times, positions)):
        error = fraction * abs(distance) - (q - q0) * math.copysign(1, distance)
        if error <= 0:
            if index == 0:
                return float(t)
            weight = previous / (previous - error)
            return float(times[index - 1] + weight * (t - times[index - 1]))
        previous = error
    return None

def constrained_step(wn, mass, requested, *, total_mass=DESIGN['M_kg'],
                     active_per_axis=2, free_body=False, dt=.0005,
                     duration=1.5, q0=0.0, v0=0.0):
    """Apply the installed force/speed/stopping law without resetting states.

    The free-body scalar surrogate uses m_eff=m*(1-n*m/M). The measured-load
    observer has the same 0.02 s time constant as SliderServo. CAD geometry,
    angular recoil and full vehicle loads belong to the separately run plant.
    """
    if not (wn > 0 and mass > 0 and dt > 0):
        raise ValueError('Positive natural frequency, mass and timestep required')
    effective_mass = mass * (1 - active_per_axis * mass / total_mass) if free_body else mass
    if effective_mass <= 0:
        raise ValueError('Nonpositive free-body equivalent mass')
    vmax, fmax = DESIGN['velocity_limit_m_s'], DESIGN['force_limit_N']
    slimit, zeta = DESIGN['command_limit_m'], DESIGN['zeta']
    target = max(-slimit, min(slimit, requested))
    q, v, load = float(q0), float(v0), 0.0
    previous_v, previous_force, previous_valid = v, 0.0, False
    rows = []
    for i in range(round(duration / dt) + 1):
        if previous_valid:
            residual = (v - previous_v) / dt - previous_force / mass
            load += dt / (DESIGN['load_observer_tau_s'] + dt) * (residual - load)
            load = max(-fmax / mass, min(fmax / mass, load))
        ideal_acceleration = wn**2 * (target - q) - 2 * zeta * wn * v
        amax = fmax / mass
        positive_speed = min(vmax, math.sqrt(2 * amax * max(0, slimit - q)))
        negative_speed = min(vmax, math.sqrt(2 * amax * max(0, slimit + q)))
        desired_velocity = v + ideal_acceleration * dt
        next_velocity_request = max(-negative_speed, min(positive_speed, desired_velocity))
        acceleration_request = (next_velocity_request - v) / dt
        force_request = mass * (acceleration_request - load)
        force = max(-fmax, min(fmax, force_request))
        actual_acceleration = force / effective_mass
        rows.append({'time_s': i * dt, 'position_m': q, 'velocity_m_s': v,
                     'requested_m': requested, 'effective_command_m': target,
                     'applied_force_N': force, 'force_request_N': force_request,
                     'load_acceleration_m_s2': load,
                     'desired_acceleration_m_s2': acceleration_request,
                     'actual_acceleration_m_s2': actual_acceleration,
                     'force_limit_active': abs(force_request) > fmax + 1e-10,
                     'speed_limit_active': abs(desired_velocity) > vmax + 1e-10,
                     'stopping_constraint_active': abs(desired_velocity) > (positive_speed if desired_velocity >= 0 else negative_speed) + 1e-10
                                                   and abs(desired_velocity) <= vmax + 1e-10})
        previous_v, previous_force, previous_valid = v, force, True
        v += actual_acceleration * dt
        q += v * dt
    return rows

def response_metrics(rows, requested):
    times = [r['time_s'] for r in rows]
    positions = [r['position_m'] for r in rows]
    h = times[1] - times[0]
    t90 = crossing(times, positions, requested)
    peak_position = max(abs(q) for q in positions)
    peak_speed = max(abs(r['velocity_m_s']) for r in rows)
    peak_force = max(abs(r['applied_force_N']) for r in rows)
    # A force/speed command limiter does not authorize hiding a state violation.
    # In a free body, observer lag can produce a real speed excess. It is
    # reported and excluded from the strictly bounded diagnostic feasible set.
    position_ok = peak_position <= DESIGN['command_limit_m'] + 1e-9
    velocity_ok = peak_speed <= DESIGN['velocity_limit_m_s'] + 1e-9
    force_ok = peak_force <= DESIGN['force_limit_N'] + 1e-9
    return {'t90_requested_s': t90,
            't90_effective_command_s': crossing(times, positions, rows[0]['effective_command_m']),
            't90_requirement_met': t90 is not None and t90 <= DESIGN['actuator_t90_requirement_s'],
            'peak_position_m': peak_position,
            'peak_speed_m_s': peak_speed,
            'peak_force_N': peak_force,
            'command_travel_bound_met': position_ok,
            'actual_speed_bound_met': velocity_ok,
            'force_bound_met': force_ok,
            'all_resource_bounds_met': position_ok and velocity_ok and force_ok,
            'speed_bound_excess_m_s': max(0.0, peak_speed - DESIGN['velocity_limit_m_s']),
            'force_limit_time_s': sum(r['force_limit_active'] for r in rows[:-1]) * h,
            'speed_limit_time_s': sum(r['speed_limit_active'] for r in rows[:-1]) * h,
            'force_speed_stopping_time_s': sum(r['stopping_constraint_active'] for r in rows[:-1]) * h,
            'final_position_m': positions[-1],
            'command_clipped': abs(requested - rows[0]['effective_command_m']) > 1e-12}

def amplitude_gain(mu, s, inertia: Callable[[float, float], float]):
    I = inertia(mu, s)
    h = 1e-5
    # A derivative of the declared inertia family, not a replacement I(0).
    dI = (inertia(mu, s + h) - inertia(mu, s - h)) / (2 * h)
    K = 2 * mu * DESIGN['M_kg'] * DESIGN['g_m_s2']
    exact = K * (I - s * dI) / I**2
    old = K / I
    return {'I_endpoint_kg_m2': I, 'I_zero_kg_m2': inertia(mu, 0.0),
            'inertia_derivative_kg_m': dI,
            'Gamma_differential_magnitude_rad_s2_per_m': abs(exact),
            'Gamma_frozen_inertia_approx_rad_s2_per_m': old,
            'Gamma_approx_relative_difference': (old - abs(exact)) / abs(exact),
            'alpha_endpoint_rad_s2': K * abs(s) / I,
            'alpha_zero_inertia_approx_rad_s2': K * abs(s) / inertia(mu, 0.0)}

def find_crossings(function, lower=.00001, upper=.10, samples=256):
    roots = []
    left, fl = lower, function(lower)
    for i in range(1, samples + 1):
        right = lower + (upper - lower) * i / samples
        fr = function(right)
        if abs(fr) < 1e-12 or fl * fr < 0:
            roots.append(right if abs(fr) < 1e-12 else legacy._bisect_root(function, left, right))
        left, fl = right, fr
    return roots

def self_test():
    checks = []
    def check(name, actual, expected, tolerance):
        checks.append({'name': name, 'actual': actual, 'expected': expected,
                       'tolerance': tolerance, 'passed': abs(actual-expected) <= tolerance})
    task = attitude_task()
    for u in task['peak_acceleration_normalized_times']:
        actual = task['angle_rad'] / DESIGN['attitude_build_s']**2 * abs(60*u-180*u*u+120*u**3)
        check('Independent quintic analytic extrema', actual, task['peak_acceleration_rad_s2'], 1e-13)
    C90 = legacy.t90_dimensionless(1.2)
    check('Known independent overdamped T90 coefficient', C90, 4.922663526544, 1e-11)
    for mu in [.015, DESIGN['mu0'], .06]:
        d = DESIGN['cog_target_m']/(2*mu)
        check('Fixed CoG force cancellation', mu*DESIGN['M_kg']*20**2*d, DESIGN['M_kg']*20**2*.01/2, 1e-12)
        check('Fixed CoG acceleration cancellation', 2*mu*20**2*d, 20**2*.01, 1e-12)
        for s in [.0,.05,.135,.15]:
            g = amplitude_gain(mu,s,legacy.effective_inertia_axis_simplified)
            h = 1e-6
            f = lambda q: legacy.hover_torque(mu,abs(q))/legacy.effective_inertia_axis_simplified(mu,q)*math.copysign(1,q)
            numeric = (f(s+h)-f(s-h))/(2*h)
            check('Finite difference of full torque/inertia acceleration', g['Gamma_differential_magnitude_rad_s2_per_m'], abs(numeric), 2e-7)
    for wn in [10.,20.,30.]:
        result = response_metrics(constrained_step(wn,.05,.05), .05)
        checks.append({'name': 'Bounded force without state reset', 'wn_rad_s': wn,
                       'passed': result['peak_force_N'] <= 2+1e-12 and result['peak_speed_m_s'] <= .25+1e-9})
    failed = [c for c in checks if not c['passed']]
    if failed:
        raise AssertionError(failed)
    return {'passed': True, 'checks': checks}
