#!/usr/bin/env python3
"""Analytic task-only point/round-trip reference; no state or control inputs."""
from __future__ import annotations

import math


def task_state(stamp, protocol):
    """Return xyz/velocity/acceleration for the frozen external task."""
    a, b = protocol['target_A_m'], protocol['target_B_m']
    start = float(protocol['movement_start_controller_s'])
    duration = float(protocol['resolved_movement_duration_s'])
    outbound_end = start + duration
    stamp = float(stamp)
    if protocol.get('trajectory_mode') == 'point_round_trip' and stamp >= float(protocol['return_start_controller_s']):
        a, b = b, protocol['target_terminal_m']
        start = float(protocol['return_start_controller_s'])
        duration = float(protocol['return_duration_s'])
    if stamp <= start:
        blend, velocity, acceleration = 0., 0., 0.
    elif stamp >= start + duration:
        blend, velocity, acceleration = 1., 0., 0.
    else:
        q = (stamp - start) / duration
        if protocol.get('polynomial_order', 5) == 7:
            blend = q**4 * (35. + q * (-84. + q * (70. - 20. * q)))
            velocity = 140. * q**3 * (1. - q)**3 / duration
            acceleration = 420. * q**2 * (1. - q)**2 * (1. - 2. * q) / duration**2
        else:
            blend = q**3 * (10. + q * (-15. + 6. * q))
            velocity = 30. * q**2 * (1. - q)**2 / duration
            acceleration = 60. * q * (1. - q) * (1. - 2. * q) / duration**2
    position = tuple(float(x + (y - x) * blend) for x, y in zip(a, b))
    speed = tuple(float((y - x) * velocity) for x, y in zip(a, b))
    accel = tuple(float((y - x) * acceleration) for x, y in zip(a, b))
    return position, speed, accel


def task_functions(protocol):
    return tuple((lambda stamp, axis=axis: task_state(stamp, protocol)[0][axis]) for axis in range(3))
