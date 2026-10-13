#!/usr/bin/env python3
"""Shared MPC objective, coordinate transformation and measured-error integral."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import types

import numpy as np
import osqp

from mmc_control import mmc_uav as control
from mmc_control.mpc_tracking_design import HorizontalTrackingIntegral
from mmc_control.mpc_constraint_design import ConstraintDesignedAttitudeMPC
import mmc_control.mpc_constraint_design as constraint_design_module
import mmc_control.mpc_tracking_design as tracking_design
from mmc_control import mpc_qp_coordinates
from .baselines import make_controller as make_baseline

ORIGINAL_OSQP_FACTORY = osqp.OSQP

FIELDS = ['controller_time_s', 'reference_x_m', 'reference_y_m',
          'measured_x_m', 'measured_y_m', 'integral_x_m_s', 'integral_y_m_s',
          'integral_ax_m_s2', 'integral_ay_m_s2',
          'base_phi_ref_rad', 'base_theta_ref_rad',
          'phi_trim_rad', 'theta_trim_rad', 'phi_trim_rate_rad_s',
          'theta_trim_rate_rad_s', 'actual_phi_ref_rad', 'actual_theta_ref_rad',
          'actual_p_ref_rad_s', 'actual_q_ref_rad_s',
          'blocked_x', 'blocked_y']


def make_controller(kind, production_mpc, configuration_dict=None, node=None):
    configuration = configuration_dict or {}
    if kind in ('pid', 'lqr'):
        return make_baseline(kind, production_mpc, configuration)
    if kind not in ('constrained_mpc', 'unconstrained_mpc'):
        raise ValueError('Unknown MMC controller: ' + str(kind))
    unknown = set(configuration) - {'mpc', 'horizontal_integral', 'qp_coordinates', 'constraint_design'}
    if unknown:
        raise ValueError('Unknown design settings: ' + str(sorted(unknown)))
    settings = {'Np': production_mpc.Np, 'dt_mpc': production_mpc.dt,
                **configuration.get('mpc', {})}
    mpc = ConstraintDesignedAttitudeMPC(production_mpc.P,
        constraint_design=configuration.get('constraint_design', {}), **settings)
    integral = HorizontalTrackingIntegral(**configuration.get('horizontal_integral', {}))
    coordinates = configuration.get('qp_coordinates', 'delta_input')
    if coordinates not in ('delta_input', 'absolute_input'):
        raise ValueError('Unknown QP coordinates: ' + str(coordinates))
    mpc._mmc_current_control_time = 0.0
    if coordinates == 'absolute_input':
        # Both the production inequality solver and the baseline zero-row
        # solver import this same OSQP module. One factory provides exactly
        # the same coordinate transformation for both variants in this node.
        osqp.OSQP = lambda: mpc_qp_coordinates.AbsoluteInputOSQP(
            ORIGINAL_OSQP_FACTORY,
            lambda: mpc._x_aug0_buffer[-mpc.nu:].copy(),
            control_time_getter=lambda: mpc._mmc_current_control_time,
            constraint_tolerance=mpc.qp_constraint_tolerance)
    else:
        osqp.OSQP = ORIGINAL_OSQP_FACTORY
    selected = make_baseline(kind, mpc)
    original_compute = selected.compute

    def timed_compute(t, state, thrust_cmd, ref_att_seq, last_u_opt=None):
        mpc._mmc_current_control_time = float(t)
        result = original_compute(t, state, thrust_cmd, ref_att_seq,
                                  last_u_opt=last_u_opt)
        if kind == 'constrained_mpc' and selected.last_dual.size != mpc._constraint_A_sparse.shape[0]:
            # The inherited observation wrapper has a historical 10*Np
            # failure placeholder. Preserve every new row on rejected solves.
            selected.last_dual = np.full(mpc._constraint_A_sparse.shape[0], np.nan)
        return result
    selected.compute = timed_compute
    selected.configuration = configuration
    original_record = selected.design_record

    def record():
        result = original_record()
        result['optimization_constraint_rows'] = (mpc._constraint_A_sparse.shape[0]
                                                  if kind == 'constrained_mpc' else 0)
        result['constraint_layout'] = mpc.constraint_layout_record()
        result['constraint_layout']['applied_to_solver'] = kind == 'constrained_mpc'
        if kind == 'unconstrained_mpc':
            result['removed_inequalities'] += ['intersample nominal slider velocity', 'nominal servo force']
        result['shared_design'] = {'configuration': configuration,
            'discretization': settings.get('discretization', 'forward_euler'),
            'solver_internal_coordinates': coordinates,
            'retained_decision_coordinates': 'Original input increments; native absolute inputs are separately preserved',
            'ordinary_horizontal_error_integral': bool(integral.enabled),
            'integral_location': 'MPC-owned target trim after the unmodified common reference generator',
            'integral_measurements': 'Existing body-origin odometry and common sample-held position reference',
            'integral_preview': 'Current trim plus k*dt times its measured rate over the common preview; bounded by the declared trim and attitude limits',
            'NDO_estimates_or_wind_truth_used_by_integral': False,
            'same_base_design_for_constraints_and_NDO_ablations': True}
        result['observation'] = 'Configured shared MPC; only explicit QP inequalities differ between its two variants'
        return result
    selected.design_record = record
    out = Path(os.environ['MMC_COMPARISON_OUT'])
    source = Path(tracking_design.__file__).absolute()
    (out / 'loaded_tracking_design.json').write_text(json.dumps({
        'module_file': str(source), 'module_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'configuration': configuration}, indent=2) + '\n')
    constraint_source = Path(constraint_design_module.__file__).absolute()
    (out / 'loaded_constraint_design.json').write_text(json.dumps({
        'module_file': str(constraint_source),
        'module_sha256': hashlib.sha256(constraint_source.read_bytes()).hexdigest(),
        'layout': mpc.constraint_layout_record(),
        'applied_to_solver': kind == 'constrained_mpc'}, indent=2) + '\n')
    if node is None:
        if integral.enabled:
            raise ValueError('MPC position integral requires the existing node measurements/reference')
        return selected
    original_build = node._build_attitude_reference_sequence
    handle = (out / 'mpc_integral.csv').open('w', newline='')
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    node._mmc_mpc_integral = integral

    def build_target(self, t):
        base = np.asarray(original_build(t), dtype=float)
        with control.optional_lock(self):
            state = control.State6(**self.state.__dict__)
            desired = control.RefPos(**self.ref_pos_now.__dict__)
        # Existing publication saturation may prevent further integral growth.
        # World-axis direction is supplied only when public angle saturation
        # occurs; no physical/model limit is relaxed by this controller.
        phi_limit = float(self.outer.roll_limit)
        theta_limit = float(self.outer.pitch_limit)
        trim_phi, trim_theta = integral.step(float(t), [state.x, state.y],
            [desired.x, desired.y], state.psi,
            dt=float(self.last_inner_dt),
            phi_reference_rad=float(base[0, 0]),
            theta_reference_rad=float(base[0, 1]),
            roll_limit_rad=phi_limit, pitch_limit_rad=theta_limit)
        target = base.copy()
        preview_time = np.arange(mpc.Np) * mpc.dt
        phi_preview = np.clip(trim_phi + preview_time * integral.last_phi_rate_rad_s,
                              -integral.trim_limit_rad[0], integral.trim_limit_rad[0])
        theta_preview = np.clip(trim_theta + preview_time * integral.last_theta_rate_rad_s,
                                -integral.trim_limit_rad[1], integral.trim_limit_rad[1])
        target[:, 0] = np.clip(base[:, 0] + phi_preview, -phi_limit, phi_limit)
        target[:, 1] = np.clip(base[:, 1] + theta_preview, -theta_limit, theta_limit)
        # The MPC integral target preview and its body-rate contribution
        # use the same current measured-error derivative.
        target[:, 3] += integral.last_phi_rate_rad_s
        target[:, 4] += integral.last_theta_rate_rad_s
        self.inner_phi_ref = float(target[0, 0])
        self.inner_theta_ref = float(target[0, 1])
        self.inner_psi_ref = float(target[0, 2])
        blocked = integral.last_record.get('blocked_xy', [False, False])
        writer.writerow(dict(zip(FIELDS, [float(t), desired.x, desired.y,
            state.x, state.y, *integral.integral_xy, *integral.acceleration_xy,
            base[0, 0], base[0, 1], trim_phi, trim_theta,
            integral.last_phi_rate_rad_s, integral.last_theta_rate_rad_s,
            target[0, 0], target[0, 1], target[0, 3], target[0, 4],
            int(blocked[0]), int(blocked[1])])) )
        if int(float(t) * 100) % 100 == 0:
            handle.flush()
        return target
    node._build_attitude_reference_sequence = types.MethodType(build_target, node)
    return selected


def finalize(out):
    """Save every native solver result in addition to the original QP atlas."""
    try:
        mpc_qp_coordinates.dump_records(Path(out))
    finally:
        osqp.OSQP = ORIGINAL_OSQP_FACTORY
