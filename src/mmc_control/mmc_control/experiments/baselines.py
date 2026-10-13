#!/usr/bin/env python3
"""PID, LQR and MPC inner-loop controllers with common physical protection."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are


STATE_NAMES = ['phi', 'theta', 'psi', 'p', 'q', 'r', 's_x', 'v_x', 's_y', 'v_y']
INPUT_NAMES = ['tau_z', 's_x_cmd', 's_y_cmd']
PID_DEFAULT = {
    'kp': [9.0, 9.0, 2.25],
    'ki': [0.35, 0.35, 0.12],
    'kd': [4.8, 4.8, 2.7],
    'derivative_time_constant_s': 0.025,
    'integral_limit_rad_s': [0.25, 0.25, 0.25],
    'antiwindup_tracking_time_s': 0.25,
}
LQR_DEFAULT = {
    'Q_diagonal': [50.0, 50.0, 30.0, 4.0, 4.0, 2.0, 1.0, 0.2, 1.0, 0.2],
    'R_diagonal': [1000.0, 12.0, 12.0],
}



def _wrapped_angles(error):
    return (np.asarray(error, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _as_triplet(value, name):
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(vector).all() or np.any(vector < 0):
        raise ValueError(f'{name} must contain three finite nonnegative numbers')
    return vector


def nominal_attitude_model(production_mpc):
    """Extract current MPC Jacobian at level, centred, zero-wind hover.

    Forward Euler is intentionally the same discretization as production
    MPC.  The 10-state restriction is dynamically closed at this equilibrium.
    The original nonlinear model's aero rotational damping is retained.
    """
    x0 = np.zeros(production_mpc.nx_state)
    x0[2] = 2.0
    u0 = np.zeros(3)
    thrust0 = float(production_mpc.P.M * production_mpc.P.g)
    a_full, b_full, f_full = production_mpc._linearize_AB(x0, u0, thrust0)
    indices = np.arange(6, 16)
    a_c = a_full[np.ix_(indices, indices)]
    b_c = b_full[indices, :]
    a_d = np.eye(10) + production_mpc.dt * a_c
    b_d = production_mpc.dt * b_c
    excluded_coupling = a_full[np.ix_(indices, np.arange(6))]
    if np.max(np.abs(excluded_coupling)) > 1e-8:
        raise ValueError('Nominal attitude/actuator subsystem is not closed')
    controllability = np.concatenate([np.linalg.matrix_power(a_d, k) @ b_d
                                     for k in range(10)], axis=1)
    if np.linalg.matrix_rank(controllability) != 10:
        raise ValueError('Nominal attitude/actuator model is not controllable')
    # Torque per unit paired slider displacement: roll<-s_y, pitch<-s_x.
    mu, inertia_x, inertia_y, inertia_z = (
        production_mpc.P.mu,
        -2.0 * production_mpc.P.mu * thrust0 / a_c[3, 8],
        2.0 * production_mpc.P.mu * thrust0 / a_c[4, 6],
        1.0 / b_c[5, 0],
    )
    return {
        'A_c': a_c, 'B_c': b_c, 'A_d': a_d, 'B_d': b_d,
        'nominal_input': u0, 'nominal_state': x0[6:16],
        'nominal_thrust_N': thrust0,
        'inertia_diagonal_kg_m2': np.array([inertia_x, inertia_y, inertia_z]),
        'controllability_rank': 10,
        'max_excluded_state_coupling': float(np.max(np.abs(excluded_coupling))),
        'max_nominal_attitude_derivative': float(np.max(np.abs(f_full[6:16]))),
        'mu': float(mu),
    }


def _json_model(model):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}


def _complex_pairs(values):
    return [[float(z.real), float(z.imag)] for z in values]


class _Baseline:
    has_optimization_constraints = False
    is_mpc = False

    def __init__(self, production_mpc, configuration=None):
        self.production_mpc = production_mpc
        self.configuration = copy.deepcopy(configuration or {})
        self.last_raw_request = np.zeros(3)
        self.last_decision = np.empty(0)
        self.last_dual = np.empty(0)
        self.last_published_input = np.zeros(3)
        self.last_reference_now = np.zeros(10)
        self.last_compute_dt = 0.0

    def __getattr__(self, name):
        return getattr(self.production_mpc, name)

    def _reference(self, ref_att_seq):
        packed = self.production_mpc._pack_reference_sequence(ref_att_seq)
        if not np.isfinite(packed).all():
            raise ValueError('Nonfinite attitude reference')
        self.last_reference_now = packed[0].copy()
        return packed

    def _record_published(self, last_u_opt):
        self.last_published_input = (np.zeros(3) if last_u_opt is None
                                     else np.asarray(last_u_opt, dtype=float).reshape(3).copy())
        if not np.isfinite(self.last_published_input).all():
            raise ValueError('Nonfinite previous published input')

    def _non_qp_diagnostics(self):
        # Required compatibility for existing black-box CSV fields. The
        # comparison metadata declares these solver fields inapplicable for
        # PID/LQR; no optimization activity is computed for those methods.
        m = self.production_mpc
        for name in ('last_model_update_dt', 'last_qp_build_dt', 'last_solver_setup_dt',
                     'last_solver_solve_dt'):
            setattr(m, name, 0.0)
        m.last_model_reused = False
        m.last_lpv_reuse_count = 0
        m.last_refresh_reason_mask = 0
        m.last_solver_status = 0
        m.last_solution_accepted = True
        m.last_constraint_violation = float('nan')
        m.consecutive_failures = 0
        m.last_predicted_output.fill(np.nan)

    def diagnostics(self):
        return {
            'kind': self.kind,
            'is_mpc': self.is_mpc,
            'has_optimization_constraints': self.has_optimization_constraints,
            'raw_request': self.last_raw_request.tolist(),
            'previous_published_input': self.last_published_input.tolist(),
            'current_reference': self.last_reference_now.tolist(),
            'compute_time_s': self.last_compute_dt,
        }

    def _common_design_record(self):
        return {
            'kind': self.kind, 'configuration': copy.deepcopy(self.configuration),
            'state_order': STATE_NAMES, 'input_order': INPUT_NAMES,
            'reference_order': STATE_NAMES,
            'model_sample_time_s': self.dt,
            'common_publication_protection': 'Unmodified production node: slider request clips, optional shared command shaping, shared yaw torque clipping/filter and rotor allocation.',
            'previous_input': 'Previous actually published input supplied by production node; raw request never substituted.',
            'physical_parameters': asdict(self.P),
        }


class PIDAttitude(_Baseline):
    """Angle/rate PID, physical torque mapping and back-calculation antiwindup.

    kp/ki/kd produce angular acceleration in rad/s^2. D uses the supplied
    instantaneous rate reference minus a low-pass-filtered measured body
    rate; it never differentiates attitude-reference steps. Integral and
    filter discretization use the same nominal 0.01 s period for all calls.
    """
    kind = 'pid'

    def __init__(self, production_mpc, configuration=None):
        settings = copy.deepcopy(PID_DEFAULT)
        settings.update(configuration or {})
        super().__init__(production_mpc, settings)
        self.model = nominal_attitude_model(production_mpc)
        self.kp = _as_triplet(settings['kp'], 'kp')
        self.ki = _as_triplet(settings['ki'], 'ki')
        self.kd = _as_triplet(settings['kd'], 'kd')
        self.integral_limit = _as_triplet(settings['integral_limit_rad_s'], 'integral limits')
        self.derivative_tau = float(settings['derivative_time_constant_s'])
        self.aw_tau = float(settings['antiwindup_tracking_time_s'])
        if not (self.derivative_tau >= 0 and self.aw_tau > 0):
            raise ValueError('Invalid PID derivative/antiwindup time constant')
        self.integral = np.zeros(3)
        self.filtered_rates = np.zeros(3)
        self._filter_initialized = False
        self._previous_raw = None
        self._previous_map = None
        self.last_antiwindup_correction = np.zeros(3)

    def _acceleration_to_input_map(self, thrust_cmd):
        jx, jy, jz = self.model['inertia_diagonal_kg_m2']
        # During zero-thrust initialization, bounded denominator prevents a
        # numerical singularity. This is a declared inverse authority map,
        # not disturbance estimation or any change to thrust allocation.
        effective_thrust = max(float(thrust_cmd), 0.1 * self.model['nominal_thrust_N'])
        slider_torque_scale = 2.0 * self.P.mu * effective_thrust
        return np.array([[0.0, 0.0, jz],
                         [0.0, jy / slider_torque_scale, 0.0],
                         [-jx / slider_torque_scale, 0.0, 0.0]])

    def compute(self, t, state, thrust_cmd, ref_att_seq, last_u_opt=None):
        begin = time.perf_counter()
        self._record_published(last_u_opt)
        ref = self._reference(ref_att_seq)[0]
        angles = np.array([state.phi, state.theta, state.psi])
        rates = np.array([state.p, state.q, state.r])
        if not np.isfinite(np.r_[angles, rates]).all():
            raise ValueError('Nonfinite PID state')
        error = _wrapped_angles(ref[:3] - angles)
        alpha = math.exp(-self.dt / self.derivative_tau) if self.derivative_tau else 0.0
        if not self._filter_initialized:
            self.filtered_rates[:] = rates
            self._filter_initialized = True
        self.filtered_rates[:] = alpha * self.filtered_rates + (1.0 - alpha) * rates
        self.last_antiwindup_correction.fill(0.0)
        if self._previous_raw is not None:
            # Back-calculation uses the full previous publication (including
            # shared yaw filter and allocation protection), not just an
            # imagined internally clipped command.
            mismatch = np.linalg.solve(self._previous_map,
                                       self.last_published_input - self._previous_raw)
            self.last_antiwindup_correction = np.divide(
                mismatch, self.aw_tau * self.ki,
                out=np.zeros(3), where=self.ki > 0)
        self.integral[:] = np.clip(
            self.integral + self.dt * (error + self.last_antiwindup_correction),
            -self.integral_limit, self.integral_limit)
        acceleration = (self.kp * error + self.ki * self.integral
                        + self.kd * (ref[3:6] - self.filtered_rates))
        mapping = self._acceleration_to_input_map(thrust_cmd)
        self.last_raw_request = mapping @ acceleration
        self._previous_raw = self.last_raw_request.copy()
        self._previous_map = mapping
        self._non_qp_diagnostics()
        self.last_compute_dt = time.perf_counter() - begin
        return tuple(float(v) for v in self.last_raw_request)

    def nominal_closed_loop_matrix(self):
        """Unsaturated sampled PID stability with both filter/integral states."""
        a, b = self.model['A_d'], self.model['B_d']
        angle_select = np.zeros((3, 10)); angle_select[:, :3] = np.eye(3)
        rate_select = np.zeros((3, 10)); rate_select[:, 3:6] = np.eye(3)
        alpha = math.exp(-self.dt / self.derivative_tau) if self.derivative_tau else 0.0
        mapping = self._acceleration_to_input_map(self.model['nominal_thrust_N'])
        f_x = (-np.diag(self.kp + self.ki * self.dt) @ angle_select
               - np.diag(self.kd * (1.0 - alpha)) @ rate_select)
        feedback = mapping @ np.c_[f_x, np.diag(self.ki), -np.diag(self.kd * alpha)]
        return np.block([
            [a + b @ feedback[:, :10], b @ feedback[:, 10:]],
            [-self.dt * angle_select, np.eye(3), np.zeros((3, 3))],
            [(1.0 - alpha) * rate_select, np.zeros((3, 3)), alpha * np.eye(3)],
        ])

    def design_record(self):
        closed = self.nominal_closed_loop_matrix()
        eigenvalues = np.linalg.eigvals(closed)
        record = self._common_design_record()
        record.update(
            nominal_model=_json_model(self.model),
            nominal_closed_loop_matrix=closed.tolist(),
            nominal_closed_loop_eigenvalues=_complex_pairs(eigenvalues),
            nominal_closed_loop_spectral_radius=float(np.max(np.abs(eigenvalues))),
            nominal_sampled_stable=bool(np.max(np.abs(eigenvalues)) < 1.0),
            derivative='Supplied current body-rate reference minus filtered measured rate; no finite difference of reference angles.',
            antiwindup='Back-calculation from previous actual publication minus previous raw request, transformed to acceleration and divided by ki*Taw; bounded integral.',
            input_mapping='tau_z=Jz*a_yaw; s_x_cmd=Jy*a_pitch/(2*mu*thrust); s_y_cmd=-Jx*a_roll/(2*mu*thrust). Inertias come from the same nominal nonlinear Jacobian.',
        )
        return record


class LQRAttitude(_Baseline):
    """Fixed nominal discrete LQR with current full-state reference tracking."""
    kind = 'lqr'

    def __init__(self, production_mpc, configuration=None):
        settings = copy.deepcopy(LQR_DEFAULT)
        settings.update(configuration or {})
        super().__init__(production_mpc, settings)
        self.model = nominal_attitude_model(production_mpc)
        q = np.asarray(settings['Q_diagonal'], dtype=float).reshape(10)
        r = np.asarray(settings['R_diagonal'], dtype=float).reshape(3)
        if not np.isfinite(np.r_[q, r]).all() or np.any(q < 0) or np.any(r <= 0):
            raise ValueError('Invalid LQR Q/R')
        self.Q = np.diag(q); self.R = np.diag(r)
        a, b = self.model['A_d'], self.model['B_d']
        self.S = solve_discrete_are(a, b, self.Q, self.R)
        self.K = np.linalg.solve(self.R + b.T @ self.S @ b, b.T @ self.S @ a)
        self.closed_loop = a - b @ self.K
        self.eigenvalues = np.linalg.eigvals(self.closed_loop)
        if np.max(np.abs(self.eigenvalues)) >= 1.0:
            raise ValueError('Unstable nominal sampled LQR')
        self.riccati_residual = a.T @ self.S @ a - self.S + self.Q - (
            a.T @ self.S @ b @ self.K)

    def compute(self, t, state, thrust_cmd, ref_att_seq, last_u_opt=None):
        begin = time.perf_counter()
        self._record_published(last_u_opt)
        ref = self._reference(ref_att_seq)[0]
        x = self.production_mpc._state6_to_vec(state)[6:16]
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite LQR state')
        error = x - ref
        error[:3] = _wrapped_angles(error[:3])
        self.last_raw_request = self.model['nominal_input'] - self.K @ error
        self._non_qp_diagnostics()
        self.last_compute_dt = time.perf_counter() - begin
        return tuple(float(v) for v in self.last_raw_request)

    def design_record(self):
        record = self._common_design_record()
        record.update(
            nominal_model=_json_model(self.model), Q=self.Q.tolist(), R=self.R.tolist(),
            Riccati_S=self.S.tolist(), K=self.K.tolist(),
            closed_loop_A=self.closed_loop.tolist(),
            closed_loop_eigenvalues=_complex_pairs(self.eigenvalues),
            closed_loop_spectral_radius=float(np.max(np.abs(self.eigenvalues))),
            nominal_sampled_stable=True,
            riccati_residual_max=float(np.max(np.abs(self.riccati_residual))),
            tracking='u=u_nominal-K*(x-x_reference_now); nominal u=0 at centred hover; current supplied attitude, body-rate and actuator-state references used. No reference lookahead, observer, integral augmentation or gain scheduling.',
        )
        return record


class UnconstrainedMPC(_Baseline):
    """Identical production LPV/condensed objective, zero inequality rows."""
    kind = 'unconstrained_mpc'
    is_mpc = True

    def __init__(self, production_mpc, configuration=None):
        if configuration:
            raise ValueError('MPC weights/model are production-frozen; no override accepted')
        super().__init__(production_mpc)
        self._solver = None
        self._zero_constraint_matrix = sparse.csc_matrix((0, production_mpc._n_dec))
        self._zero_bounds = np.empty(0)
        self.last_stationarity_residual = float('nan')

    def compute(self, t, state, thrust_cmd, ref_att_seq, last_u_opt=None):
        begin = time.perf_counter()
        self._record_published(last_u_opt)
        m = self.production_mpc
        x0 = m._state6_to_vec(state)
        update_begin = time.perf_counter()
        m._update_lpv_model(x0, thrust_cmd, self.last_published_input)
        m.last_model_update_dt = time.perf_counter() - update_begin
        m.last_model_reused = False; m.last_lpv_reuse_count = 0; m.last_refresh_reason_mask = 0
        build_begin = time.perf_counter()
        m._x_aug0_buffer[:m.nx_state] = x0
        m._x_aug0_buffer[m.nx_state:] = self.last_published_input
        y0 = m.Phi @ m._x_aug0_buffer + m._cached_Y_const
        reference = self._reference(ref_att_seq)
        m.last_Y_ref.fill(0.0)
        m.last_Y_ref.reshape(m.Np, m.ny)[:, :m.ny_state] = reference
        gradient = m._build_qp_gradient(y0 - m.last_Y_ref)
        q_vec = 2.0 * gradient
        m.last_qp_build_dt = time.perf_counter() - build_begin
        setup_begin = time.perf_counter()
        if self._solver is None:
            self._solver = osqp.OSQP()
            self._solver.setup(P=m._build_upper_triangular_hessian(), q=q_vec,
                               A=self._zero_constraint_matrix, l=self._zero_bounds,
                               u=self._zero_bounds, warm_starting=True,
                               eps_abs=1e-5, eps_rel=1e-5, verbose=False)
        else:
            self._solver.update(Px=m._build_upper_triangular_hessian_data(), q=q_vec)
        m.last_solver_setup_dt = time.perf_counter() - setup_begin
        solve_begin = time.perf_counter()
        result = self._solver.solve()
        m.last_solver_solve_dt = time.perf_counter() - solve_begin
        m._mmc_observed_result = result
        self.last_decision = (np.asarray(result.x).copy() if result.x is not None
                              else np.full(m._n_dec, np.nan))
        self.last_dual = (np.asarray(result.y).copy() if result.y is not None else np.empty(0))
        m.last_solver_status = int(result.info.status_val)
        finite = np.isfinite(self.last_decision).all()
        # There are no optimization inequalities; status/finite checks stay
        # equal to production. Shared postcompute physical guards still run.
        m.last_constraint_violation = 0.0 if finite else float('inf')
        m.last_solution_accepted = bool(m.last_solver_status in (1, 2) and finite)
        if m.last_solution_accepted:
            m.consecutive_failures = 0
            m.last_predicted_output[:] = (y0 + np.einsum('ij,j->i', m.Gamma,
                                                        self.last_decision, optimize=False)).reshape(m.Np, m.ny)
            self.last_raw_request = self.last_published_input + self.last_decision[:m.nu]
            self.last_stationarity_residual = float(np.max(np.abs(m.H_mat @ self.last_decision + gradient)))
        else:
            m.last_predicted_output.fill(np.nan)
            m.consecutive_failures += 1
            self.last_stationarity_residual = float('nan')
            self.last_raw_request = (np.zeros(3) if m.consecutive_failures >= m.max_consecutive_failures
                                     else self.last_published_input * m.decay_factor)
        self.last_compute_dt = time.perf_counter() - begin
        return tuple(float(v) for v in self.last_raw_request)

    def design_record(self):
        record = self._common_design_record()
        record.update(_mpc_design_record(self.production_mpc))
        record.update(
            optimization_constraint_rows=0,
            solver='OSQP; same warm-start and tolerance settings as production',
            removed_inequalities=['input increments', 'absolute inputs', 'predicted actual slider position', 'predicted actual slider velocity'],
            maintained='Identical LPV affine model, augmented published-input memory, reference packing, output selection, Q/R/Rdelta, horizon and Hessian 1e-4 regularizer; common downstream physical/publication protection.',
        )
        return record


class _SolveCapture:
    def __init__(self, solver, owner):
        self._solver, self._owner = solver, owner

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def solve(self, *args, **kwargs):
        result = self._solver.solve(*args, **kwargs)
        m = self._owner.production_mpc
        m._mmc_observed_result = result
        self._owner.last_decision = (np.asarray(result.x).copy() if result.x is not None
                                     else np.full(m._n_dec, np.nan))
        self._owner.last_dual = (np.asarray(result.y).copy() if result.y is not None
                                else np.full(2*m._n_dec + 4*m.Np, np.nan))
        return result


class ConstrainedMPC(_Baseline):
    """Observe and forward the unmodified production solver/output."""
    kind = 'constrained_mpc'
    is_mpc = True
    has_optimization_constraints = True

    def __init__(self, production_mpc, configuration=None):
        if configuration:
            raise ValueError('MPC weights/model are production-frozen; no override accepted')
        super().__init__(production_mpc)
        original_ensure = production_mpc._ensure_solver_ready
        production_mpc._ensure_solver_ready = lambda *a, **k: _SolveCapture(original_ensure(*a, **k), self)

    def compute(self, t, state, thrust_cmd, ref_att_seq, last_u_opt=None):
        begin = time.perf_counter()
        self._record_published(last_u_opt)
        result = self.production_mpc.compute(t, state, thrust_cmd, ref_att_seq,
                                             last_u_opt=self.last_published_input)
        self.last_reference_now = self.production_mpc.last_ref_sequence_10[0].copy()
        self.last_raw_request = np.asarray(result, dtype=float).copy()
        self.last_compute_dt = time.perf_counter() - begin
        return result

    def design_record(self):
        record = self._common_design_record()
        record.update(_mpc_design_record(self.production_mpc))
        record.update(optimization_constraint_rows=2*self._n_dec + 4*self.Np,
                      observation='Production compute return forwarded unchanged; solver result copied after solve, no objective/model/constraint changes.')
        return record


def _mpc_design_record(m):
    return {
        'prediction_steps': m.Np,
        'Q_y': m.Q_y.tolist(), 'R_u': m.R_u.tolist(), 'R_delta': m.R_delta.tolist(),
        'hessian_regularization': 1e-4,
        'slider_command_limit_m': m.u2_lim,
        'predicted_slider_position_limit_m': m.u2_lim,
        'slider_speed_limit_m_s': m.slider_vel_lim,
        'yaw_torque_optimization_limit_Nm': m.tau_z_lim,
        'solver_eps_abs': 1e-5, 'solver_eps_rel': 1e-5,
    }


def make_controller(kind, production_mpc, configuration_dict=None):
    """Construct only an explicit MMC experimental inner-loop replacement."""
    constructors = {'pid': PIDAttitude, 'lqr': LQRAttitude,
                    'unconstrained_mpc': UnconstrainedMPC, 'constrained_mpc': ConstrainedMPC}
    if kind not in constructors:
        raise ValueError(f'Unknown MMC controller: {kind}')
    return constructors[kind](production_mpc, configuration_dict)
