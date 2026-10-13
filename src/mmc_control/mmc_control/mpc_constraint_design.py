"""Explicit optional rail feasibility constraints for the actuator-constrained control experiments.

The shared LPV model, condensed objective, horizon, published-input memory
and safety fallback remain in the original MPC. Explicit linear inequalities
are added; an opt-in equivalent compilation can merge first-interval bounds
into the existing command rows while preserving the sparse row pattern.
This module uses the existing measured rail state and nominal
servo parameters; it accepts no wind, disturbance estimate or joint force.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from .mpc_tracking_design import ConfiguredAttitudeMPC


def _fractions(values, name, *, allow_zero):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of sample fractions")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
        raise ValueError(f"{name} must contain real numeric fractions")
    result = tuple(float(value) for value in values)
    if (any(not math.isfinite(value) or value > 1.0
                   or value < 0.0 or (value == 0.0 and not allow_zero)
                   for value in result)
            or tuple(sorted(set(result))) != result):
        raise ValueError(f"{name} must contain sorted, distinct fractions in the interval")
    return result


def validate_constraint_design(configuration, physical_velocity_limit):
    if not math.isfinite(physical_velocity_limit) or physical_velocity_limit <= 0.0:
        raise ValueError("Physical velocity limit must be finite and positive")
    if configuration is not None and not isinstance(configuration, Mapping):
        raise ValueError("Constraint design must be a mapping")
    settings = dict(configuration) if configuration is not None else {}
    unknown = set(settings) - {"velocity_fractions", "force_fractions",
                               "force_limit_N", "velocity_margin_m_s",
                               "first_interval_velocity_fractions", "first_interval_force",
                               "compile_first_interval_into_command_bounds"}
    if unknown:
        raise ValueError(f"Unsupported constraint settings: {sorted(unknown)}")
    settings.setdefault("velocity_fractions", [])
    settings.setdefault("force_fractions", [])
    settings.setdefault("force_limit_N", 2.0)
    settings.setdefault("velocity_margin_m_s", 0.0)
    settings.setdefault("first_interval_velocity_fractions", [])
    settings.setdefault("first_interval_force", False)
    settings["velocity_fractions"] = list(_fractions(
        settings["velocity_fractions"], "velocity_fractions", allow_zero=False))
    settings["force_fractions"] = list(_fractions(
        settings["force_fractions"], "force_fractions", allow_zero=True))
    settings["first_interval_velocity_fractions"] = list(_fractions(
        settings["first_interval_velocity_fractions"], "first_interval_velocity_fractions", allow_zero=False))
    if not isinstance(settings["first_interval_force"], (bool, np.bool_)):
        raise ValueError("first_interval_force must be boolean")
    settings["first_interval_force"] = bool(settings["first_interval_force"])
    compile_first = settings.get("compile_first_interval_into_command_bounds", False)
    if not isinstance(compile_first, (bool, np.bool_)):
        raise ValueError("compile_first_interval_into_command_bounds must be boolean")
    if "compile_first_interval_into_command_bounds" in settings:
        settings["compile_first_interval_into_command_bounds"] = bool(compile_first)
    if compile_first and not (settings["first_interval_velocity_fractions"]
                              or settings["first_interval_force"]):
        raise ValueError("First-interval bound compilation requires first-interval constraints")
    for key in ("force_limit_N", "velocity_margin_m_s"):
        if isinstance(settings[key], (bool, np.bool_)) or not isinstance(settings[key], Real):
            raise ValueError(f"{key} must be a real number")
    force = float(settings["force_limit_N"])
    margin = float(settings["velocity_margin_m_s"])
    if not math.isfinite(force) or not 0.0 < force <= 2.0:
        raise ValueError("Nominal force bound must be positive and no greater than 2 N")
    if not math.isfinite(margin) or not 0.0 <= margin < physical_velocity_limit:
        raise ValueError("Velocity margin must be nonnegative and smaller than the speed limit")
    if margin and not (settings["velocity_fractions"] or settings["first_interval_velocity_fractions"]):
        raise ValueError("A velocity margin requires velocity sampling rows")
    if force != 2.0 and not (settings["force_fractions"] or settings["first_interval_force"]):
        raise ValueError("A custom nominal force bound requires force sampling rows")
    settings["force_limit_N"], settings["velocity_margin_m_s"] = force, margin
    return settings


class ConstraintDesignedAttitudeMPC(ConfiguredAttitudeMPC):
    """Shared MPC with optional held-input velocity and nominal-force rows.

    Each stage starts at the current measured state (stage zero) or the
    previous shared LPV prediction node. The command is held during that
    interval. Exact nominal two-state servo evolution gives the substep
    velocity or force. The actual Gazebo servo also has common load
    compensation and protections: nominal force is not a force measurement.

    The zero-row unconstrained wrapper uses this same class and objective,
    removing every inequality including these optional rows.
    """

    def __init__(self, P, *, constraint_design=None, **settings):
        if (not math.isfinite(float(P.slider_vel_max)) or not 0.0 < P.slider_vel_max <= .25
                or not math.isfinite(float(P.u2_lim)) or not 0.0 < P.u2_lim <= .15):
            raise ValueError("Rail capability cannot exceed 0.25 m/s or 0.15 m stroke")
        if not 0.0 < float(settings.get("slider_soft_limit", .135)) <= .135:
            raise ValueError("Working command cannot exceed 0.135 m")
        self.constraint_design = validate_constraint_design(
            constraint_design, float(P.slider_vel_max))
        super().__init__(P, **settings)

    def _build_static_qp_structures(self):
        super()._build_static_qp_structures()
        self._base_constraint_rows = self._constraint_A_sparse.shape[0]
        n, stages = self._n_dec, self.Np
        self._constraint_blocks = [
            dict(name="input_increment", row_start=0, row_stop=n,
                 shape=[stages, 3], channels=["tau_z", "chi_cmd", "ups_cmd"],
                 unit=["Nm", "m", "m"]),
            dict(name="absolute_input", row_start=n, row_stop=2*n,
                 shape=[stages, 3], channels=["tau_z", "chi_cmd", "ups_cmd"],
                 unit=["Nm", "m", "m"]),
            dict(name="predicted_slider_state", row_start=2*n, row_stop=10*stages,
                 shape=[stages, 4], channels=["chi", "chi_d", "ups", "ups_d"],
                 unit=["m", "m/s", "m", "m/s"]),
        ]
        servo_block = np.zeros((3, 3))
        omega, damping, mass = float(self.P.wn_mass), float(self.P.zeta_mass), float(self.P.m)
        servo_block[:2, :2] = [[0.0, 1.0], [-omega**2, -2*damping*omega]]
        servo_block[1, 2] = omega**2
        force_row = mass * servo_block[1, :2]
        self._extra_specs = []
        self._first_interval_rows = None
        self._first_interval_velocity_coefficients = []
        row_stop = self._base_constraint_rows
        for name, key in (("intersample_velocity", "velocity_fractions"),
                          ("nominal_servo_force", "force_fractions")):
            fractions = self.constraint_design[key]
            if not fractions:
                continue
            coefficients = []
            for fraction in fractions:
                transition = expm(servo_block * (self.dt * fraction))
                if name == "intersample_velocity":
                    coefficients.append(transition[1, :])
                else:
                    coefficients.append(np.r_[force_row @ transition[:2, :2],
                                              force_row @ transition[:2, 2] + mass*omega**2])
            coefficients = np.asarray(coefficients)
            count = stages * len(fractions) * 2
            limit = (self.slider_vel_lim - self.constraint_design["velocity_margin_m_s"]
                     if name == "intersample_velocity" else self.constraint_design["force_limit_N"])
            self._extra_specs.append((row_stop, row_stop+count, coefficients, limit))
            self._constraint_blocks.append(dict(
                name=name, row_start=row_stop, row_stop=row_stop+count,
                shape=[stages, len(fractions), 2], fractions=fractions,
                channels=["chi_d", "ups_d"] if name == "intersample_velocity" else ["F_chi_nom", "F_ups_nom"],
                unit="m/s" if name == "intersample_velocity" else "N",
                physical_bound=limit,
                stage_start="current measured rail state for stage 0; preceding shared prediction node thereafter",
                force_includes_load_compensation=False))
            row_stop += count
        first_fractions = self.constraint_design["first_interval_velocity_fractions"]
        self._first_interval_velocity_coefficients = np.asarray([
            expm(servo_block*(self.dt*fraction))[1, :].copy() for fraction in first_fractions],
            dtype=float).reshape(-1, 3)
        if any(coefficient[2] <= 0.0 for coefficient in self._first_interval_velocity_coefficients):
            raise ValueError("First-interval command envelope requires positive held-input velocity gain")
        if first_fractions or self.constraint_design["first_interval_force"]:
            self._first_interval_rows = row_stop
            for index, direction in enumerate(("upper", "lower")):
                self._constraint_blocks.append(dict(
                    name="first_interval_command_"+direction,
                    row_start=row_stop+2*index, row_stop=row_stop+2*index+2,
                    shape=[1, 2], channels=["chi_cmd", "ups_cmd"], unit="m",
                    bound_definition="Intersection of exact nominal first-interval velocity command bounds and optional nominal start-force bounds, translated by actually published previous input",
                    stage_start="current measured rail state", fractions=first_fractions,
                    force_includes_load_compensation=False))
            row_stop += 4
        extra_count = row_stop - self._base_constraint_rows
        if not extra_count:
            return
        # Keep every causal coefficient, including current zeros, in CSC.
        # LPV updates can change values without changing the solver pattern.
        rows, columns = [], []
        for begin, end, coefficients, _ in self._extra_specs:
            per_stage = 2 * len(coefficients)
            for column in range(n):
                for stage in range(column // self.nu, stages):
                    start = begin - self._base_constraint_rows + stage * per_stage
                    rows.extend(range(start, start + per_stage))
                    columns.extend([column] * per_stage)
        if self._first_interval_rows is not None:
            rows.extend(range(self._first_interval_rows-self._base_constraint_rows,
                              self._first_interval_rows-self._base_constraint_rows+4))
            columns.extend([1, 2, 1, 2])
        extra = sparse.csc_matrix((np.ones(len(rows)), (rows, columns)), shape=(extra_count, n))
        self._constraint_A_sparse = sparse.vstack([self._constraint_A_sparse, extra], format="csc")
        self._constraint_A_sparse.sort_indices()
        all_rows = self._constraint_A_sparse.indices
        all_cols = np.repeat(np.arange(n), np.diff(self._constraint_A_sparse.indptr))
        self._slider_A_data_indices = np.flatnonzero(
            (all_rows >= 2*n) & (all_rows < self._base_constraint_rows))
        compact_rows = all_rows[self._slider_A_data_indices] - 2*n
        self._slider_Gamma_rows = (compact_rows // 4)*self.ny + 6 + compact_rows % 4
        self._slider_Gamma_cols = all_cols[self._slider_A_data_indices]
        self._extra_A_indices = np.flatnonzero(all_rows >= self._base_constraint_rows)
        self._extra_A_rows = all_rows[self._extra_A_indices] - self._base_constraint_rows
        self._extra_A_cols = all_cols[self._extra_A_indices]
        self._constraint_A_sparse.data[self._extra_A_indices] = 0.0
        self._l_vec_buffer = np.zeros(row_stop)
        self._u_vec_buffer = np.zeros(row_stop)
        self._rail_start_maps = np.zeros((stages, 4, n))
        self._rail_start_free = np.zeros((stages, 4))
        self._extra_dense = np.zeros((extra_count, n))
        if self._first_interval_rows is not None:
            begin = self._first_interval_rows - self._base_constraint_rows
            self._extra_dense[begin+np.arange(4), [1, 2, 1, 2]] = 1.0
            self._constraint_A_sparse.data[self._extra_A_indices] = self._extra_dense[
                self._extra_A_rows, self._extra_A_cols]
        self._command_maps = np.kron(np.tril(np.ones((stages, stages))), np.eye(self.nu)).reshape(stages, self.nu, n)

    def _update_additional_constraints(self, x0, last_u_opt, Y0, l_vec, u_vec):
        if self._first_interval_rows is not None:
            q, v = x0[[12, 14]], x0[[13, 15]]
            lower, upper = np.full(2, -np.inf), np.full(2, np.inf)
            limit = self.slider_vel_lim - self.constraint_design["velocity_margin_m_s"]
            coefficients = self._first_interval_velocity_coefficients
            if len(coefficients):
                a_q, a_v, b_u = coefficients.T
                free = a_q[:, None]*q + a_v[:, None]*v
                lower = np.max((-limit-free)/b_u[:, None], axis=0)
                upper = np.min((limit-free)/b_u[:, None], axis=0)
            if self.constraint_design["first_interval_force"]:
                omega, damping, mass = self.P.wn_mass, self.P.zeta_mass, self.P.m
                acceleration = self.constraint_design["force_limit_N"]/mass
                lower = np.maximum(lower, q+(-acceleration+2*damping*omega*v)/omega**2)
                upper = np.minimum(upper, q+(acceleration+2*damping*omega*v)/omega**2)
            start = self._first_interval_rows
            delta_lower, delta_upper = lower-last_u_opt[1:], upper-last_u_opt[1:]
            compiled = False
            if self.constraint_design.get("compile_first_interval_into_command_bounds", False):
                # The first two rail command rows are exactly delta_u0.
                # Intersect with their original working-command box, not
                # just the nominal interval, to preserve valid l <= u data.
                command_rows = self._n_dec + np.array([1, 2])
                merged_lower = np.maximum(l_vec[command_rows], delta_lower)
                merged_upper = np.minimum(u_vec[command_rows], delta_upper)
                if np.all(merged_lower <= merged_upper):
                    l_vec[command_rows], u_vec[command_rows] = merged_lower, merged_upper
                    l_vec[start:start+4], u_vec[start:start+4] = -np.inf, np.inf
                    compiled = True
            if not compiled:
                # Keep the original four one-sided rows for an incompatible
                # interval/working-box intersection. No invalid two-sided
                # data is passed to OSQP; the common infeasible fallback applies.
                l_vec[start:start+2], u_vec[start:start+2] = -np.inf, delta_upper
                l_vec[start+2:start+4], u_vec[start+2:start+4] = delta_lower, np.inf
        if not self._extra_specs:
            return
        stage_free, stage_maps = self._rail_start_free, self._rail_start_maps
        stage_free[0] = x0[12:16]
        stage_free[1:] = Y0.reshape(self.Np, self.ny)[:-1, 6:10]
        stage_maps[0].fill(0.0)
        stage_maps[1:] = self.Gamma.reshape(self.Np, self.ny, self._n_dec)[:-1, 6:10]
        for begin, end, coefficients, limit in self._extra_specs:
            q_coef, v_coef, u_coef = coefficients.T
            free = (stage_free[:, None, [0, 2]] * q_coef[None, :, None]
                    + stage_free[:, None, [1, 3]] * v_coef[None, :, None]
                    + last_u_opt[None, None, 1:] * u_coef[None, :, None]).ravel()
            response = (stage_maps[:, None, [0, 2], :] * q_coef[None, :, None, None]
                        + stage_maps[:, None, [1, 3], :] * v_coef[None, :, None, None]
                        + self._command_maps[:, None, 1:, :] * u_coef[None, :, None, None])
            self._extra_dense[begin-self._base_constraint_rows:end-self._base_constraint_rows] = response.reshape(-1, self._n_dec)
            l_vec[begin:end] = -limit - free
            u_vec[begin:end] = limit - free
        self._constraint_A_sparse.data[self._extra_A_indices] = self._extra_dense[
            self._extra_A_rows, self._extra_A_cols]

    def constraint_layout_record(self):
        layout = dict(
            schema="mmc-constraint-layout-v6", prediction_steps=self.Np,
            base_rows=self._base_constraint_rows,
            total_rows=self._constraint_A_sparse.shape[0], decision_size=self._n_dec,
            settings=self.constraint_design,
            servo=dict(mass_kg=float(self.P.m), omega_n_rad_s=float(self.P.wn_mass),
                       damping_ratio=float(self.P.zeta_mass)),
            blocks=self._constraint_blocks)
        if self.constraint_design.get("compile_first_interval_into_command_bounds", False):
            start = self._first_interval_rows
            layout["first_interval_compilation"] = dict(
                schema="mmc-first-interval-bound-compilation-v1",
                original_absolute_input_rows=[self._n_dec+1, self._n_dec+2],
                channels=["chi_cmd", "ups_cmd"], unit="m",
                original_working_command_bound_m=float(self.u2_lim),
                fallback_upper_rows=[start, start+1],
                fallback_lower_rows=[start+2, start+3],
                nonempty_policy="Both nominal intervals intersect their original working-command boxes; compile the tighter bounds into the existing first-step absolute-input rows",
                empty_policy="Keep original working bounds and activate the four one-sided interval rows; common infeasible-solve fallback",
                fallback_sparse_pattern_retained=True,
                binding_policy="Distinguish nominal compiled-bound activity from the original physical working bound using the actual bound side and signed original-unit dual; retained unbounded rows are inactive")
        return copy.deepcopy(layout)
