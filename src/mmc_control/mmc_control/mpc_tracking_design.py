"""Optional, shared MPC tracking design for the controlled experiments.

The production controller is unchanged unless this module is explicitly used.
The integral uses measured position and the common position reference only;
it has no disturbance estimate, wind input, or simulator truth input.
"""

from __future__ import annotations

import inspect
import math
from typing import Optional, Sequence

import numpy as np

from .mmc_uav import AttitudeMPC


DISCRETIZATIONS = ("forward_euler", "zoh4")


def discretize_affine_lpv(a_c, b_c, f0, x0, u0, dt, method="forward_euler"):
    """Discretize a locally affine plant with a held input.

    ``zoh4`` is the fourth order Taylor approximation of the exact held-input
    matrix exponential.  The affine residual is discretized with the same
    integral of the transition matrix as B; it is never dropped or treated as
    an external disturbance measurement.
    """
    if method not in DISCRETIZATIONS:
        raise ValueError(f"Unknown LPV discretization: {method!r}")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("LPV sample time must be finite and positive")
    a_c = np.asarray(a_c, dtype=float)
    b_c = np.asarray(b_c, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    f0 = np.asarray(f0, dtype=float).reshape(-1)
    if (a_c.shape != (x0.size, x0.size)
            or b_c.shape != (x0.size, u0.size) or f0.shape != x0.shape):
        raise ValueError("Inconsistent locally affine plant dimensions")
    if not all(np.isfinite(v).all() for v in (a_c, b_c, f0, x0, u0)):
        raise ValueError("Locally affine plant must be finite")
    c_c = f0 - a_c @ x0 - b_c @ u0
    eye = np.eye(x0.size)
    if method == "forward_euler":
        return eye + dt * a_c, dt * b_c, dt * c_c
    scaled_a = dt * a_c
    a2 = scaled_a @ scaled_a
    a3 = a2 @ scaled_a
    a4 = a3 @ scaled_a
    a_d = eye + scaled_a + 0.5 * a2 + a3 / 6.0 + a4 / 24.0
    integral_transition = dt * (eye + 0.5 * scaled_a + a2 / 6.0 + a3 / 24.0)
    return a_d, integral_transition @ b_c, integral_transition @ c_c


class ConfiguredAttitudeMPC(AttitudeMPC):
    """Original condensed MPC with an explicitly selected discretization.

    All costs, output selection, input memory, constraints, solver settings,
    safety fallback, and prediction assembly remain in ``AttitudeMPC``.
    The default path calls the original LPV update verbatim.
    """

    def __init__(self, P, *, discretization="forward_euler", **settings):
        if discretization not in DISCRETIZATIONS:
            raise ValueError(f"Unknown LPV discretization: {discretization!r}")
        allowed = set(inspect.signature(AttitudeMPC.__init__).parameters) - {"self", "P"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unsupported MPC parameters: {sorted(unknown)}")
        if "Np" in settings:
            horizon = settings["Np"]
            if (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer))
                    or horizon <= 0):
                raise ValueError("Np must be a positive integer")
        for name, value in settings.items():
            if value is None and name == "dt_mpc":
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name in ("dt_mpc", "slider_soft_limit", "tau_z_lim") and number <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.discretization = discretization
        super().__init__(P, **settings)

    def _update_lpv_model(self, x0, thrust_cmd, last_u):
        if self.discretization == "forward_euler":
            return super()._update_lpv_model(x0, thrust_cmd, last_u)
        a_c, b_c, f0 = self._linearize_AB(x0, last_u, thrust_cmd)
        a_d, b_d, c_d = discretize_affine_lpv(
            a_c, b_c, f0, x0, last_u, self.dt, self.discretization)
        self._A_d_base_buffer[:] = a_d
        self._B_d_base_buffer[:] = b_d
        self._c_d_base_buffer[:] = c_d
        self.A_d[:self.nx_state, :self.nx_state] = a_d
        self.A_d[:self.nx_state, self.nx_state:] = b_d
        self.A_d[self.nx_state:, :self.nx_state] = 0.0
        self.A_d[self.nx_state:, self.nx_state:] = self._eye_nu
        self.B_d[:self.nx_state, :] = b_d
        self.B_d[self.nx_state:, :] = self._eye_nu
        self.c_d[:self.nx_state] = c_d
        self.c_d[self.nx_state:] = 0.0
        self.Phi, self.Gamma = self._build_prediction_matrices(self.A_d, self.B_d)
        self.H_mat = self._assemble_hessian_from_gamma()
        self._cached_P_upper = None
        self._cached_P_upper_data = None
        self._cached_Y_const = self._build_constant_response_vector()


def make_tracking_mpc(params, configuration=None, *, Np=35, dt_mpc=0.01):
    """Construct an isolated MPC from original constructor settings plus method."""
    settings = dict(configuration or {})
    settings.setdefault("Np", Np)
    settings.setdefault("dt_mpc", dt_mpc)
    return ConfiguredAttitudeMPC(params, **settings)


def _pair(value, name, *, nonnegative=False, positive=False):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(2, float(array))
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or pair")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array.copy()


class HorizontalTrackingIntegral:
    """Bounded ordinary position-error integral owned by the MPC design.

    The output is an additional roll/pitch tracking target in radians. The
    caller keeps the common position and attitude-reference generators and
    separately records their outputs and the actual MPC target. Positive
    world x acceleration means positive pitch at zero yaw; positive world y
    acceleration means negative roll, matching the production ENU mapping.

    ``blocked_axes`` is a caller-declared integration block. If saturation
    directions are supplied, only increments pointing farther into a bound
    are blocked, so reverse error can unwind an existing integral.
    """

    def __init__(self, ki=0.0, acceleration_limit=0.8, *, enabled=None,
                 nominal_dt=0.01, trim_limit_deg=8.0, integral_limit=None, g=9.81):
        self.ki = _pair(ki, "ki", nonnegative=True)
        self.acceleration_limit = _pair(acceleration_limit, "acceleration_limit", positive=True)
        if enabled is not None and not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be boolean")
        self.enabled = bool(np.any(self.ki > 0.0)) if enabled is None else bool(enabled)
        if self.enabled and not np.any(self.ki > 0.0):
            raise ValueError("An enabled horizontal integral requires a positive gain")
        self.g = float(g)
        self.nominal_dt = float(nominal_dt)
        self.trim_limit_rad = np.radians(_pair(trim_limit_deg, "trim_limit_deg", positive=True))
        if not math.isfinite(self.g) or self.g <= 0.0:
            raise ValueError("g must be finite and positive")
        if not math.isfinite(self.nominal_dt) or self.nominal_dt <= 0.0:
            raise ValueError("nominal_dt must be finite and positive")
        natural_limit = np.divide(self.acceleration_limit, self.ki,
                                  out=np.zeros(2), where=self.ki > 0.0)
        self.integral_limit = (natural_limit if integral_limit is None else
                               np.minimum(_pair(integral_limit, "integral_limit", nonnegative=True),
                                          natural_limit))
        self.reset()

    def reset(self):
        self.integral_xy = np.zeros(2)
        self.acceleration_xy = np.zeros(2)
        self.last_error = np.zeros(2)
        self.last_phi_rate_rad_s = 0.0
        self.last_theta_rate_rad_s = 0.0
        self.last_record = {}
        self._last_time: Optional[float] = None
        self._last_trim = np.zeros(2)

    def _angles(self, integral, yaw):
        acceleration = np.clip(self.ki * integral, -self.acceleration_limit,
                               self.acceleration_limit)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        forward = acceleration[0] * cosine + acceleration[1] * sine
        lateral = acceleration[1] * cosine - acceleration[0] * sine
        theta = math.atan(forward / self.g)
        phi = math.atan(-lateral * math.cos(theta) / self.g)
        return np.array([phi, theta]), acceleration

    def step(self, t, measured_xy, reference_xy, yaw, blocked_axes=(False, False), *,
             dt=None, phi_reference_rad=0.0, theta_reference_rad=0.0,
             roll_limit_rad=None, pitch_limit_rad=None,
             saturation_direction_world_xy: Optional[Sequence[float]] = None):
        t, yaw = float(t), float(yaw)
        if not math.isfinite(t) or not math.isfinite(yaw):
            raise ValueError("Control time and yaw must be finite")
        measured = _pair(measured_xy, "measured_xy")
        reference = _pair(reference_xy, "reference_xy")
        self.last_error[:] = reference - measured
        blocked = np.asarray(blocked_axes, dtype=bool).copy()
        if blocked.shape != (2,):
            raise ValueError("blocked_axes must be a pair")
        if self._last_time is not None and t < self._last_time:
            raise ValueError("Control clock moved backward; reset the integral")
        elapsed = (self.nominal_dt if self._last_time is None else t - self._last_time)
        if dt is not None:
            elapsed = float(dt)
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise ValueError("Integral dt must be finite and nonnegative")
        if self._last_time is not None and t == self._last_time:
            elapsed = 0.0
        baseline = np.array([float(phi_reference_rad), float(theta_reference_rad)])
        if not np.isfinite(baseline).all():
            raise ValueError("Baseline roll and pitch references must be finite")
        angle_limits = np.array([math.inf, math.inf])
        for index, limit in enumerate((roll_limit_rad, pitch_limit_rad)):
            if limit is not None:
                angle_limits[index] = float(limit)
                if not math.isfinite(float(limit)) or float(limit) <= 0.0:
                    raise ValueError("Attitude limits must be finite and positive")
        increment = self.last_error * elapsed if self.enabled else np.zeros(2)
        if saturation_direction_world_xy is not None:
            direction = _pair(saturation_direction_world_xy, "saturation_direction_world_xy")
            blocked = blocked & (increment * direction > 0.0)
        increment[blocked] = 0.0
        candidate = np.clip(self.integral_xy + increment,
                            -self.integral_limit, self.integral_limit)
        # Check each proposed axis increment against roll/pitch saturation.
        # This preserves reverse integration while blocking further windup.
        original_angles, _ = self._angles(self.integral_xy, yaw)
        candidate_angles, _ = self._angles(candidate, yaw)
        total = baseline + candidate_angles
        for axis in range(2):
            single = self.integral_xy.copy()
            single[axis] = candidate[axis]
            change = self._angles(single, yaw)[0] - original_angles
            outward = ((total > angle_limits) & (change > 0.0)) | (
                (total < -angle_limits) & (change < 0.0))
            trim_outward = ((candidate_angles > self.trim_limit_rad) & (change > 0.0)) | (
                (candidate_angles < -self.trim_limit_rad) & (change < 0.0))
            if np.any(outward | trim_outward):
                candidate[axis] = self.integral_xy[axis]
                blocked[axis] = True
        self.integral_xy[:] = candidate if self.enabled else 0.0
        trim, acceleration = self._angles(self.integral_xy, yaw)
        trim = np.clip(trim, -self.trim_limit_rad, self.trim_limit_rad)
        # Keep the requested total attitude inside the caller's common bounds.
        trim = np.clip(baseline + trim, -angle_limits, angle_limits) - baseline
        if not self.enabled:
            trim.fill(0.0)
        self.acceleration_xy[:] = acceleration
        rate = ((trim - self._last_trim) / elapsed if elapsed > 0.0 else np.zeros(2))
        self.last_phi_rate_rad_s, self.last_theta_rate_rad_s = map(float, rate)
        self.last_record = {
            "controller_time_s": t, "enabled": self.enabled, "dt_s": elapsed,
            "error_xy_m": self.last_error.tolist(),
            "integral_xy_m_s": self.integral_xy.tolist(),
            "acceleration_xy_m_s2": acceleration.tolist(),
            "trim_phi_theta_rad": trim.tolist(), "trim_rate_phi_theta_rad_s": rate.tolist(),
            "blocked_xy": blocked.tolist(),
            "integral_bound_xy": (np.abs(self.integral_xy) >= self.integral_limit).tolist(),
        }
        self._last_time = t
        self._last_trim[:] = trim
        return float(trim[0]), float(trim[1])
