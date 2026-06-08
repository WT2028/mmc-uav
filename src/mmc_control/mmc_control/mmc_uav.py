# ENU 坐标系：X 向东，Y 向北，Z 向上
import csv
import math
import signal
import time
from contextlib import nullcontext
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import casadi as ca
import osqp
import numpy as np
import rclpy
from scipy import sparse
try:
    from actuator_msgs.msg import Actuators
except ImportError:  # pragma: no cover - 测试桩环境兜底
    class Actuators:  # type: ignore[override]
        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id="")
            self.position = []
            self.velocity = []
            self.normalized = []
from geometry_msgs.msg import Twist, Vector3
try:
    from geometry_msgs.msg import PoseStamped
except ImportError:  # pragma: no cover - 测试桩环境兜底
    class PoseStamped:  # type: ignore[override]
        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id="")
            self.pose = SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
from mmc_interfaces.msg import WindCommand, WindStatus
from nav_msgs.msg import Odometry
try:
    from nav_msgs.msg import Path
except ImportError:  # pragma: no cover - 测试桩环境兜底
    class Path:  # type: ignore[override]
        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id="")
            self.poses = []
try:
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:  # pragma: no cover - 测试桩环境兜底
    class Marker:  # type: ignore[override]
        ADD = 0
        ARROW = 0
        SPHERE = 2

        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id="")
            self.ns = ""
            self.id = 0
            self.type = 0
            self.action = 0
            self.pose = SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            self.scale = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.color = SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)

    class MarkerArray:  # type: ignore[override]
        def __init__(self):
            self.markers = []
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import EntityWrench, Entity
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64

try:
    from mmc_control.project_paths import get_default_log_dir
    from mmc_control.wind_config import WindConfigSummary, parse_world_wind_config
except ImportError:  # pragma: no cover — 直接源码执行时的后备方案
    from project_paths import get_default_log_dir
    from wind_config import WindConfigSummary, parse_world_wind_config

RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)

# 限幅（直接截断）辅助函数
def clamp(value, minlim, maxlim):
    if value <= minlim:
        return minlim
    elif value >= maxlim:
        return maxlim
    else:
        return value


def optional_lock(owner):
    lock = getattr(owner, "_callback_state_lock", None)
    return lock if lock is not None else nullcontext()


def wrap_angle_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def unwrap_angle_near(angle: float, anchor: float) -> float:
    return float(anchor + wrap_angle_pi(float(angle) - float(anchor)))


def unwrap_angle_sequence(angles: Sequence[float], anchor: float) -> np.ndarray:
    values = np.asarray(angles, dtype=float).copy()
    if values.ndim == 0:
        values = values.reshape(1)
    previous = float(anchor)
    for idx in range(values.shape[0]):
        values[idx] = unwrap_angle_near(values[idx], previous)
        previous = float(values[idx])
    return values


def slew_limit_angle(target: float, anchor: float, max_delta: float) -> float:
    target_unwrapped = unwrap_angle_near(float(target), float(anchor))
    if max_delta <= 0.0:
        return float(target_unwrapped)
    return float(anchor + clamp(target_unwrapped - float(anchor), -max_delta, max_delta))


def slew_limit_angle_sequence(angles: Sequence[float], anchor: float, max_delta: float) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    limited = np.zeros_like(values, dtype=float)
    previous = float(anchor)
    for idx in range(values.shape[0]):
        previous = slew_limit_angle(values[idx], previous, max_delta)
        limited[idx] = previous
    return limited


def angle_sequence_to_rate_sequence(
    angles: Sequence[float],
    dt: float,
    *,
    rate_limit: Optional[float] = None,
) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    rates = np.zeros_like(values, dtype=float)
    dt = max(float(dt), 1e-6)

    if values.shape[0] >= 2:
        rates[0] = (values[1] - values[0]) / dt
        rates[-1] = (values[-1] - values[-2]) / dt
        if values.shape[0] >= 3:
            rates[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)

    if rate_limit is not None and float(rate_limit) > 0.0:
        limit = float(rate_limit)
        rates = np.clip(rates, -limit, limit)
    return rates


def blend_angle_near(anchor: float, target: float, alpha: float) -> float:
    alpha = clamp(float(alpha), 0.0, 1.0)
    target_unwrapped = unwrap_angle_near(float(target), float(anchor))
    return float(float(anchor) + alpha * (target_unwrapped - float(anchor)))


def predict_angle_tracking_sequence(
    targets: Sequence[float],
    initial_angle: float,
    dt: float,
    *,
    rate_limit: Optional[float] = None,
) -> np.ndarray:
    max_delta = 0.0
    if rate_limit is not None and float(rate_limit) > 0.0:
        max_delta = float(rate_limit) * max(float(dt), 1e-6)
    if max_delta > 0.0:
        return slew_limit_angle_sequence(
            targets,
            anchor=float(initial_angle),
            max_delta=max_delta,
        )
    return unwrap_angle_sequence(targets, float(initial_angle))


def quintic_smoothstep(alpha: float) -> float:
    alpha = clamp(float(alpha), 0.0, 1.0)
    return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def shape_slider_command(
    current: float,
    target: float,
    dt: float,
    tau: float,
    rate_limit: float,
    limit: float,
) -> float:
    if limit <= 0.0:
        return 0.0

    dt = max(float(dt), 0.0)
    target = clamp(float(target), -float(limit), float(limit))

    if tau <= 1e-6 or dt <= 1e-9:
        shaped_target = target
    else:
        alpha = clamp(dt / (tau + dt), 0.0, 1.0)
        shaped_target = float(current) + alpha * (target - float(current))

    if rate_limit > 0.0 and dt > 0.0:
        max_delta = float(rate_limit) * dt
        shaped_target = float(current) + clamp(
            shaped_target - float(current),
            -max_delta,
            max_delta,
        )

    shaped_target = clamp(shaped_target, -float(limit), float(limit))
    if abs(shaped_target) < 1e-6:
        return 0.0
    return float(shaped_target)

def quaternion_to_euler_enu(qx: float, qy: float, qz: float, qw: float):
    """
    四元数转欧拉角（滚转、俯仰、偏航），采用 ENU 约定。
    roll=phi，pitch=theta，yaw=psi。
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def body_velocity_to_world_enu(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    vx_body: float,
    vy_body: float,
    vz_body: float,
) -> Tuple[float, float, float]:
    """
    将 Odometry 中机体系线速度旋转到 ENU 世界系。
    """
    phi, theta, psi = quaternion_to_euler_enu(qx, qy, qz, qw)
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    tb_v = np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
        [-stheta,       ctheta * sphi,                       ctheta * cphi],
    ])
    velocity_world = tb_v @ np.array([float(vx_body), float(vy_body), float(vz_body)], dtype=float)
    return float(velocity_world[0]), float(velocity_world[1]), float(velocity_world[2])

# ===================== 参数定义 =====================
@dataclass
class Params:
    g: float = 9.81
    l: float = 0.4 #机体边长[m]
    m_b: float = 1.25 #机体（不含滑块）的质量[kg]
    m: float = 0.05 #单滑块质量[kg]
    Ib_x: float = 0.0585 #机体（不含滑块）的基础转动惯量[kg·m^2]
    Ib_y: float = 0.0585
    Ib_z: float = 0.06
    Im_x: float = 0.0006 #单滑块的基础转动惯量[kg·m^2]
    Im_y: float = 0.0006
    Im_z: float = 0.0012
    b_thrust: float = 1.5e-4  # 推力系数 b
    d_yaw: float = 3e-7    # 反扭力矩系数 d
    wn_mass: float = 20.0    #滑块执行器也就是滑块的伺服电机的自然频率，未来需要实机测试得到
    zeta_mass: float = 1.20  #滑块执行器的阻尼比，未来需要实机测试得到
    slider_vel_max: float = 0.25 # 滑块物理最大速度[m/s]
    dt: float = 0.001
    T: float = 25.0 #仿真总时长
    u2_lim: float = 0.15 #滑块移动限制在±0.15m内
    thrust_min: float = 0.0
    thrust_to_weight_ratio_max: float = 2.14

    # 气动阻力模型系数
    rho: float = 1.225  # 空气密度，单位 kg/m^3

    # 阻力系数（Cd），参考方形钝体
    Cd_side: float = 1.2  # 侧面阻力系数（X/Y 轴）
    Cd_top: float = 1.3  # 垂直阻力系数（Z 轴，平板流）

    # 迎风面积（参考面积）[m^2]
    # 假设机身高度 h=0.1 m，边长 l=0.4 m
    S_side: float = 0.072  # 侧面参考面积 = l * h = 0.4 * 0.09
    S_top: float = 0.16  # 顶面参考面积 = l * l = 0.4 * 0.4

    kp: float = 0.04
    kq: float = 0.04
    kr: float = 0.02

    @property
    def M(self):  # 总质量
        return self.m_b + 4.0 * self.m

    @property
    def mu(self):  # μ 质量比
        return self.m / self.M

    @property
    def thrust_max(self):  # 最大总推力，需随当前总质量动态更新
        return self.thrust_to_weight_ratio_max * self.M * self.g


@dataclass(frozen=True)
class SliderActuatorCostProfile:
    wn_mass: float
    zeta_mass: float
    settling_time_s: float
    actuator_penalty_scale: float
    tracking_scale: float
    rate_penalty_scale: float
    command_penalty_scale: float
    delta_penalty_scale: float
    q_chi: float
    q_chi_d: float
    q_ups: float
    q_ups_d: float
    r_chi: float
    r_ups: float
    rd_chi: float
    rd_ups: float


def build_slider_actuator_cost_profile(
    P: Params,
    *,
    q_chi: float,
    q_chi_d: float,
    q_ups: float,
    q_ups_d: float,
    r_chi: float,
    r_ups: float,
    rd_chi: float,
    rd_ups: float,
) -> SliderActuatorCostProfile:
    """
    将滑块执行器的二阶动态参数映射为 MPC 代价缩放。

    预测模型已经显式包含 wn_mass / zeta_mass，
    这里再把它们折算进滑块相关的状态、绝对控制量和控制增量权重，
    让优化器从一开始就把“执行器不是理想位置源”考虑进去。
    """
    wn = max(float(P.wn_mass), 1e-3)
    zeta = max(float(P.zeta_mass), 0.05)

    nominal_wn = 20.0
    nominal_zeta = 0.7
    nominal_settling = 4.0 / (nominal_wn * nominal_zeta)
    settling_time = 4.0 / (wn * zeta)

    settling_ratio = max(1.0, settling_time / nominal_settling)
    damping_penalty = max(1.0, nominal_zeta / zeta)
    actuator_penalty_scale = min(8.0, settling_ratio * damping_penalty)

    # 执行器越慢或越欠阻尼：
    # 1) 适当放松滑块位置跟踪欲望
    # 2) 提高滑块速度、绝对命令、命令增量的内部惩罚
    tracking_scale = max(0.55, 1.0 / math.sqrt(actuator_penalty_scale))
    rate_penalty_scale = min(8.0, actuator_penalty_scale)
    command_penalty_scale = min(4.0, math.sqrt(actuator_penalty_scale))
    delta_penalty_scale = min(8.0, actuator_penalty_scale)

    return SliderActuatorCostProfile(
        wn_mass=wn,
        zeta_mass=zeta,
        settling_time_s=settling_time,
        actuator_penalty_scale=actuator_penalty_scale,
        tracking_scale=tracking_scale,
        rate_penalty_scale=rate_penalty_scale,
        command_penalty_scale=command_penalty_scale,
        delta_penalty_scale=delta_penalty_scale,
        q_chi=float(q_chi) * tracking_scale,
        q_chi_d=float(q_chi_d) * rate_penalty_scale,
        q_ups=float(q_ups) * tracking_scale,
        q_ups_d=float(q_ups_d) * rate_penalty_scale,
        r_chi=float(r_chi) * command_penalty_scale,
        r_ups=float(r_ups) * command_penalty_scale,
        rd_chi=float(rd_chi) * delta_penalty_scale,
        rd_ups=float(rd_ups) * delta_penalty_scale,
    )


P = Params()
QUINTIC_BLEND_MAX_ABS_ACCEL = 10.0 / math.sqrt(3.0)

FLIGHT_LOG_HEADERS = [
    'Time',
    'X', 'Y', 'Z',
    'X_ref', 'Y_ref', 'Z_ref',
    'VX', 'VY', 'VZ',
    'Roll_deg', 'Pitch_deg', 'Yaw_deg',
    'Roll_rate_deg_s', 'Pitch_rate_deg_s', 'Yaw_rate_deg_s',
    'Roll_ref_deg', 'Pitch_ref_deg', 'Yaw_ref_deg',
    'Roll_err_deg', 'Pitch_err_deg',
    'Outer_sum_ex', 'Outer_sum_ey', 'Outer_sum_ez',
    'Scene_hover_ready',
    'Adaptive_phase_active',
    'Adaptive_phase_time',
    'Adaptive_phase_rate',
    'Adaptive_phase_metric',
    'Manual_mode',
    'Manual_input_forward', 'Manual_input_lateral',
    'Manual_hold_x', 'Manual_hold_y',
    'Manual_horiz_speed',
    'Brake_phi_ref_deg', 'Brake_theta_ref_deg',
    'VX_body', 'VY_body',
    'Actuation_backend',
    'Yaw_control_mode',
    'Auto_scene_mode',
    'NDO_enabled',
    'Body_mass_kg',
    'Moving_mass_kg',
    'System_mass_kg',
    'Mass_ratio_mu',
    'Slider_wn_mass',
    'Slider_zeta_mass',
    'World_sdf_path',
    'Auto_scene_yaw_ref_mode',
    'Figure_eight_forward_tilt_deg',
    'Upper_rotor_cmd',
    'Lower_rotor_cmd',
    'Upper_rotor_actual',
    'Lower_rotor_actual',
    'Upper_rotor_speed_cmd',
    'Lower_rotor_speed_cmd',
    'Upper_rotor_speed_actual',
    'Lower_rotor_speed_actual',
    'Upper_rotor_speed_cmd_rpm',
    'Lower_rotor_speed_cmd_rpm',
    'Upper_rotor_speed_actual_rpm',
    'Lower_rotor_speed_actual_rpm',
    'Rotor_total_thrust_est',
    'Rotor_tau_z_est',
    'Rotor_speed_util_actual',
    'Rotor_thrust_util_est',
    'Rotor_thrust_margin_est',
    'Rotor_max_speed_rad_s',
    'Thrust_cmd_outer',
    'Thrust_cmd',
    'Thrust_retarget_ratio',
    'Tau_z_cmd',
    'Inner_loop_dt',
    'Inner_exec_dt',
    'Inner_mpc_dt',
    'Inner_mpc_model_dt',
    'Inner_mpc_qp_build_dt',
    'Inner_mpc_setup_dt',
    'Inner_mpc_solve_dt',
    'Inner_mpc_model_reused',
    'Inner_mpc_lpv_reuse_count',
    'Inner_mpc_lpv_reuse_max_skips',
    'Inner_mpc_prediction_horizon',
    'Inner_mpc_control_horizon',
    'Inner_mpc_refresh_reason_mask',
    'Inner_observer_dt',
    'Inner_drag_dt',
    'Inner_publish_dt',
    'Inner_ref_build_dt',
    'Inner_log_dt',
    'Wrench_dt_for_scale',
    'Wrench_scale',
    'Wrench_force_z_world',
    'Wrench_force_z_published',
    'Raw_roll_ref_deg',
    'Raw_pitch_ref_deg',
    'Shaped_roll_ref_deg',
    'Shaped_pitch_ref_deg',
    'Shaped_p_ref_deg_s',
    'Shaped_q_ref_deg_s',
    'Inner_roll_ref_deg',
    'Inner_pitch_ref_deg',
    'Inner_yaw_ref_deg',
    'Slider_X_ref_coord',
    'Slider_X_cmd_raw',
    'Slider_X_cmd', 'Slider_X_actual',
    'Slider_X_vel_actual',
    'Slider_Y_ref_coord',
    'Slider_Y_cmd_raw',
    'Slider_Y_cmd', 'Slider_Y_actual',
    'Slider_Y_vel_actual',
    'Wind_config_valid',
    'Wind_enable',
    'Wind_vx_world',
    'Wind_vy_world',
    'Wind_vz_world',
    'Wind_speed_world',
    'Wind_force_scaling',
    'Wind_mag_rise_time',
    'Wind_mag_sin_amp_pct',
    'Wind_mag_noise_stddev',
    'Wind_dir_rise_time',
    'Wind_dir_sin_amp_deg',
    'Wind_dir_noise_stddev',
    'Wind_vertical_noise_stddev',
    'Wind_activation_mode',
    'Wind_hover_target_z',
    'Wind_hover_z_tol',
    'Wind_hover_speed_tol',
    'Wind_hover_hold_time',
    'Wind_command_seq',
    'Wind_command_source',
    'Wind_command_pending',
    'Wind_status_publish_ok',
    'Wind_status_detail',
    'Wind_runtime_active',
    'Wind_activation_time',
    'NDO_d_force_hat_x',
    'NDO_d_force_hat_y',
    'NDO_d_force_hat_z',
    'NDO_d_torque_hat_x',
    'NDO_d_torque_hat_y',
    'NDO_d_torque_hat_z',
    'NDO_comp_phi_force_deg',
    'NDO_comp_theta_force_deg',
    'NDO_comp_phi_torque_deg',
    'NDO_comp_theta_torque_deg',
    'NDO_comp_phi_total_deg',
    'NDO_comp_theta_total_deg',
]
FLIGHT_LOG_COLUMN_INDEX = {name: idx for idx, name in enumerate(FLIGHT_LOG_HEADERS)}


def rotor_speed_rad_s_to_rpm(speed_rad_s: float) -> float:
    return float(speed_rad_s) * RAD_S_TO_RPM


def scaled_inertia(base_inertia: float, mass: float, base_mass: float = 0.05) -> float:
    """Scale the moving-mass body inertia linearly when scanning slider mass."""
    if base_mass <= 1e-12:
        return float(base_inertia)
    return float(base_inertia) * max(float(mass), 1e-9) / float(base_mass)


def clamp_slider_state(position: float, velocity: float, pos_limit: float, vel_limit: float) -> Tuple[float, float]:
    pos = clamp(float(position), -float(pos_limit), float(pos_limit))
    vel = clamp(float(velocity), -float(vel_limit), float(vel_limit))
    if abs(pos) >= float(pos_limit) and pos * vel > 0.0:
        vel = 0.0
    return pos, vel

# ===================== 六自由度状态与输入 =====================
@dataclass
class State6:
    # 位置
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # 速度（惯性系）
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # 姿态欧拉角
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    # 角速度（机体系）
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    # 滑块位置及速度
    chi: float = 0.0
    chi_d: float = 0.0
    ups: float = 0.0
    ups_d: float = 0.0


@dataclass
class AxisCoordinatorConfig:
    axis_name: str
    angle_index: int
    rate_index: int
    slider_index: int
    prediction_horizon: int
    dt: float
    ref_wn: float
    ref_zeta: float
    angle_rate_limit: float
    ref_accel_limit: float
    k_angle: float
    k_rate: float
    slider_soft_limit: float
    slider_rate_limit: float
    thrust_nominal: float
    slider_sign: float
    state_measurement_blend: float

    @classmethod
    def for_pitch(cls, params: "Params", prediction_horizon: int, dt: float) -> "AxisCoordinatorConfig":
        # Coordinator is a reference-planning layer, not the physical actuator.
        # Let it look ahead faster than the final command shaper; otherwise the
        # horizon intent gets pinned near ~1 cm by duplicate rate limiting.
        rate_limit = max(4.0 * float(params.slider_vel_max), 1e-6)
        return cls(
            axis_name="pitch",
            angle_index=1,
            rate_index=4,
            slider_index=6,
            prediction_horizon=int(prediction_horizon),
            dt=float(dt),
            ref_wn=max(1.6 * float(params.wn_mass), 1.0),
            ref_zeta=max(float(params.zeta_mass), 0.1),
            angle_rate_limit=2.0,
            ref_accel_limit=max(3.2 * rate_limit / max(float(dt), 1e-3), rate_limit),
            k_angle=0.24,
            k_rate=0.09,
            slider_soft_limit=float(params.u2_lim),
            slider_rate_limit=rate_limit,
            thrust_nominal=float(params.M * params.g),
            slider_sign=1.0,
            state_measurement_blend=0.01,
        )

    @classmethod
    def for_roll(cls, params: "Params", prediction_horizon: int, dt: float) -> "AxisCoordinatorConfig":
        rate_limit = max(4.0 * float(params.slider_vel_max), 1e-6)
        return cls(
            axis_name="roll",
            angle_index=0,
            rate_index=3,
            slider_index=8,
            prediction_horizon=int(prediction_horizon),
            dt=float(dt),
            ref_wn=max(1.6 * float(params.wn_mass), 1.0),
            ref_zeta=max(float(params.zeta_mass), 0.1),
            angle_rate_limit=2.0,
            ref_accel_limit=max(3.2 * rate_limit / max(float(dt), 1e-3), rate_limit),
            k_angle=0.24,
            k_rate=0.09,
            slider_soft_limit=float(params.u2_lim),
            slider_rate_limit=rate_limit,
            thrust_nominal=float(params.M * params.g),
            slider_sign=-1.0,
            state_measurement_blend=0.01,
        )


@dataclass
class AxisCoordinatorState:
    shaped_angle: float = 0.0
    shaped_rate: float = 0.0
    slider_ref: float = 0.0
    slider_rate: float = 0.0
    last_raw_angle: Optional[float] = None


class AxisManeuverCoordinator:
    def __init__(self, cfg: AxisCoordinatorConfig):
        self.cfg = cfg
        self.state = AxisCoordinatorState()

    @staticmethod
    def _sign_reversal_active(raw_target: float, measured_angle: float, shaped_angle: float) -> bool:
        deadband = math.radians(0.05)
        if abs(raw_target) <= deadband:
            return False
        return (
            raw_target * measured_angle < -(deadband ** 2)
            or raw_target * shaped_angle < -(deadband ** 2)
        )

    def _update_axis_state(
        self,
        axis_state: AxisCoordinatorState,
        raw_target: float,
        prev_raw_target: float,
    ) -> Tuple[float, float]:
        dt = max(float(self.cfg.dt), 1e-6)
        wn = max(float(self.cfg.ref_wn), 1e-6)
        zeta = max(float(self.cfg.ref_zeta), 0.0)
        acc_lim = max(float(self.cfg.ref_accel_limit), 0.0)
        rate_lim = max(float(self.cfg.angle_rate_limit), 0.0)

        angle_prev = float(axis_state.shaped_angle)
        rate_prev = float(axis_state.shaped_rate)

        angle_err = float(raw_target) - angle_prev
        accel = wn * wn * angle_err - 2.0 * zeta * wn * rate_prev
        if acc_lim > 0.0:
            accel = clamp(accel, -acc_lim, acc_lim)

        rate_next = rate_prev + accel * dt
        if rate_lim > 0.0:
            rate_next = clamp(rate_next, -rate_lim, rate_lim)
        angle_next = angle_prev + rate_next * dt

        # 对真正上升/下降的 raw 指令段保持同向单调；raw 常值段则允许
        # shaped 参考继续朝目标推进，避免 outer 40ms / inner 10ms 时
        # 因“同值保持”被错误冻结而形成 braking ratchet。
        if raw_target > prev_raw_target + 1e-9 and angle_next < angle_prev:
            angle_next = angle_prev
            rate_next = max(0.0, rate_next)
        elif raw_target < prev_raw_target - 1e-9 and angle_next > angle_prev:
            angle_next = angle_prev
            rate_next = min(0.0, rate_next)

        return angle_next, rate_next

    def _update_slider_state(
        self,
        axis_state: AxisCoordinatorState,
        angle_ref: float,
        rate_ref: float,
        measured_angle: float,
        measured_rate: float,
        thrust_cmd: float,
    ) -> Tuple[float, float]:
        dt = max(float(self.cfg.dt), 1e-6)
        slider_lim = max(float(self.cfg.slider_soft_limit), 0.0)
        thrust_ratio = max(float(thrust_cmd), 0.0) / max(float(self.cfg.thrust_nominal), 1e-6)
        thrust_gain = clamp(thrust_ratio, 0.5, 1.5)
        angle_err = float(angle_ref) - float(measured_angle)
        rate_err = float(rate_ref) - float(measured_rate)
        slider_target = self.cfg.slider_sign * (
            float(self.cfg.k_angle) * angle_err + float(self.cfg.k_rate) * rate_err
        )
        slider_target *= thrust_gain
        slider_target = clamp(slider_target, -slider_lim, slider_lim)

        slider_prev = float(axis_state.slider_ref)
        slider_rate = (slider_target - slider_prev) / dt
        rate_lim = max(float(self.cfg.slider_rate_limit), 0.0)
        if rate_lim > 0.0:
            slider_rate = clamp(slider_rate, -rate_lim, rate_lim)
        slider_next = slider_prev + slider_rate * dt
        slider_next = clamp(slider_next, -slider_lim, slider_lim)
        return slider_next, slider_rate

    def build_reference_sequence(
        self,
        state: State6,
        thrust_cmd: float,
        raw_angle_now: float,
        raw_angle_future: Sequence[float],
    ) -> np.ndarray:
        horizon = max(int(self.cfg.prediction_horizon), 1)
        raw_future = [float(v) for v in raw_angle_future][:horizon]
        if len(raw_future) < horizon:
            raw_future.extend([float(raw_angle_now)] * (horizon - len(raw_future)))

        if self.cfg.angle_index == 1:
            measured_angle = float(state.theta)
            measured_rate = float(state.q)
        else:
            measured_angle = float(state.phi)
            measured_rate = float(state.p)

        seq = np.zeros((horizon, 10), dtype=float)
        prev_raw_target = (
            float(self.state.last_raw_angle)
            if self.state.last_raw_angle is not None
            else float(raw_angle_now)
        )
        if self.state.last_raw_angle is None:
            base_angle = measured_angle
            base_rate = measured_rate
            base_slider = float(state.chi if self.cfg.slider_index == 6 else state.ups)
            base_slider_rate = float(state.chi_d if self.cfg.slider_index == 6 else state.ups_d)
        else:
            sign_reversal_active = AxisManeuverCoordinator._sign_reversal_active(
                float(raw_angle_now),
                measured_angle,
                float(self.state.shaped_angle),
            )
            # 当内层实际姿态/滑块已经偏离上一次 rollout 状态时，
            # 需要用少量实测状态重新锚定预测域，避免 coordinator
            # 长时间沿着“自认为的形状”滚动而不回看真实机体状态。
            state_measurement_blend = (
                0.0
                if sign_reversal_active
                else clamp(float(self.cfg.state_measurement_blend), 0.0, 1.0)
            )
            base_angle = (
                (1.0 - state_measurement_blend) * float(self.state.shaped_angle)
                + state_measurement_blend * measured_angle
            )
            base_rate = (
                (1.0 - state_measurement_blend) * float(self.state.shaped_rate)
                + state_measurement_blend * measured_rate
            )
            measured_slider = float(state.chi if self.cfg.slider_index == 6 else state.ups)
            measured_slider_rate = float(state.chi_d if self.cfg.slider_index == 6 else state.ups_d)
            base_slider = (
                (1.0 - state_measurement_blend) * float(self.state.slider_ref)
                + state_measurement_blend * measured_slider
            )
            base_slider_rate = (
                (1.0 - state_measurement_blend) * float(self.state.slider_rate)
                + state_measurement_blend * measured_slider_rate
            )
        rollout_state = AxisCoordinatorState(
            shaped_angle=base_angle,
            shaped_rate=base_rate,
            slider_ref=base_slider,
            slider_rate=base_slider_rate,
            last_raw_angle=self.state.last_raw_angle,
        )
        next_step_state: Optional[AxisCoordinatorState] = None
        for k in range(horizon):
            raw_target = float(raw_angle_now) if k == 0 else raw_future[k - 1]
            angle_ref, rate_ref = self._update_axis_state(rollout_state, raw_target, prev_raw_target)
            rollout_state.shaped_angle = angle_ref
            rollout_state.shaped_rate = rate_ref
            slider_ref, slider_rate = self._update_slider_state(
                rollout_state,
                angle_ref,
                rate_ref,
                measured_angle,
                measured_rate,
                thrust_cmd,
            )
            rollout_state.slider_ref = slider_ref
            rollout_state.slider_rate = slider_rate

            row = seq[k]
            row[self.cfg.angle_index] = angle_ref
            row[self.cfg.rate_index] = rate_ref
            row[self.cfg.slider_index] = slider_ref
            row[self.cfg.slider_index + 1] = slider_rate
            prev_raw_target = raw_target
            if k == 0:
                next_step_state = AxisCoordinatorState(
                    shaped_angle=angle_ref,
                    shaped_rate=rate_ref,
                    slider_ref=slider_ref,
                    slider_rate=slider_rate,
                    last_raw_angle=float(raw_target),
                )

        if next_step_state is not None:
            self.state.shaped_angle = float(next_step_state.shaped_angle)
            self.state.shaped_rate = float(next_step_state.shaped_rate)
            self.state.slider_ref = float(next_step_state.slider_ref)
            self.state.slider_rate = float(next_step_state.slider_rate)
            self.state.last_raw_angle = float(next_step_state.last_raw_angle)
        else:
            self.state.last_raw_angle = float(raw_angle_now)
        return seq


def world_name_from_topic(topic: str) -> str:
    parts = [part for part in str(topic).split("/") if part]
    if len(parts) >= 2 and parts[0] == "world":
        return parts[1]
    return ""


def hover_ready_for_wind_activation(state: State6, wind_config: WindConfigSummary) -> bool:
    return (
        abs(state.z - wind_config.hover_target_z) <= wind_config.hover_z_tol
        and abs(state.vx) <= wind_config.hover_speed_tol
        and abs(state.vy) <= wind_config.hover_speed_tol
        and abs(state.vz) <= wind_config.hover_speed_tol
    )

# 位置变换矩阵：机体系⇄惯性系
def T_transformation(x: State6):
    cphi = math.cos(x.phi)
    sphi = math.sin(x.phi)
    ctheta = math.cos(x.theta)
    stheta = math.sin(x.theta)
    cpsi = math.cos(x.psi)
    spsi = math.sin(x.psi)

    # 机体系到惯性系
    Tb_v = np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
        [-stheta,       ctheta * sphi,                       ctheta * cphi]
    ])

    # 惯性系到机体系
    Tv_b = Tb_v.T

    return Tb_v, Tv_b

@dataclass
class Input6:
    thrust: float = 0.0
    tau_z: float = 0.0
    chi_cmd: float = 0.0
    ups_cmd: float = 0.0

# ===================== 转动惯量相关（论文式(24)(26)） =====================
@dataclass
class Sigma:
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s1: float
    ds2: float
    ds3: float
    ds4: float
    ds5: float
    ds6: float
    ds7: float
    ds8: float
    ds9: float
    ds10: float
    ds1: float

def cal_sigma(P: Params, chi, ups, chi_d, ups_d) -> Sigma:
    mu = P.mu
    m_b = P.m_b
    m = P.m
    l = P.l

    s2 = 4.0 * m_b * mu * mu * chi * chi
    s3 = 4.0 * m_b * mu * mu * ups * ups
    s4 = 16.0 * m * mu * mu * chi * chi
    s5 = 16.0 * m * mu * mu * ups * ups
    s6 = m * l * l / 2.0
    s7 = 8.0 * m * mu * ups * ups
    s8 = 8.0 * m * mu * chi * chi
    s9 = 2.0 * m * chi * chi
    s10 = 2.0 * m * ups * ups
    s1 = -4.0 * mu * chi * ups * (4.0 * m * mu - 2.0 * m + m_b * mu)

    ds2 = 8.0 * m_b * mu * mu * chi * chi_d
    ds3 = 8.0 * m_b * mu * mu * ups * ups_d
    ds4 = 32.0 * m * mu * mu * chi * chi_d
    ds5 = 32.0 * m * mu * mu * ups * ups_d
    ds6 = 0.0
    ds7 = 16.0 * m * mu * ups * ups_d
    ds8 = 16.0 * m * mu * chi * chi_d
    ds9 = 4.0 * m * chi * chi_d
    ds10 = 4.0 * m * ups * ups_d
    ds1 = (-4.0 * mu * chi_d * ups * (4.0 * m * mu - 2.0 * m + m_b * mu)
           - 4.0 * mu * chi * ups_d * (4.0 * m * mu - 2.0 * m + m_b * mu))

    return Sigma(s2, s3, s4, s5, s6, s7, s8, s9, s10, s1,
                 ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds1)

@dataclass
class Inertia:
    Ixx: float
    Iyy: float
    Izz: float
    Ixy: float
    dIxx: float
    dIyy: float
    dIzz: float
    dIxy: float

def cal_inertia(P: Params, S: Sigma) -> Inertia:
    m = P.m
    l = P.l
    Ib_x = P.Ib_x
    Ib_y = P.Ib_y
    Ib_z = P.Ib_z
    Im_x = P.Im_x
    Im_y = P.Im_y
    Im_z = P.Im_z

    add_xx = S.s10 + S.s6 + S.s5 + S.s3 - S.s8
    add_yy = S.s9 + S.s6 + S.s4 + S.s2 - S.s7
    add_zz = m * l * l + S.s10 + S.s9 + S.s5 + S.s4 + S.s3 + S.s2 - S.s8 - S.s7

    Ixx = Ib_x + 4.0 * Im_x + add_xx
    Iyy = Ib_y + 4.0 * Im_y + add_yy
    Izz = Ib_z + 4.0 * Im_z + add_zz
    Ixy = S.s1

    dIxx = S.ds10 + S.ds6 + S.ds5 + S.ds3 - S.ds8
    dIyy = S.ds9 + S.ds6 + S.ds4 + S.ds2 - S.ds7
    dIzz = S.ds10 + S.ds9 + S.ds5 + S.ds4 + S.ds3 + S.ds2 - S.ds8 - S.ds7
    dIxy = S.ds1

    return Inertia(Ixx, Iyy, Izz, Ixy, dIxx, dIyy, dIzz, dIxy)

# ===================== 姿态运动学方程 =====================
def attitude_kinematics(p, q, r, phi, theta):
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    ctheta = math.cos(theta)
    ttheta = math.tan(theta)

    phid = p + r * (cphi * ttheta) + q * (sphi * ttheta)
    thetad = q * cphi - r * sphi
    psid = r * (cphi / ctheta) + q * (sphi / ctheta)

    return phid, thetad, psid

def calculate_aerodynamic_forces(x: State6, P: Params):
    """
    计算气动阻力和转动阻尼力矩
    返回: (F_drag_inertial, M_damping_inertial, M_damping_body)
    - F_drag_inertial: 惯性系下的平动阻力 [N]
    - M_damping_inertial: 惯性系下的转动阻尼力矩 [N*m]
    - M_damping_body: 机体系下的转动阻尼力矩 [N*m]
    """
    Tb_v, Tv_b = T_transformation(x)
    
    v_ground = np.array([x.vx, x.vy, x.vz])
    v_body = Tv_b @ v_ground
    
    Fx = -0.5 * P.rho * P.S_side * P.Cd_side * v_body[0] * abs(v_body[0])
    Fy = -0.5 * P.rho * P.S_side * P.Cd_side * v_body[1] * abs(v_body[1])
    Fz = -0.5 * P.rho * P.S_top * P.Cd_top * v_body[2] * abs(v_body[2])
    
    F_drag_body = np.array([Fx, Fy, Fz])
    F_drag_inertial = Tb_v @ F_drag_body
    
    M_damping_body = -np.array([P.kp * x.p, P.kq * x.q, P.kr * x.r])
    M_damping_inertial = Tb_v @ M_damping_body

    return F_drag_inertial, M_damping_inertial, M_damping_body


def resolve_body_wrench_to_world(
    x: State6,
    thrust_real: float,
    tau_z_real: float,
    drag_force: np.ndarray,
    damping_torque: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将机体系控制推力/偏航力矩转换到世界系，并叠加阻力与阻尼。
    这里返回物理意义上的连续力/力矩，不再做脉冲缩放。
    """
    Tb_v, _ = T_transformation(x)
    z_world = Tb_v[:, 2]
    force_world = thrust_real * z_world + drag_force

    torque_control_body = np.array([0.0, 0.0, tau_z_real], dtype=float)
    torque_control_inertial = Tb_v @ torque_control_body
    torque_world = torque_control_inertial + damping_torque

    return force_world, torque_world


def resolve_aux_wrench_to_world(
    drag_force: np.ndarray,
    damping_torque: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    仅返回环境/阻尼项对应的世界系力和力矩。
    rotor_physics 模式下，控制推力 / 偏航力矩不再经 base_link wrench 注入。
    """
    return np.asarray(drag_force, dtype=float), np.asarray(damping_torque, dtype=float)


def update_wrench_scale(
    dt_actual: float,
    prev_dt_for_scale: float,
    dt_min: float = 0.008,
    dt_max: float = 0.100,
    alpha_old: float = 0.85,
    alpha_new: float = 0.15,
) -> Tuple[float, float]:
    """
    根据最新控制周期更新 ApplyLinkWrench 的脉冲面积补偿。
    """
    dt_limited = clamp(dt_actual, dt_min, dt_max)
    dt_for_scale = alpha_old * prev_dt_for_scale + alpha_new * dt_limited
    scale = dt_for_scale * 1000.0
    return dt_for_scale, scale

# ===================== 六自由度非线性动力学模型 =====================
def update_6dof_model(P: Params, x: State6, u: Input6) -> State6:
    F_drag_inertial, M_damping_inertial, M_damping_body = calculate_aerodynamic_forces(x, P)
    
    Tb_v, Tv_b = T_transformation(x)

    Md = M_damping_body

    # 滑块执行器二阶动态
    wn, zt = P.wn_mass, P.zeta_mass
    chi_dd = wn * wn * (u.chi_cmd - x.chi) - 2.0 * zt * wn * x.chi_d
    ups_dd = wn * wn * (u.ups_cmd - x.ups) - 2.0 * zt * wn * x.ups_d

    # 转动惯量
    S = cal_sigma(P, x.chi, x.ups, x.chi_d, x.ups_d)
    I = cal_inertia(P, S)

    M = P.M
    mu = P.mu
    m = P.m
    b = P.b_thrust
    g = P.g

    # 转动方程矩阵形式（论文式(57)）
    # 旋转动力学矩阵形式：A * w_dot = B_vec
    # 为了得到 w_dot（角加速度），B_vec 的每一项都必须是“力矩 / 主惯量”

    A = np.array([
        [1.0, I.Ixy / I.Ixx, 0.0],
        [I.Ixy / I.Iyy, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    sum_w1w2 = u.thrust / b

    B_p = (-I.dIxx * x.p / I.Ixx - I.dIxy * x.q / I.Ixx
           - x.q * I.Izz * x.r / I.Ixx + x.r * I.Iyy * x.q / I.Ixx
           - x.q * (4.0 * m * mu * x.ups * x.chi_d - 4.0 * m * mu * x.chi * x.ups_d) / I.Ixx
           - 2.0 * b * mu * x.ups * sum_w1w2 / I.Ixx
           + Md[0] / I.Ixx)

    B_q = (-I.dIxy * x.p / I.Iyy - I.dIyy * x.q / I.Iyy
           - x.r * (I.Ixx * x.p + I.Ixy * x.q) / I.Iyy + x.p * I.Izz * x.r / I.Iyy
           - x.p * (4.0 * m * mu * x.ups * x.chi_d - 4.0 * m * mu * x.chi * x.ups_d) / I.Iyy
           + 2.0 * b * mu * x.chi * sum_w1w2 / I.Iyy
           + Md[1] / I.Iyy)

    B_r = (-I.dIzz * x.r / I.Izz - x.p * (I.Ixy * x.p + I.Iyy * x.q) / I.Izz
           + x.q * (I.Ixx * x.p + I.Ixy * x.q) / I.Izz
           - 4.0 * m * mu * x.ups * chi_dd / I.Izz - 4.0 * m * mu * x.chi * ups_dd / I.Izz
           + u.tau_z / I.Izz
           + Md[2] / I.Izz)

    B_vec = np.array([B_p, B_q, B_r])

    try:
        pd, qd, rd = np.linalg.solve(A, B_vec)
    except np.linalg.LinAlgError:
        pd, qd, rd = 0.0, 0.0, 0.0

    # 平动方程（论文式(48)）
    cphi = math.cos(x.phi)
    sphi = math.sin(x.phi)
    ctheta = math.cos(x.theta)
    stheta = math.sin(x.theta)
    cpsi = math.cos(x.psi)
    spsi = math.sin(x.psi)

    x_dd = (u.thrust / M * (cphi * stheta * cpsi + sphi * spsi)
            - mu * (2.0 * chi_dd - 2.0 * x.r * x.ups_d)
            + 2.0 * rd * mu * x.ups + 2.0 * x.r * mu * x.ups_d
            - (2.0 * mu * x.ups * x.p * x.q
               - 2.0 * mu * x.chi * x.q * x.q
               - 2.0 * mu * x.chi * x.r * x.r)
            + F_drag_inertial[0] / M)

    y_dd = (u.thrust / M * (cphi * stheta * spsi - sphi * cpsi)
            - mu * (2.0 * ups_dd + 2.0 * x.r * x.chi_d)
            - (2.0 * rd * mu * x.chi + 2.0 * x.r * mu * x.chi_d)
            + 2.0 * mu * x.ups * x.q * x.q
            + 2.0 * mu * x.ups * x.p * x.p
            + 2.0 * mu * x.chi * x.p * x.q
            + F_drag_inertial[1] / M)

    z_dd = (u.thrust / M * cphi * ctheta - g
            - mu * (2.0 * x.p * x.ups_d - 2.0 * x.q * x.chi_d)
            - (2.0 * pd * mu * x.ups - 2.0 * qd * mu * x.chi
               + 2.0 * x.p * mu * x.ups_d - 2.0 * x.q * mu * x.chi_d)
            - (2.0 * mu * x.ups * x.q * x.r + 2.0 * mu * x.chi * x.p * x.r)
            + F_drag_inertial[2] / M)

    # 欧拉角变化率
    phid, thetad, psid = attitude_kinematics(x.p, x.q, x.r, x.phi, x.theta)

    dx = State6()
    dx.x, dx.y, dx.z = x.vx, x.vy, x.vz
    dx.vx, dx.vy, dx.vz = x_dd, y_dd, z_dd
    dx.phi, dx.theta, dx.psi = phid, thetad, psid
    dx.p, dx.q, dx.r = pd, qd, rd
    dx.chi, dx.ups = x.chi_d, x.ups_d
    dx.chi_d, dx.ups_d = chi_dd, ups_dd

    return dx

# ===================== 四阶龙格-库塔数值积分（六自由度） =====================
def rk4_state6(P: Params, x: State6, t: float, h: float, u: Input6):  # 把无人机推向未来的下一个时刻。
    k1 = update_6dof_model(P, x, u)

    x2 = State6(**{**x.__dict__})
    x2.x += 0.5 * h * k1.x
    x2.y += 0.5 * h * k1.y
    x2.z += 0.5 * h * k1.z
    x2.vx += 0.5 * h * k1.vx
    x2.vy += 0.5 * h * k1.vy
    x2.vz += 0.5 * h * k1.vz
    x2.phi += 0.5 * h * k1.phi
    x2.theta += 0.5 * h * k1.theta
    x2.psi += 0.5 * h * k1.psi
    x2.p += 0.5 * h * k1.p
    x2.q += 0.5 * h * k1.q
    x2.r += 0.5 * h * k1.r
    x2.chi += 0.5 * h * k1.chi
    x2.ups += 0.5 * h * k1.ups
    x2.chi_d += 0.5 * h * k1.chi_d
    x2.ups_d += 0.5 * h * k1.ups_d

    k2 = update_6dof_model(P, x2, u)

    x3 = State6(**{**x.__dict__})
    x3.x += 0.5 * h * k2.x
    x3.y += 0.5 * h * k2.y
    x3.z += 0.5 * h * k2.z
    x3.vx += 0.5 * h * k2.vx
    x3.vy += 0.5 * h * k2.vy
    x3.vz += 0.5 * h * k2.vz
    x3.phi += 0.5 * h * k2.phi
    x3.theta += 0.5 * h * k2.theta
    x3.psi += 0.5 * h * k2.psi
    x3.p += 0.5 * h * k2.p
    x3.q += 0.5 * h * k2.q
    x3.r += 0.5 * h * k2.r
    x3.chi += 0.5 * h * k2.chi
    x3.ups += 0.5 * h * k2.ups
    x3.chi_d += 0.5 * h * k2.chi_d
    x3.ups_d += 0.5 * h * k2.ups_d

    k3 = update_6dof_model(P, x3, u)

    x4 = State6(**{**x.__dict__})
    x4.x += h * k3.x
    x4.y += h * k3.y
    x4.z += h * k3.z
    x4.vx += h * k3.vx
    x4.vy += h * k3.vy
    x4.vz += h * k3.vz
    x4.phi += h * k3.phi
    x4.theta += h * k3.theta
    x4.psi += h * k3.psi
    x4.p += h * k3.p
    x4.q += h * k3.q
    x4.r += h * k3.r
    x4.chi += h * k3.chi
    x4.ups += h * k3.ups
    x4.chi_d += h * k3.chi_d
    x4.ups_d += h * k3.ups_d

    k4 = update_6dof_model(P, x4, u)

    # 四阶龙格-库塔更新：x += (h/6) * (k1 + 2k2 + 2k3 + k4)
    x.x += (h / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x)
    x.y += (h / 6.0) * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y)
    x.z += (h / 6.0) * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z)
    x.vx += (h / 6.0) * (k1.vx + 2.0 * k2.vx + 2.0 * k3.vx + k4.vx)
    x.vy += (h / 6.0) * (k1.vy + 2.0 * k2.vy + 2.0 * k3.vy + k4.vy)
    x.vz += (h / 6.0) * (k1.vz + 2.0 * k2.vz + 2.0 * k3.vz + k4.vz)
    x.phi += (h / 6.0) * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi)
    x.theta += (h / 6.0) * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta)
    x.psi += (h / 6.0) * (k1.psi + 2.0 * k2.psi + 2.0 * k3.psi + k4.psi)
    x.p += (h / 6.0) * (k1.p + 2.0 * k2.p + 2.0 * k3.p + k4.p)
    x.q += (h / 6.0) * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q)
    x.r += (h / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r)
    x.chi += (h / 6.0) * (k1.chi + 2.0 * k2.chi + 2.0 * k3.chi + k4.chi)
    x.ups += (h / 6.0) * (k1.ups + 2.0 * k2.ups + 2.0 * k3.ups + k4.ups)
    x.chi_d += (h / 6.0) * (k1.chi_d + 2.0 * k2.chi_d + 2.0 * k3.chi_d + k4.chi_d)
    x.ups_d += (h / 6.0) * (k1.ups_d + 2.0 * k2.ups_d + 2.0 * k3.ups_d + k4.ups_d)

    # ================= 物理限位与碰撞模拟 =================
    # 逻辑：如果滑块位置超出边界，且速度方向是“继续向外”，则强制将位置拉回边界，并将速度设为0。
    # 这模拟了刚性限位器的非弹性碰撞效果。

    # 1. 检查 chi (前后移动滑块)
    if x.chi >= P.u2_lim:
        x.chi = P.u2_lim
        if x.chi_d > 0:  # 如果还在向外冲
            x.chi_d = 0.0
    elif x.chi <= -P.u2_lim:
        x.chi = -P.u2_lim
        if x.chi_d < 0:  # 如果还在向外冲
            x.chi_d = 0.0

    # 2. 检查 ups (左右移动滑块)
    if x.ups >= P.u2_lim:
        x.ups = P.u2_lim
        if x.ups_d > 0:
            x.ups_d = 0.0
    elif x.ups <= -P.u2_lim:
        x.ups = -P.u2_lim
        if x.ups_d < 0:
            x.ups_d = 0.0

# ===================== 轨迹 & 双环控制结构 =====================
# ---- 三维位置参考（外环使用） ----
@dataclass
class RefPos:
    """
    通用参考轨迹状态，由规划器生成，传给外环控制器
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    psi: float = 0.0  # 期望偏航角

@dataclass
class RefAtt:
    phi: float
    theta: float
    psi: float


@dataclass
class AdaptivePhaseScheduleConfig:
    enabled: bool = False
    min_rate: float = 0.40
    filter_time_constant: float = 0.30
    along_track_window: float = 1.00
    cross_track_window: float = 0.75
    speed_floor: float = 0.25
    position_floor: float = 0.25
    velocity_floor: float = 0.25
    lag_weight: float = 1.0
    cross_track_weight: float = 1.15
    velocity_weight: float = 0.30
    projection_align_time_constant: float = 0.30
    projection_deadband: float = 0.02
    projection_max_correction: float = 0.10


@dataclass
class AdaptivePhaseScheduleState:
    last_update_time: Optional[float] = None
    effective_time: float = 0.0
    phase_rate: float = 1.0
    phase_accel: float = 0.0
    error_metric: float = 0.0


@dataclass
class ManualXYCommand:
    forward: float = 0.0
    lateral: float = 0.0


@dataclass
class ManualXYStatus:
    ref_x: Optional[float] = None
    ref_y: Optional[float] = None
    ref_vx: float = 0.0
    ref_vy: float = 0.0
    ref_ax: float = 0.0
    ref_ay: float = 0.0
    phi_ref_override: Optional[float] = None
    theta_ref_override: Optional[float] = None
    psi_ref_override: Optional[float] = None
    manual_input_forward: float = 0.0
    manual_input_lateral: float = 0.0
    hold_x: Optional[float] = None
    hold_y: Optional[float] = None
    mode_label: str = "LOCKED"
    brake_phi_ref: float = 0.0
    brake_theta_ref: float = 0.0


class ManualXYMode(Enum):
    LOCKED = "LOCKED"
    HOLD = "HOLD"
    COMMAND = "COMMAND"
    BRAKE = "BRAKE"


class ManualXYController:
    def __init__(
        self,
        unlock_altitude_tol: float,
        unlock_hold_time: float,
        max_tilt_deg: float,
        max_speed: float,
        speed_rise_time: float,
        speed_fall_time: float,
        brake_max_tilt_deg: float,
        brake_full_speed: float,
        brake_rise_time: float,
        brake_stop_speed_threshold: float,
        brake_capture_speed: float,
        brake_relock_dwell: float,
        stop_tilt_deg: float,
        cmd_timeout: float,
    ):
        self.unlock_altitude_tol = max(0.0, unlock_altitude_tol)
        self.unlock_hold_time = max(0.0, unlock_hold_time)
        self.max_tilt = math.radians(max_tilt_deg)
        self.max_speed = max(1e-3, max_speed)
        self.speed_rise_tau = max(1e-3, speed_rise_time / 3.0)
        self.speed_fall_tau = max(1e-3, speed_fall_time / 3.0)
        self.brake_max_tilt = math.radians(brake_max_tilt_deg)
        self.brake_tau = max(1e-3, brake_rise_time / 3.0)
        self.brake_full_speed = max(1e-3, brake_full_speed)
        self.brake_stop_speed_threshold = max(0.0, brake_stop_speed_threshold)
        self.brake_capture_speed = max(0.0, brake_capture_speed)
        self.brake_relock_dwell = max(0.0, brake_relock_dwell)
        self.stop_tilt = math.radians(stop_tilt_deg)
        self.cmd_timeout = max(0.0, cmd_timeout)

        self.mode = ManualXYMode.LOCKED
        self.ready = False
        self.ready_hold_start_sec: Optional[float] = None
        self.yaw_hold_ref: Optional[float] = None

        self.command = ManualXYCommand()
        self.last_command_time_sec: Optional[float] = None
        self.filtered_theta_ref = 0.0
        self.filtered_phi_ref = 0.0
        self.filtered_vx_cmd = 0.0
        self.filtered_vy_cmd = 0.0
        self.command_deadband = 1e-3
        self.hold_x: Optional[float] = None
        self.hold_y: Optional[float] = None
        self.brake_start_sec: Optional[float] = None
        self.brake_stop_hold_start_sec: Optional[float] = None
        self.brake_capture_vx = False
        self.brake_capture_vy = False
        self.brake_dir_vx = 0.0
        self.brake_dir_vy = 0.0
        self.brake_hold_x_candidate: Optional[float] = None
        self.brake_hold_y_candidate: Optional[float] = None
        self.brake_hold_x_window_start_sec: Optional[float] = None
        self.brake_hold_y_window_start_sec: Optional[float] = None
        self.brake_kd = 2.5
        self.prev_vx_body = 0.0
        self.prev_vy_body = 0.0

    def set_command(self, forward: float, lateral: float, now_sec: float):
        self.command.forward = clamp(float(forward), -1.0, 1.0)
        self.command.lateral = clamp(float(lateral), -1.0, 1.0)
        self.last_command_time_sec = now_sec

    def _current_command(self, now_sec: float) -> ManualXYCommand:
        if self.last_command_time_sec is None:
            return ManualXYCommand()
        if (now_sec - self.last_command_time_sec) > self.cmd_timeout:
            return ManualXYCommand()
        return self.command

    def _is_nonzero_command(self, command: ManualXYCommand) -> bool:
        return (
            abs(command.forward) > self.command_deadband
            or abs(command.lateral) > self.command_deadband
        )

    def _update_ready(self, state: State6, desired_z: float, now_sec: float):
        if self.ready:
            return

        # 第二阶段的键盘解锁逻辑更贴合项目规则：
        # 只要飞行器进入悬停高度带并稳定停留一小段时间，
        # 即使粗悬停还存在可见的低频漂移，
        # 手动 XY 控制也可以接管。
        hover_like = (
            abs(state.z - desired_z) <= self.unlock_altitude_tol
            and state.z >= max(0.5, desired_z - self.unlock_altitude_tol)
        )

        if not hover_like:
            self.ready_hold_start_sec = None
            return

        if self.ready_hold_start_sec is None:
            self.ready_hold_start_sec = now_sec
            return

        if (now_sec - self.ready_hold_start_sec) >= self.unlock_hold_time:
            self.ready = True
            self.yaw_hold_ref = state.psi
            self.hold_x = state.x
            self.hold_y = state.y

    @staticmethod
    def _lowpass_alpha(dt: float, tau: float) -> float:
        if dt <= 0.0:
            return 1.0
        return 1.0 - math.exp(-dt / max(tau, 1e-6))

    def _normalize_command(self, command: ManualXYCommand) -> Tuple[float, float]:
        mag = math.hypot(command.forward, command.lateral)
        if mag <= 1.0 or mag <= 1e-9:
            return command.forward, command.lateral
        return command.forward / mag, command.lateral / mag

    @staticmethod
    def _body_velocity(state: State6) -> Tuple[float, float]:
        # 手动水平控制只关心“机头朝向对齐后的水平速度”，
        # 不应把大滚转/俯仰和垂向速度的投影一起卷进来。
        # 否则一旦 BRAKE 期间发生明显下沉，Vz 会通过完整机体系变换
        # 污染成假的 Vx_body / Vy_body，再把俯仰/滚转轴一起带偏。
        cpsi = math.cos(state.psi)
        spsi = math.sin(state.psi)
        vx_heading = cpsi * state.vx + spsi * state.vy
        vy_heading = -spsi * state.vx + cpsi * state.vy
        return float(vx_heading), float(vy_heading)

    @staticmethod
    def _axis_sign(value: float, deadband: float = 1e-6) -> float:
        if value > deadband:
            return 1.0
        if value < -deadband:
            return -1.0
        return 0.0

    def _effective_brake_capture_speed(self) -> float:
        return min(
            self.brake_capture_speed,
            max(self.brake_stop_speed_threshold, 1e-6),
        )

    def _brake_hold_window_sec(self) -> float:
        return 2.25

    def _brake_hold_trigger_tilt(self) -> float:
        return min(
            self.brake_max_tilt,
            max(1.5 * self.stop_tilt, 0.2 * self.brake_max_tilt),
        )

    def _shape_speed_axis(self, current: float, target: float, dt: float, command_active: bool) -> float:
        # 速度指令整形：
        # 1. 有输入时，建立过程用较慢的 rise tau，让起步更柔和；
        # 2. 收油、反向或进入 BRAKE 时，用更快的 fall tau/刹车 tau，
        #    让旧方向的残余目标速度更快卸掉，保留“跟手”感。
        if not command_active:
            tau = self.brake_tau
        elif current * target < 0.0:
            tau = self.speed_fall_tau
        elif abs(target) > abs(current):
            tau = self.speed_rise_tau
        else:
            tau = self.speed_fall_tau

        alpha = self._lowpass_alpha(dt, tau)
        value = current + alpha * (target - current)
        value = clamp(value, -self.max_speed, self.max_speed)
        if abs(value) < 1e-5:
            return 0.0
        return value

    @staticmethod
    def _speed_error_to_tilt(
        target_vx: float,
        target_vy: float,
        vx_body: float,
        vy_body: float,
        tilt_limit: float,
        full_speed: float,
    ) -> Tuple[float, float]:
        gain = tilt_limit / max(full_speed, 1e-6)
        theta_ref = clamp((target_vx - vx_body) * gain, -tilt_limit, tilt_limit)
        # 横向轴沿用当前项目已验证的飞行语义：
        # “左”键对应负滚转，但对应正的水平位移/速度方向。
        # 因此 lateral 速度误差到滚转参考需要保留一个符号翻转。
        phi_ref = clamp(-(target_vy - vy_body) * gain, -tilt_limit, tilt_limit)
        return theta_ref, phi_ref

    def _update_brake_capture(
        self,
        speed: float,
        captured: bool,
        direction: float,
        tilt: float,
        brakes_when_same_sign: bool,
    ) -> bool:
        tilt_abs = abs(tilt)
        tilt_pushes_drift = False
        if abs(speed) > 1e-6 and tilt_abs > 1e-6:
            if brakes_when_same_sign:
                tilt_pushes_drift = speed * tilt < 0.0
            else:
                tilt_pushes_drift = speed * tilt > 0.0

        if captured:
            if tilt_pushes_drift:
                return False
            capture_speed = self._effective_brake_capture_speed()
            if speed * direction < 0.0 and abs(speed) > capture_speed:
                return False
            return True
        if direction == 0.0:
            return True
        capture_speed = self._effective_brake_capture_speed()
        return (
            abs(speed) <= capture_speed
            and tilt_abs <= self.stop_tilt
            and not tilt_pushes_drift
        )

    def _soften_small_rebound_brake(self, reference: float, speed: float, direction: float) -> float:
        if direction == 0.0 or abs(reference) <= 1e-9:
            return reference
        if speed * direction >= 0.0:
            return reference
        capture_speed = self._effective_brake_capture_speed()
        if capture_speed <= 1e-6:
            return 0.0

        ramp = clamp(abs(speed) / capture_speed, 0.0, 1.0)
        return reference * ramp * ramp

    def _begin_brake_phase(self, now_sec: float, vx_body: float, vy_body: float):
        self.brake_start_sec = now_sec
        self.brake_stop_hold_start_sec = None
        capture_speed = self._effective_brake_capture_speed()
        self.brake_dir_vx = self._axis_sign(vx_body, capture_speed)
        self.brake_dir_vy = self._axis_sign(vy_body, capture_speed)
        self.brake_capture_vx = self.brake_dir_vx == 0.0
        self.brake_capture_vy = self.brake_dir_vy == 0.0
        self.brake_hold_x_candidate = None
        self.brake_hold_y_candidate = None
        self.brake_hold_x_window_start_sec = None
        self.brake_hold_y_window_start_sec = None

    def _clear_brake_hold_candidates(self):
        self.brake_hold_x_candidate = None
        self.brake_hold_y_candidate = None
        self.brake_hold_x_window_start_sec = None
        self.brake_hold_y_window_start_sec = None

    def _update_brake_hold_candidate(
        self,
        now_sec: float,
        position: float,
        speed: float,
        direction: float,
        brake_reference: float,
        candidate: Optional[float],
        window_start_sec: Optional[float],
        brakes_when_same_sign: bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        if candidate is not None or direction == 0.0:
            return candidate, window_start_sec

        moving_along_brake_entry = speed * direction > 0.0
        if not moving_along_brake_entry:
            return candidate, None

        if window_start_sec is None:
            trigger_tilt = self._brake_hold_trigger_tilt()
            if brakes_when_same_sign:
                in_primary_brake_lobe = brake_reference * direction > 0.0
            else:
                in_primary_brake_lobe = brake_reference * direction < 0.0
            if (
                in_primary_brake_lobe
                and abs(brake_reference) >= trigger_tilt
                and abs(speed) > self.brake_stop_speed_threshold
            ):
                return candidate, now_sec
            return candidate, None

        if (now_sec - window_start_sec) >= self._brake_hold_window_sec():
            return position, window_start_sec
        return candidate, window_start_sec

    def _finalize_status(
        self,
        status: ManualXYStatus,
        vx_body: float,
        vy_body: float,
    ) -> ManualXYStatus:
        self.prev_vx_body = vx_body
        self.prev_vy_body = vy_body
        return status

    def advance(self, state: State6, base_ref: RefPos, now_sec: float, dt: float) -> ManualXYStatus:
        desired_z = base_ref.z
        self._update_ready(state, desired_z, now_sec)
        vx_body, vy_body = self._body_velocity(state)

        if not self.ready:
            self.mode = ManualXYMode.LOCKED
            self.brake_start_sec = None
            self.brake_stop_hold_start_sec = None
            self.filtered_theta_ref = 0.0
            self.filtered_phi_ref = 0.0
            self.filtered_vx_cmd = 0.0
            self.filtered_vy_cmd = 0.0
            self.brake_capture_vx = False
            self.brake_capture_vy = False
            self.brake_dir_vx = 0.0
            self.brake_dir_vy = 0.0
            self._clear_brake_hold_candidates()
            self.hold_x = base_ref.x
            self.hold_y = base_ref.y
            return self._finalize_status(
                ManualXYStatus(
                    ref_x=base_ref.x,
                    ref_y=base_ref.y,
                    ref_vx=base_ref.vx,
                    ref_vy=base_ref.vy,
                    ref_ax=base_ref.ax,
                    ref_ay=base_ref.ay,
                    psi_ref_override=None,
                    manual_input_forward=0.0,
                    manual_input_lateral=0.0,
                    hold_x=self.hold_x,
                    hold_y=self.hold_y,
                    mode_label=self.mode.value,
                    brake_phi_ref=0.0,
                    brake_theta_ref=0.0,
                ),
                vx_body,
                vy_body,
            )

        if self.hold_x is None:
            self.hold_x = state.x
        if self.hold_y is None:
            self.hold_y = state.y
        if self.yaw_hold_ref is None:
            self.yaw_hold_ref = state.psi

        cmd = self._current_command(now_sec)
        has_input = self._is_nonzero_command(cmd)
        norm_forward, norm_lateral = self._normalize_command(cmd)
        prev_mode = self.mode

        if self.mode == ManualXYMode.LOCKED:
            self.mode = ManualXYMode.HOLD

        if self.mode == ManualXYMode.HOLD and has_input:
            self.mode = ManualXYMode.COMMAND
        elif self.mode == ManualXYMode.COMMAND and not has_input:
            self.mode = ManualXYMode.BRAKE
        elif self.mode == ManualXYMode.BRAKE and has_input:
            self.mode = ManualXYMode.COMMAND

        if prev_mode != self.mode:
            if self.mode == ManualXYMode.BRAKE:
                self._begin_brake_phase(now_sec, vx_body, vy_body)
            elif prev_mode == ManualXYMode.BRAKE:
                self.brake_start_sec = None
                self.brake_stop_hold_start_sec = None
                self.brake_capture_vx = False
                self.brake_capture_vy = False
                self.brake_dir_vx = 0.0
                self.brake_dir_vy = 0.0
                self._clear_brake_hold_candidates()

        if self.mode == ManualXYMode.HOLD:
            self.brake_start_sec = None
            self.brake_stop_hold_start_sec = None
            self.filtered_theta_ref = 0.0
            self.filtered_phi_ref = 0.0
            self.filtered_vx_cmd = 0.0
            self.filtered_vy_cmd = 0.0
            self.brake_capture_vx = False
            self.brake_capture_vy = False
            self.brake_dir_vx = 0.0
            self.brake_dir_vy = 0.0
            self._clear_brake_hold_candidates()
            return self._finalize_status(
                ManualXYStatus(
                    ref_x=self.hold_x,
                    ref_y=self.hold_y,
                    ref_vx=0.0,
                    ref_vy=0.0,
                    ref_ax=0.0,
                    ref_ay=0.0,
                    phi_ref_override=None,
                    theta_ref_override=None,
                    psi_ref_override=self.yaw_hold_ref,
                    manual_input_forward=0.0,
                    manual_input_lateral=0.0,
                    hold_x=self.hold_x,
                    hold_y=self.hold_y,
                    mode_label=self.mode.value,
                    brake_phi_ref=0.0,
                    brake_theta_ref=0.0,
                ),
                vx_body,
                vy_body,
            )

        brake_theta_ref = 0.0
        brake_phi_ref = 0.0
        if self.mode == ManualXYMode.COMMAND:
            # COMMAND 阶段改为“目标机体系速度”：
            # 键盘只决定希望的移动方向和速度上限，
            # 不再直接指定固定倾角。
            target_vx_cmd = norm_forward * self.max_speed
            # 左右键的飞行语义沿用旧基线：
            # 左键应建立“左飞”的速度目标，同时最终仍映射成负滚转参考。
            # 这里的内部横向速度目标符号因此与键盘 lateral 输入保持反号。
            target_vy_cmd = -norm_lateral * self.max_speed
            self.filtered_vx_cmd = self._shape_speed_axis(
                self.filtered_vx_cmd,
                target_vx_cmd,
                dt,
                command_active=True,
            )
            self.filtered_vy_cmd = self._shape_speed_axis(
                self.filtered_vy_cmd,
                target_vy_cmd,
                dt,
                command_active=True,
            )
            target_theta, target_phi = self._speed_error_to_tilt(
                target_vx=self.filtered_vx_cmd,
                target_vy=self.filtered_vy_cmd,
                vx_body=vx_body,
                vy_body=vy_body,
                tilt_limit=self.max_tilt,
                full_speed=self.max_speed,
            )
        else:
            # BRAKE 阶段统一成“目标速度回零 + 更大的制动倾角上限”。
            if self.brake_start_sec is None:
                self._begin_brake_phase(now_sec, vx_body, vy_body)
            self.filtered_vx_cmd = self._shape_speed_axis(
                self.filtered_vx_cmd,
                0.0,
                dt,
                command_active=False,
            )
            self.filtered_vy_cmd = self._shape_speed_axis(
                self.filtered_vy_cmd,
                0.0,
                dt,
                command_active=False,
            )
            self.brake_capture_vx = self._update_brake_capture(
                vx_body,
                self.brake_capture_vx,
                self.brake_dir_vx,
                state.theta,
                brakes_when_same_sign=False,
            )
            self.brake_capture_vy = self._update_brake_capture(
                vy_body,
                self.brake_capture_vy,
                self.brake_dir_vy,
                state.phi,
                brakes_when_same_sign=True,
            )
            if self.brake_capture_vx:
                self.filtered_vx_cmd = 0.0
            if self.brake_capture_vy:
                self.filtered_vy_cmd = 0.0
            brake_theta_ref, brake_phi_ref = self._speed_error_to_tilt(
                target_vx=self.filtered_vx_cmd,
                target_vy=self.filtered_vy_cmd,
                vx_body=vx_body,
                vy_body=vy_body,
                tilt_limit=self.brake_max_tilt,
                full_speed=self.brake_full_speed,
            )
            if dt > 1e-6:
                dvx = (vx_body - self.prev_vx_body) / dt
                dvy = (vy_body - self.prev_vy_body) / dt
                gain = self.brake_max_tilt / max(self.brake_full_speed, 1e-6)
                brake_theta_ref -= self.brake_kd * dvx * gain * dt
                brake_phi_ref += self.brake_kd * dvy * gain * dt
                brake_theta_ref = clamp(
                    brake_theta_ref,
                    -self.brake_max_tilt,
                    self.brake_max_tilt,
                )
                brake_phi_ref = clamp(
                    brake_phi_ref,
                    -self.brake_max_tilt,
                    self.brake_max_tilt,
                )
            brake_theta_ref = self._soften_small_rebound_brake(
                brake_theta_ref,
                vx_body,
                self.brake_dir_vx,
            )
            brake_phi_ref = self._soften_small_rebound_brake(
                brake_phi_ref,
                vy_body,
                self.brake_dir_vy,
            )
            if self.brake_capture_vx:
                brake_theta_ref = 0.0
            if self.brake_capture_vy:
                brake_phi_ref = 0.0
            (
                self.brake_hold_x_candidate,
                self.brake_hold_x_window_start_sec,
            ) = self._update_brake_hold_candidate(
                now_sec=now_sec,
                position=state.x,
                speed=vx_body,
                direction=self.brake_dir_vx,
                brake_reference=brake_theta_ref,
                candidate=self.brake_hold_x_candidate,
                window_start_sec=self.brake_hold_x_window_start_sec,
                brakes_when_same_sign=False,
            )
            (
                self.brake_hold_y_candidate,
                self.brake_hold_y_window_start_sec,
            ) = self._update_brake_hold_candidate(
                now_sec=now_sec,
                position=state.y,
                speed=vy_body,
                direction=self.brake_dir_vy,
                brake_reference=brake_phi_ref,
                candidate=self.brake_hold_y_candidate,
                window_start_sec=self.brake_hold_y_window_start_sec,
                brakes_when_same_sign=True,
            )
            target_theta = brake_theta_ref
            target_phi = brake_phi_ref

        self.filtered_theta_ref = target_theta
        self.filtered_phi_ref = target_phi

        if abs(self.filtered_theta_ref) < 1e-5:
            self.filtered_theta_ref = 0.0
        if abs(self.filtered_phi_ref) < 1e-5:
            self.filtered_phi_ref = 0.0
        if abs(self.filtered_vx_cmd) < 1e-5:
            self.filtered_vx_cmd = 0.0
        if abs(self.filtered_vy_cmd) < 1e-5:
            self.filtered_vy_cmd = 0.0

        if self.mode == ManualXYMode.BRAKE:
            horiz_speed = math.hypot(state.vx, state.vy)
            target_speed_mag = math.hypot(self.filtered_vx_cmd, self.filtered_vy_cmd)
            if (
                horiz_speed <= self.brake_stop_speed_threshold
                and target_speed_mag <= self.brake_stop_speed_threshold
                and abs(state.phi) <= self.stop_tilt
                and abs(state.theta) <= self.stop_tilt
            ):
                if self.brake_stop_hold_start_sec is None:
                    self.brake_stop_hold_start_sec = now_sec
                elif (now_sec - self.brake_stop_hold_start_sec) >= self.brake_relock_dwell:
                    self.hold_x = (
                        self.brake_hold_x_candidate
                        if self.brake_hold_x_candidate is not None
                        else state.x
                    )
                    self.hold_y = (
                        self.brake_hold_y_candidate
                        if self.brake_hold_y_candidate is not None
                        else state.y
                    )
                    self.filtered_theta_ref = 0.0
                    self.filtered_phi_ref = 0.0
                    self.filtered_vx_cmd = 0.0
                    self.filtered_vy_cmd = 0.0
                    self.mode = ManualXYMode.HOLD
                    self.brake_start_sec = None
                    self.brake_stop_hold_start_sec = None
                    self._clear_brake_hold_candidates()
                    return self._finalize_status(
                        ManualXYStatus(
                            ref_x=self.hold_x,
                            ref_y=self.hold_y,
                            ref_vx=0.0,
                            ref_vy=0.0,
                            ref_ax=0.0,
                            ref_ay=0.0,
                            phi_ref_override=None,
                            theta_ref_override=None,
                            psi_ref_override=self.yaw_hold_ref,
                            manual_input_forward=0.0,
                            manual_input_lateral=0.0,
                            hold_x=self.hold_x,
                            hold_y=self.hold_y,
                            mode_label=self.mode.value,
                            brake_phi_ref=0.0,
                            brake_theta_ref=0.0,
                        ),
                        vx_body,
                        vy_body,
                    )
            else:
                self.brake_stop_hold_start_sec = None

        return self._finalize_status(
            ManualXYStatus(
                ref_x=state.x,
                ref_y=state.y,
                ref_vx=state.vx,
                ref_vy=state.vy,
                ref_ax=0.0,
                ref_ay=0.0,
                phi_ref_override=self.filtered_phi_ref,
                theta_ref_override=self.filtered_theta_ref,
                psi_ref_override=self.yaw_hold_ref,
                manual_input_forward=(
                    self.filtered_vx_cmd / self.max_speed if self.mode == ManualXYMode.COMMAND else 0.0
                ),
                manual_input_lateral=(
                    -self.filtered_vy_cmd / self.max_speed if self.mode == ManualXYMode.COMMAND else 0.0
                ),
                hold_x=self.hold_x,
                hold_y=self.hold_y,
                mode_label=self.mode.value,
                brake_phi_ref=brake_phi_ref,
                brake_theta_ref=brake_theta_ref,
            ),
            vx_body,
            vy_body,
        )

# ===================== 通用轨迹规划器 =====================
class QuinticPolynomial:
    """
    五次多项式求解器：用于在两点之间生成平滑过渡轨迹（求出p（t）的6个系数，p（t）是五次多项式，它算出的p\v\a轨迹数据对无人机来说是很平滑的）
    约束条件：起点和终点的 位置(p)、速度(v)、加速度(a) 均连续
    """

    def __init__(self, p0, v0, a0, pe, ve, ae, T):
        self.a0 = p0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array([
            [T ** 3, T ** 4, T ** 5],
            [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [6 * T, 12 * T ** 2, 20 * T ** 3]
        ])
        b = np.array([
            pe - self.a0 - self.a1 * T - self.a2 * T ** 2,
            ve - self.a1 - 2 * self.a2 * T,
            ae - 2 * self.a2
        ])

        x = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = x[0], x[1], x[2]

    def calc(self, t):
        """返回平滑后的数据：[位置、速度、加速度]"""
        if t < 0: return self.a0, self.a1, 2 * self.a2

        t2, t3, t4, t5 = t ** 2, t ** 3, t ** 4, t ** 5
        pos = self.a0 + self.a1 * t + self.a2 * t2 + self.a3 * t3 + self.a4 * t4 + self.a5 * t5
        vel = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t2 + 4 * self.a4 * t3 + 5 * self.a5 * t4
        acc = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t2 + 20 * self.a5 * t3
        return pos, vel, acc

class UniversalTrajectoryPlanner:
    """
    通用轨迹规划器：负责计算“当前时刻应该在哪里”
    """

    def __init__(self):
        self.funcs = None  # 三个坐标轴的目标函数（fx、fy、fz）
        self.polys = None  # 三个坐标轴的五次过渡多项式
        self.t_start = 0.0
        self.t_end_trans = 0.0
        self.is_active = False
        self.dt_diff = 1e-4  # 数值微分步长

    def set_mission(self, func_x, func_y, func_z, current_t, current_state: State6, trans_time=5.0):
        """
        设定新任务：从 current_state 平滑过渡到 func_xyz 定义的轨迹
        """
        self.funcs = (func_x, func_y, func_z)
        self.t_start = current_t
        self.t_end_trans = current_t + trans_time

        # 1. 起点状态 (当前无人机状态)
        # 注意：这里取惯性系下的状态
        s0_x = [current_state.x, current_state.vx, 0.0]  # 暂时忽略当前加速度测量，设为0以防噪声
        s0_y = [current_state.y, current_state.vy, 0.0]
        s0_z = [current_state.z, current_state.vz, 0.0]

        # 2. 终点状态（目标函数在 t_end 时的理想状态）
        target_t = self.t_end_trans
        se_x = self._eval_func(func_x, target_t)
        se_y = self._eval_func(func_y, target_t)
        se_z = self._eval_func(func_z, target_t)

        # 3. 生成过渡多项式
        self.polys = (
            QuinticPolynomial(*s0_x, *se_x, trans_time),
            QuinticPolynomial(*s0_y, *se_y, trans_time),
            QuinticPolynomial(*s0_z, *se_z, trans_time)
        )
        self.is_active = True
        print(f"[Planner] 轨迹规划已更新，将在 {trans_time}s 内平滑切入目标轨道。")

    def _eval_func(self, func, t):
        """用数值微分计算任意函数在当前时刻的位置、速度和加速度"""
        # 中心差分法：利用 t 前后极小的时间点 h 来估算导数
        h = self.dt_diff
        p = func(t)
        p_plus = func(t + h)
        p_minus = func(t - h)
        v = (p_plus - p_minus) / (2 * h)  # 速度中心差分：v = (f(t+h) - f(t-h)) / 2h
        a = (p_plus - 2 * p + p_minus) / (h ** 2)  # 加速度中心差分：a = (f(t+h) - 2f(t) + f(t-h)) / h^2
        return p, v, a

    def get_ref(self, t) -> RefPos:
        """获取 t 时刻的参考状态"""
        if not self.is_active:
            # 未激活时默认原点悬停
            return RefPos()

        x, vx, ax = 0, 0, 0
        y, vy, ay = 0, 0, 0
        z, vz, az = 0, 0, 0

        if t < self.t_end_trans:
            # ==== 过渡阶段（五次多项式） ====
            tau = t - self.t_start
            x, vx, ax = self.polys[0].calc(tau)
            y, vy, ay = self.polys[1].calc(tau)
            z, vz, az = self.polys[2].calc(tau)
        else:
            # ==== 锁定阶段（用户定义函数） ====
            x, vx, ax = self._eval_func(self.funcs[0], t)
            y, vy, ay = self._eval_func(self.funcs[1], t)
            z, vz, az = self._eval_func(self.funcs[2], t)

        # 默认偏航角为 0，如有需要可在此处添加逻辑（如切向偏航）
        return RefPos(x, y, z, vx, vy, vz, ax, ay, az, psi=0.0)

# ===================== 控制分配混控器 =====================
class CoaxialMixer:
    """
    共轴双旋翼混控器：
    当前悬停调试阶段采用：Thrust > Yaw
    当旋翼组合无法同时满足总推力和偏航力矩时，优先保证总推力，
    再在剩余可行域内尽量逼近期望偏航力矩。
    """

    def __init__(self, P: Params):
        self.b = P.b_thrust  # 推力系数
        self.d = P.d_yaw  # 反扭系数

        # 下旋翼效率系数 (经验值：0.8 - 0.9)
        # 意味着下旋翼w1要产生同样的力，转速平方需要除以这个系数（即转速要更高）
        self.lower_rotor_eff = 0.85

        # 定义电机物理限制 (RPM 或者 rad/s 的平方)
        # P.thrust_max 是总推力，单电机最大约为其一半
        self.w_sq_max = (P.thrust_max / 2.0) / self.b * 1.2
        self.w_sq_min = 0.0  # 电机最小转速平方
        self.min_speed_ratio = 0.0

    def set_min_speed_ratio(self, ratio: float):
        self.min_speed_ratio = clamp(float(ratio), 0.0, 0.95)

    def _static_min_w_sq(self) -> float:
        return self.w_sq_max * self.min_speed_ratio * self.min_speed_ratio

    def _effective_split_bounds(self, thrust_cmd: float):
        x_max = self.lower_rotor_eff * self.w_sq_max
        y_max = self.w_sq_max

        thrust_sum_des = max(0.0, float(thrust_cmd) / self.b)
        thrust_sum = min(thrust_sum_des, x_max + y_max)

        w_sq_floor_static = self._static_min_w_sq()
        x_min_static = self.lower_rotor_eff * w_sq_floor_static
        y_min_static = w_sq_floor_static
        min_sum_static = x_min_static + y_min_static
        floor_scale = min(1.0, thrust_sum / max(min_sum_static, 1e-9))
        x_min = x_min_static * floor_scale
        y_min = y_min_static * floor_scale

        x_low = max(x_min, thrust_sum - y_max)
        x_high = min(x_max, thrust_sum - y_min)
        if x_low > x_high:
            x_mid = 0.5 * (x_low + x_high)
            x_low = x_mid
            x_high = x_mid
        return thrust_sum, x_low, x_high

    def feasible_tau_z_bounds(self, thrust_cmd: float) -> Tuple[float, float]:
        thrust_sum, x_low, x_high = self._effective_split_bounds(thrust_cmd)
        yaw_diff_min = 2.0 * x_low - thrust_sum
        yaw_diff_max = 2.0 * x_high - thrust_sum
        return float(self.d * yaw_diff_min), float(self.d * yaw_diff_max)

    def clamp_tau_z(self, thrust_cmd: float, tau_z_cmd: float) -> float:
        tau_min, tau_max = self.feasible_tau_z_bounds(thrust_cmd)
        return float(clamp(float(tau_z_cmd), tau_min, tau_max))

    def mix(self, thrust_cmd, tau_z_cmd):
        """
        输入: 期望推力 [N], 期望偏航力矩 [N*m]
        输出: 实际推力 [N], 实际偏航力矩 [N*m], 下旋翼电机1转速平方, 上旋翼电机2转速平方
        偏航力矩符号（正值代表向左/逆时针转）。
        """
        # 记 x = eff * w1_sq，y = w2_sq，表示等效旋翼贡献。
        # 则有：
        #   总推力：thrust = b * (x + y)
        #   偏航力矩：yaw = d * (x - y)
        # 这里优先让 x + y（总推力）尽量贴近 thrust_cmd，
        # 再把期望偏航投影到旋翼上下限定义的可行域内。
        thrust_sum, x_low, x_high = self._effective_split_bounds(thrust_cmd)
        tau_z_limited = self.clamp_tau_z(thrust_cmd, tau_z_cmd)
        yaw_diff_des = tau_z_limited / self.d

        x_des = 0.5 * (thrust_sum + yaw_diff_des)
        x = max(x_low, min(x_des, x_high))
        y = thrust_sum - x

        w1_sq = x / self.lower_rotor_eff
        w2_sq = y

        # === 4. 反算实际输出 ===
        thrust_actual = self.b * (w1_sq * self.lower_rotor_eff + w2_sq)
        tau_z_actual = self.d * (w1_sq * self.lower_rotor_eff - w2_sq)

        return thrust_actual, tau_z_actual, w1_sq, w2_sq

# ===================== 外环位置环 PID 控制器 =====================  “导航员”
class OuterPosController:
    """
    外环位置控制器 (经典 PID)：
    接收 RefPos，计算输出期望推力与姿态角。
    PID 算出的只是为了消除误差所需的【额外】加速度，而不是让飞机飞这条轨迹所需的【物理】加速度。
    
    [优化] 新增功能:
    1. 微分项滤波 - 减少噪声放大
    2. 条件积分 - 防止积分饱和
    3. 抗饱和机制 - 执行器饱和时回退积分
    """

    def __init__(self,
                 P: Params,
                 dt_outer: float = 0.04,
                 # PID 参数：
                 kp: tuple = (2.20, 2.20, 4.3),
                 ki: tuple = (0.18, 0.18, 0.24),
                 kd: tuple = (0.85, 0.85, 2.4),
                 # 积分限幅
                 int_lim: tuple = (18.0, 18.0, 1.6),
                 # 输出姿态限幅
                 max_angle_deg: float = 25.0):
        self.P = P
        self.dt_outer = float(dt_outer)

        # 参数解包
        self.kp_x, self.kp_y, self.kp_z = kp
        self.ki_x, self.ki_y, self.ki_z = ki
        self.kd_x, self.kd_y, self.kd_z = kd

        self.ilim_x, self.ilim_y, self.ilim_z = int_lim
        self.roll_limit = math.radians(max_angle_deg)
        self.pitch_limit = math.radians(max_angle_deg)

        # 积分累积器
        self.sum_ex = 0.0
        self.sum_ey = 0.0
        self.sum_ez = 0.0

        # 微分滤波器状态
        self.edx_filt = 0.0
        self.edy_filt = 0.0
        self.edz_filt = 0.0
        self.d_filter_alpha = 0.85  # 微分滤波系数（0 到 1，越大响应越快）

        # 条件积分阈值
        self.sat_threshold = 0.95  # 姿态饱和阈值（比例）
        self.x_int_err_gate = 2.50
        self.x_int_vel_gate = 0.16
        self.y_int_err_gate = self.x_int_err_gate
        self.y_int_vel_gate = self.x_int_vel_gate
        self.x_int_decay = 0.995
        self.y_int_decay = self.x_int_decay
        self.x_track_int_err_gate = 1.20
        self.y_track_int_err_gate = self.x_track_int_err_gate
        self.x_track_int_vel_gate = 0.35
        self.y_track_int_vel_gate = self.x_track_int_vel_gate
        self.x_track_int_decay = 0.998
        self.y_track_int_decay = self.x_track_int_decay
        # 水平抗扰控制权限应由姿态安全包络决定，而不是由轨迹规划的
        # 名义加速度限制决定。旧值 0.55 m/s^2 会把无 NDO 对照组限制在
        # 约 3--5° 小俯仰，导致强风下不是“控制效果较差”，而是持续平移跑飞。
        # 这里保留 5% 余量，避免外环长期贴满姿态硬限。
        self.horizontal_accel_limit_margin = 0.95
        self.ax_cmd_limit = self._attitude_limited_horizontal_accel_limit(self.pitch_limit)
        self.ay_cmd_limit = self._attitude_limited_horizontal_accel_limit(self.roll_limit)
        # 悬停点附近的极小测速噪声不应持续变成水平小倾角。
        self.hold_xy_err_deadband = 0.03
        self.hold_xy_vel_deadband = 0.04
        # 当目标已静止且接近目标点时，额外加入一点速度阻尼，
        # 帮助压掉耦合场景在目标附近出现的水平打圈极限环。
        # 终端速度阻尼仍保留 1 m 左右的捕获半径，用来压制到点后的
        # 低速小半径绕圈；但积分洗出半径必须更小，否则关闭 NDO
        # 组在 3--5 m/s 持续横风下的 0.5--1.5 m 级稳态偏差会被
        # 误判成“已到点”，反馈基线无法继续积累抗风积分。
        self.terminal_hover_capture_radius = 1.0
        self.terminal_hover_integral_decay_radius = 0.20
        self.terminal_hover_velocity_damping = 1.6
        self.terminal_hover_integral_decay = 0.85
        # 控制侧的垂向推力校准，用于补偿 Gazebo 中的悬停偏差。
        self.z_thrust_scale = 1.0

    def _attitude_limited_horizontal_accel_limit(self, angle_limit: float) -> float:
        limit = abs(float(angle_limit))
        if not math.isfinite(limit) or limit <= 0.0:
            return 0.55
        margin = clamp(float(getattr(self, "horizontal_accel_limit_margin", 0.95)), 0.1, 1.0)
        return max(0.55, margin * self.P.g * math.tan(limit))

    @staticmethod
    def _has_dynamic_vertical_reference(ref: RefPos) -> bool:
        return abs(float(ref.vz)) > 1e-6 or abs(float(ref.az)) > 1e-6

    def _is_stationary_horizontal_target(self, ref: RefPos) -> bool:
        return (
            abs(ref.vx) <= 1e-6
            and abs(ref.vy) <= 1e-6
            and abs(ref.ax) <= 1e-6
            and abs(ref.ay) <= 1e-6
        )

    @staticmethod
    def _axis_has_dynamic_reference(ref_v: float, ref_a: float) -> bool:
        return abs(float(ref_v)) > 1e-6 or abs(float(ref_a)) > 1e-6

    @staticmethod
    def _dynamic_tracking_velocity_gate(base_gate: float, ref_v: float) -> float:
        return max(float(base_gate), 0.35 * abs(float(ref_v)) + 0.08)

    def _terminal_hover_capture_blend(
        self,
        ex: float,
        ey: float,
        *,
        radius: Optional[float] = None,
    ) -> float:
        capture_radius = (
            float(self.terminal_hover_capture_radius)
            if radius is None
            else float(radius)
        )
        capture_radius = max(capture_radius, 1e-6)
        radius = math.hypot(float(ex), float(ey))
        if radius >= capture_radius:
            return 0.0
        return 1.0 - clamp(radius / capture_radius, 0.0, 1.0)

    def _is_near_stationary_horizontal_hold(
        self,
        s: State6,
        ref: RefPos,
        ex: float,
        ey: float,
    ) -> bool:
        return (
            self._is_stationary_horizontal_target(ref)
            and abs(ex) <= self.hold_xy_err_deadband
            and abs(ey) <= self.hold_xy_err_deadband
            and abs(s.vx) <= self.hold_xy_vel_deadband
            and abs(s.vy) <= self.hold_xy_vel_deadband
        )

    def _apply_stationary_target_braking(
        self,
        pos_err: float,
        vel: float,
        accel_cmd: float,
        accel_limit: float,
    ) -> float:
        accel_limit = max(float(accel_limit), 1e-6)
        pos_err = float(pos_err)
        vel = float(vel)
        accel_cmd = float(accel_cmd)

        if abs(pos_err) <= 1e-6 or abs(vel) <= 1e-6:
            return accel_cmd
        # 仅在“目标静止且仍朝目标方向冲得过快”时追加刹车，不影响远距离正常加速段。
        if pos_err * vel <= 0.0:
            return accel_cmd

        stopping_distance = vel * vel / (2.0 * accel_limit)
        if stopping_distance <= abs(pos_err):
            return accel_cmd

        required_brake = clamp(
            vel * vel / (2.0 * max(abs(pos_err), 1e-6)),
            0.0,
            accel_limit,
        )
        desired_accel = -math.copysign(required_brake, vel)
        if vel > 0.0:
            return min(accel_cmd, desired_accel)
        return max(accel_cmd, desired_accel)

    def _apply_terminal_hover_damping(
        self,
        *,
        ex: float,
        ey: float,
        vx: float,
        vy: float,
        ax_cmd: float,
        ay_cmd: float,
    ) -> tuple[float, float]:
        blend = self._terminal_hover_capture_blend(ex, ey)
        if blend <= 0.0:
            return float(ax_cmd), float(ay_cmd)

        damping_gain = float(self.terminal_hover_velocity_damping) * blend
        return (
            float(ax_cmd) - damping_gain * float(vx),
            float(ay_cmd) - damping_gain * float(vy),
        )

    def compute(
        self,
        s: State6,
        ref: RefPos,
        *,
        attitude_mapping_psi: Optional[float] = None,
    ):
        """
        计算外环控制量
        """
        # 1. 计算位置误差向量，供比例项使用
        ex = ref.x - s.x
        ey = ref.y - s.y
        ez = ref.z - s.z

        # 2. 计算速度误差，供微分项使用
        edx = ref.vx - s.vx
        edy = ref.vy - s.vy
        # 静态悬停目标时，Z 轴继续采用“收敛到定点”的口径；
        # 但对 YZ 横八字这类动态高度参考，必须恢复对 ref.vz/ref.az
        # 的跟踪，否则 Z 轴只会把轨迹当成一串静态高度点来追，
        # 在 3D 轨迹图上表现成明显的垂向相位滞后。
        dynamic_vertical_reference = self._has_dynamic_vertical_reference(ref)
        edz = (ref.vz - s.vz) if dynamic_vertical_reference else (-s.vz)

        near_stationary_hold = self._is_near_stationary_horizontal_hold(s, ref, ex, ey)
        stationary_horizontal_target = self._is_stationary_horizontal_target(ref)
        dynamic_x_target = self._axis_has_dynamic_reference(ref.vx, ref.ax)
        dynamic_y_target = self._axis_has_dynamic_reference(ref.vy, ref.ay)
        terminal_hover_blend = 0.0
        if near_stationary_hold:
            ex = 0.0
            ey = 0.0
            edx = 0.0
            edy = 0.0
            self.sum_ex = 0.0
            self.sum_ey = 0.0
            self.edx_filt = 0.0
            self.edy_filt = 0.0

        # 微分项一阶低通滤波
        self.edx_filt = self.d_filter_alpha * edx + (1 - self.d_filter_alpha) * self.edx_filt
        self.edy_filt = self.d_filter_alpha * edy + (1 - self.d_filter_alpha) * self.edy_filt
        self.edz_filt = self.d_filter_alpha * edz + (1 - self.d_filter_alpha) * self.edz_filt

        if stationary_horizontal_target and not near_stationary_hold:
            terminal_hover_blend = self._terminal_hover_capture_blend(
                ex,
                ey,
                radius=float(self.terminal_hover_integral_decay_radius),
            )

        if terminal_hover_blend > 0.0:
            decay = 1.0 - (1.0 - float(self.terminal_hover_integral_decay)) * terminal_hover_blend
            self.sum_ex *= decay
            self.sum_ey *= decay
        else:
            x_int_err_gate = self.x_track_int_err_gate if dynamic_x_target else self.x_int_err_gate
            x_int_vel_metric = edx if dynamic_x_target else float(s.vx)
            x_int_vel_gate = (
                self._dynamic_tracking_velocity_gate(self.x_track_int_vel_gate, ref.vx)
                if dynamic_x_target
                else self.x_int_vel_gate
            )
            x_int_decay = self.x_track_int_decay if dynamic_x_target else self.x_int_decay
            if abs(ex) < x_int_err_gate and abs(x_int_vel_metric) < x_int_vel_gate:
                self.sum_ex = clamp(self.sum_ex + ex * self.dt_outer, -self.ilim_x, self.ilim_x)
            else:
                self.sum_ex *= x_int_decay

            y_int_err_gate = self.y_track_int_err_gate if dynamic_y_target else self.y_int_err_gate
            y_int_vel_metric = edy if dynamic_y_target else float(s.vy)
            y_int_vel_gate = (
                self._dynamic_tracking_velocity_gate(self.y_track_int_vel_gate, ref.vy)
                if dynamic_y_target
                else self.y_int_vel_gate
            )
            y_int_decay = self.y_track_int_decay if dynamic_y_target else self.y_int_decay
            if abs(ey) < y_int_err_gate and abs(y_int_vel_metric) < y_int_vel_gate:
                self.sum_ey = clamp(self.sum_ey + ey * self.dt_outer, -self.ilim_y, self.ilim_y)
            else:
                self.sum_ey *= y_int_decay

        self.sum_ez = clamp(self.sum_ez + ez * self.dt_outer, -self.ilim_z, self.ilim_z)

        # 4. PID 控制律（使用滤波后的微分项）
        ux = self.kp_x * ex + self.ki_x * self.sum_ex + self.kd_x * self.edx_filt
        uy = self.kp_y * ey + self.ki_y * self.sum_ey + self.kd_y * self.edy_filt
        uz = self.kp_z * ez + self.ki_z * self.sum_ez + self.kd_z * self.edz_filt

        # 5. 前馈控制
        # 最终期望加速度 = 规划器给出的理论加速度（ref.ax）+ PID 给出的补偿加速度（ux）
        ax_cmd = ref.ax + ux
        ay_cmd = ref.ay + uy
        az_cmd = (ref.az + uz) if dynamic_vertical_reference else uz

        if stationary_horizontal_target and abs(ref.vx) <= 1e-6 and abs(ref.ax) <= 1e-6:
            ax_cmd = self._apply_stationary_target_braking(ex, s.vx, ax_cmd, self.ax_cmd_limit)
        if stationary_horizontal_target and abs(ref.vy) <= 1e-6 and abs(ref.ay) <= 1e-6:
            ay_cmd = self._apply_stationary_target_braking(ey, s.vy, ay_cmd, self.ay_cmd_limit)
        if stationary_horizontal_target and not near_stationary_hold:
            ax_cmd, ay_cmd = self._apply_terminal_hover_damping(
                ex=ex,
                ey=ey,
                vx=s.vx,
                vy=s.vy,
                ax_cmd=ax_cmd,
                ay_cmd=ay_cmd,
            )

        # 水平位置环先直接限加速度幅值，避免再通过参考平滑引入额外相位滞后。
        ax_cmd = clamp(ax_cmd, -self.ax_cmd_limit, self.ax_cmd_limit)
        ay_cmd = clamp(ay_cmd, -self.ay_cmd_limit, self.ay_cmd_limit)

        # 6. 几何映射
        thrust, phi, theta, yaw = self._accel_to_attitude(
            ax_cmd,
            ay_cmd,
            az_cmd,
            ref.psi,
            attitude_mapping_psi=attitude_mapping_psi,
        )

        # 抗饱和机制：X 由 theta 控制且同号，Y 由 phi 控制且在 ENU 下反号。
        if abs(theta) >= self.pitch_limit * self.sat_threshold:
            if theta > 0:
                self.sum_ex = min(self.sum_ex, 0.0)
            else:
                self.sum_ex = max(self.sum_ex, 0.0)

        if abs(phi) >= self.roll_limit * self.sat_threshold:
            if phi > 0:
                self.sum_ey = max(self.sum_ey, 0.0)
            else:
                self.sum_ey = min(self.sum_ey, 0.0)

        # 地面待机锁：高度 < 0.1m 时，强制清空 XY 积分，并要求 0 姿态。
        if s.z < 0.1:
            self.sum_ex = 0.0
            self.sum_ey = 0.0
            phi = 0.0
            theta = 0.0

        return thrust, phi, theta, yaw

    def _accel_to_attitude(
        self,
        ax,
        ay,
        az,
        psi_ref,
        *,
        attitude_mapping_psi: Optional[float] = None,
    ):
        """
        将期望加速度映射为欧拉角和推力。
        """
        g = self.P.g
        m = self.P.M

        # 垂直方向合加速度
        acc_z_net = g + az
        if abs(acc_z_net) < 1e-4: acc_z_net = 1e-4

        # 辅助变量
        # 这些几何映射得到的公式  我还没深入推导过这个几何映射的过程
        A = ax / acc_z_net
        B = ay / acc_z_net
        yaw = psi_ref
        yaw_for_mapping = yaw if attitude_mapping_psi is None else float(attitude_mapping_psi)

        # 俯仰角
        # 假设小角度：ax ~ theta * g（向前倾斜产生向前加速度）
        tan_theta = A * math.cos(yaw_for_mapping) + B * math.sin(yaw_for_mapping)
        theta = math.atan(tan_theta)  # 使用反正切求解

        # 滚转角
        # 关键符号修正：
        # 之前使用的是 (B*cos - A*sin)。
        # 在当前物理模型和 ENU 坐标定义下，正滚转角会产生负 Y 加速度。
        # 因此如果希望获得正 Y 加速度（B > 0），就需要负滚转角，
        # 所以这里要写成 - (B*cos - A*sin)。
        tan_phi = - (B * math.cos(yaw_for_mapping) - A * math.sin(yaw_for_mapping)) * math.cos(theta)
        phi = math.atan(tan_phi)  # 使用反正切求解

        # 限幅
        phi = clamp(phi, -self.roll_limit, self.roll_limit)
        theta = clamp(theta, -self.pitch_limit, self.pitch_limit)

        # 计算总推力
        # 推力要与限幅后的姿态保持一致，避免角度被截断后仍沿用过大的推力补偿。
        thrust = self.z_thrust_scale * m * acc_z_net / (math.cos(phi) * math.cos(theta))
        thrust = clamp(thrust, self.P.thrust_min, self.P.thrust_max)

        return thrust, phi, theta, yaw

    def retarget_thrust_for_attitude(
        self,
        thrust_cmd: float,
        phi_old: float,
        theta_old: float,
        phi_new: float,
        theta_new: float,
    ) -> float:
        """保持同一垂向净加速度需求，同时更新倾斜后的总推力补偿。"""
        cos_old = math.cos(phi_old) * math.cos(theta_old)
        cos_new = math.cos(phi_new) * math.cos(theta_new)
        cos_old = max(cos_old, 0.1)
        cos_new = max(cos_new, 0.1)
        thrust = thrust_cmd * (cos_old / cos_new)
        return clamp(thrust, self.P.thrust_min, self.P.thrust_max)

    def reset_horizontal_hold_state(self):
        self.sum_ex = 0.0
        self.sum_ey = 0.0
        self.edx_filt = 0.0
        self.edy_filt = 0.0

def build_casadi_dynamics(P: Params):
    x = ca.SX.sym('x', 16)
    u = ca.SX.sym('u', 3)
    thrust = ca.SX.sym('thrust')

    # 状态顺序：x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, chi, chi_d, ups, ups_d
    vx_g, vy_g, vz_g = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    chi, chi_d = x[12], x[13]
    ups, ups_d = x[14], x[15]

    # 控制输入顺序：u = [tau_z, chi_cmd, ups_cmd]
    tau_z, chi_cmd, ups_cmd = u[0], u[1], u[2]

    cphi = ca.cos(phi)
    sphi = ca.sin(phi)
    ctheta = ca.cos(theta)
    stheta = ca.sin(theta)
    cpsi = ca.cos(psi)
    spsi = ca.sin(psi)

    Tb_v = ca.vertcat(
        ca.horzcat(cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi),
        ca.horzcat(spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi),
        ca.horzcat(-stheta, ctheta * sphi, ctheta * cphi)
    )
    Tv_b = Tb_v.T

    v_ground = ca.vertcat(vx_g, vy_g, vz_g)
    v_air_body = Tv_b @ v_ground
    vx_air, vy_air, vz_air = v_air_body[0], v_air_body[1], v_air_body[2]

    Fx_drag = -0.5 * P.rho * P.S_side * P.Cd_side * vx_air * ca.fabs(vx_air)
    Fy_drag = -0.5 * P.rho * P.S_side * P.Cd_side * vy_air * ca.fabs(vy_air)
    Fz_drag = -0.5 * P.rho * P.S_top * P.Cd_top * vz_air * ca.fabs(vz_air)
    F_drag_body = ca.vertcat(Fx_drag, Fy_drag, Fz_drag)
    Fd = Tb_v @ F_drag_body

    w_b = ca.vertcat(p, q, r)
    M_d_params = ca.DM([
        [P.kp, 0.0, 0.0],
        [0.0, P.kq, 0.0],
        [0.0, 0.0, P.kr]
    ])
    Md = -M_d_params @ w_b

    wn, zt = P.wn_mass, P.zeta_mass
    chi_dd = wn * wn * (chi_cmd - chi) - 2.0 * zt * wn * chi_d
    ups_dd = wn * wn * (ups_cmd - ups) - 2.0 * zt * wn * ups_d

    S = cal_sigma(P, chi, ups, chi_d, ups_d)
    I = cal_inertia(P, S)

    M = P.M
    mu = P.mu
    m = P.m
    b = P.b_thrust
    g = P.g

    A_dyn = ca.vertcat(
        ca.horzcat(1.0, I.Ixy / I.Ixx, 0.0),
        ca.horzcat(I.Ixy / I.Iyy, 1.0, 0.0),
        ca.horzcat(0.0, 0.0, 1.0)
    )

    sum_w1w2 = thrust / b

    B_p = (-I.dIxx * p / I.Ixx - I.dIxy * q / I.Ixx
           - q * I.Izz * r / I.Ixx + r * I.Iyy * q / I.Ixx
           - q * (4.0 * m * mu * ups * chi_d - 4.0 * m * mu * chi * ups_d) / I.Ixx
           - 2.0 * b * mu * ups * sum_w1w2 / I.Ixx
           + Md[0] / I.Ixx)

    B_q = (-I.dIxy * p / I.Iyy - I.dIyy * q / I.Iyy
           - r * (I.Ixx * p + I.Ixy * q) / I.Iyy + p * I.Izz * r / I.Iyy
           - p * (4.0 * m * mu * ups * chi_d - 4.0 * m * mu * chi * ups_d) / I.Iyy
           + 2.0 * b * mu * chi * sum_w1w2 / I.Iyy
           + Md[1] / I.Iyy)

    B_r = (-I.dIzz * r / I.Izz - p * (I.Ixy * p + I.Iyy * q) / I.Izz
           + q * (I.Ixx * p + I.Ixy * q) / I.Izz
           - 4.0 * m * mu * ups * chi_dd / I.Izz - 4.0 * m * mu * chi * ups_dd / I.Izz
           + tau_z / I.Izz
           + Md[2] / I.Izz)

    B_vec = ca.vertcat(B_p, B_q, B_r)
    w_dot = ca.solve(A_dyn, B_vec)
    pd, qd, rd = w_dot[0], w_dot[1], w_dot[2]

    x_dd = (thrust / M * (cphi * stheta * cpsi + sphi * spsi)
            - mu * (2.0 * chi_dd - 2.0 * r * ups_d)
            + 2.0 * rd * mu * ups + 2.0 * r * mu * ups_d
            - (2.0 * mu * ups * p * q
               - 2.0 * mu * chi * q * q
               - 2.0 * mu * chi * r * r)
            + Fd[0] / M)

    y_dd = (thrust / M * (cphi * stheta * spsi - sphi * cpsi)
            - mu * (2.0 * ups_dd + 2.0 * r * chi_d)
            - (2.0 * rd * mu * chi + 2.0 * r * mu * chi_d)
            + 2.0 * mu * ups * q * q
            + 2.0 * mu * ups * p * p
            + 2.0 * mu * chi * p * q
            + Fd[1] / M)

    z_dd = (thrust / M * cphi * ctheta - g
            - mu * (2.0 * p * ups_d - 2.0 * q * chi_d)
            - (2.0 * pd * mu * ups - 2.0 * qd * mu * chi
               + 2.0 * p * mu * ups_d - 2.0 * q * mu * chi_d)
            - (2.0 * mu * ups * q * r + 2.0 * mu * chi * p * r)
            + Fd[2] / M)

    ttheta = ca.tan(theta)
    phid = p + r * (cphi * ttheta) + q * (sphi * ttheta)
    thetad = q * cphi - r * sphi
    psid = r * (cphi / ctheta) + q * (sphi / ctheta)

    dx = ca.vertcat(
        vx_g, vy_g, vz_g,
        x_dd, y_dd, z_dd,
        phid, thetad, psid,
        pd, qd, rd,
        chi_d, chi_dd, ups_d, ups_dd
    )

    A = ca.jacobian(dx, x)
    B = ca.jacobian(dx, u)
    return ca.Function('get_AB', [x, u, thrust], [A, B, dx])

def compute_numerical_AB(P: Params, s_op: State6, u_op: np.ndarray, thrust_cmd: float, epsilon=1e-5):
    """
    使用中心差分法计算雅可比矩阵 A 和 B。
    优点：自动同步 update_6dof_model 的任何修改，物理绝对一致。
    缺点：比解析法慢，但对于 Python 仿真通常可接受。
    """
    nx = 16
    nu = 3

    # 将状态和输入转为向量
    # 注意：这里需要一个辅助函数把状态向量转回 State6，或者手动赋值
    # 为了方便，这里在函数内部直接修改 s_op 的副本来实现

    # 基础工作点导数 f0
    # 构造 Input6 对象
    u_base = Input6(thrust=thrust_cmd, tau_z=u_op[0], chi_cmd=u_op[1], ups_cmd=u_op[2])
    dx_base = update_6dof_model(P, s_op, u_base)
    f0 = np.array([dx_base.x, dx_base.y, dx_base.z, dx_base.vx, dx_base.vy, dx_base.vz,
                   dx_base.phi, dx_base.theta, dx_base.psi, dx_base.p, dx_base.q, dx_base.r,
                   dx_base.chi, dx_base.chi_d, dx_base.ups, dx_base.ups_d])

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    # --- 计算 A 矩阵（df/dx） ---
    # 状态列表顺序对应：x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, chi, chi_d, ups, ups_d
    # 为了通用性，利用 State6 的 __dict__ 进行微扰会更整洁，但为了性能我们手动映射
    state_vars = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'chi', 'chi_d', 'ups', 'ups_d']

    for i, var_name in enumerate(state_vars):
        # 保存原始值
        original_val = getattr(s_op, var_name)  # 等价于 original_val = s_op.var_name

        # 正向微扰
        setattr(s_op, var_name, original_val + epsilon)  # 等价于 s_op.var_name = original_val + epsilon
        dx_plus = update_6dof_model(P, s_op, u_base)
        f_plus = np.array([dx_plus.x, dx_plus.y, dx_plus.z, dx_plus.vx, dx_plus.vy, dx_plus.vz,
                           dx_plus.phi, dx_plus.theta, dx_plus.psi, dx_plus.p, dx_plus.q, dx_plus.r,
                           dx_plus.chi, dx_plus.chi_d, dx_plus.ups, dx_plus.ups_d])

        # 反向微扰
        setattr(s_op, var_name, original_val - epsilon)
        dx_minus = update_6dof_model(P, s_op, u_base)
        f_minus = np.array([dx_minus.x, dx_minus.y, dx_minus.z, dx_minus.vx, dx_minus.vy, dx_minus.vz,
                            dx_minus.phi, dx_minus.theta, dx_minus.psi, dx_minus.p, dx_minus.q, dx_minus.r,
                            dx_minus.chi, dx_minus.chi_d, dx_minus.ups, dx_minus.ups_d])

        # 恢复原始值 !!! 非常重要
        setattr(s_op, var_name, original_val)

        # 中心差分
        A[:, i] = (f_plus - f_minus) / (2 * epsilon)

    # --- 计算 B 矩阵（df/du） ---
    # 输入顺序：tau_z, chi_cmd, ups_cmd
    u_vars = [0, 1, 2]  # 对应 u_op 的索引

    for i in range(nu):
        # 保存原始输入
        orig_u = u_op[i]

        # 正向
        u_op[i] = orig_u + epsilon
        u_plus = Input6(thrust=thrust_cmd, tau_z=u_op[0], chi_cmd=u_op[1], ups_cmd=u_op[2])
        dx_plus = update_6dof_model(P, s_op, u_plus)
        f_plus = np.array([dx_plus.x, dx_plus.y, dx_plus.z, dx_plus.vx, dx_plus.vy, dx_plus.vz,
                           dx_plus.phi, dx_plus.theta, dx_plus.psi, dx_plus.p, dx_plus.q, dx_plus.r,
                           dx_plus.chi, dx_plus.chi_d, dx_plus.ups, dx_plus.ups_d])

        # 反向
        u_op[i] = orig_u - epsilon
        u_minus = Input6(thrust=thrust_cmd, tau_z=u_op[0], chi_cmd=u_op[1], ups_cmd=u_op[2])
        dx_minus = update_6dof_model(P, s_op, u_minus)
        f_minus = np.array([dx_minus.x, dx_minus.y, dx_minus.z, dx_minus.vx, dx_minus.vy, dx_minus.vz,
                            dx_minus.phi, dx_minus.theta, dx_minus.psi, dx_minus.p, dx_minus.q, dx_minus.r,
                            dx_minus.chi, dx_minus.chi_d, dx_minus.ups, dx_minus.ups_d])

        # 恢复
        u_op[i] = orig_u

        B[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return A, B

# ===================== 内环：LPV-MPC 姿态控制（tau_z, chi_cmd, ups_cmd） =====================
class AttitudeMPC:
    """
    内环姿态 LPV + MPC 控制器（纯 Np 预测时域，NumPy + OSQP QP 路径）
    
    [控制权限说明]
    ==============
    MPC控制器是系统中唯一具备滑块位移命令输出权限的模块。
    
    输入来源：
    - 外环PID：期望推力 thrust_cmd
    - 轨迹规划器：期望姿态角序列 ref_att_seq (包含外环PID的输出)
    - NDO模块：姿态角补偿已叠加到ref_att_seq中
    
    输出：
    - tau_z：偏航力矩
    - chi_cmd：纵向滑块位移命令 (唯一输出源)
    - ups_cmd：横向滑块位移命令 (唯一输出源)
    
    统一架构优势：
    - 避免多模块同时输出滑块位移导致的过补偿
    - MPC可以综合考虑姿态跟踪和滑块约束
    - 实现平滑的姿态-滑块协调控制
    
    y_tilde = [phi, theta, psi, p, q, r, chi, chi_d, ups, ups_d, tau_z, chi_cmd, ups_cmd]
    """

    def __init__(self,
                 P: Params,
                 Np: int = 35,
                 dt_mpc: float = 0.01,
                 # 输出权重（Q）
                 q_phi: float = 700.0,    
                 q_theta: float = 700.0,
                 q_psi: float = 600.0,    
                 q_p: float = 100.0,      
                 q_q: float = 100.0,
                 q_r: float = 200.0,    
                 q_chi: float = 0.05,     
                 q_chi_d: float = 0.2,
                 q_ups: float = 0.05,
                 q_ups_d: float = 0.2,
                 # 绝对控制量 [u] 的权重，加入 Q_aug 后对应后 3 维
                 r_tauz: float = 1.0,
                 r_chi: float = 0.15,
                 r_ups: float = 0.15,
                 # 增量控制量 [Δu] 的权重
                 rd_tauz: float = 1.0,
                 rd_chi: float = 1.2,
                 rd_ups: float = 1.2,
                 # 偏航力矩限幅
                 tau_z_lim: float = 1.0,
                 slider_soft_limit: float = 0.135):
        self.P = P
        self.dt = P.dt if dt_mpc is None else float(dt_mpc)

        self.nu = 3
        self.nx_state = 16
        self.nx = self.nx_state + self.nu  # 19 维 = [x; u_{k-1}]
        self.ny_state = 10
        self.ny = self.ny_state + self.nu  # 13 维 = [y; u]
        self.Np = max(int(Np), 1)

        # 基础输出选择矩阵 C （10x16）
        self.C_state = np.zeros((self.ny_state, self.nx_state), dtype=float)
        self.C_state[0, 6] = 1.0
        self.C_state[1, 7] = 1.0
        self.C_state[2, 8] = 1.0  # phi、theta、psi
        self.C_state[3, 9] = 1.0
        self.C_state[4, 10] = 1.0
        self.C_state[5, 11] = 1.0  # p、q、r
        self.C_state[6, 12] = 1.0
        self.C_state[7, 13] = 1.0
        self.C_state[8, 14] = 1.0
        self.C_state[9, 15] = 1.0  # chi、chi_d、ups、ups_d

        # 扩维输出矩阵 C_tilde = [[C,0],[0,I]]
        self.C = np.zeros((self.ny, self.nx), dtype=float)
        self.C[:self.ny_state, :self.nx_state] = self.C_state
        self.C[self.ny_state:, self.nx_state:] = np.eye(self.nu)

        self.slider_actuator_cost_profile = build_slider_actuator_cost_profile(
            self.P,
            q_chi=q_chi,
            q_chi_d=q_chi_d,
            q_ups=q_ups,
            q_ups_d=q_ups_d,
            r_chi=r_chi,
            r_ups=r_ups,
            rd_chi=rd_chi,
            rd_ups=rd_ups,
        )

        # 权重矩阵
        # Q 为状态权重矩阵 数值越大，代表越在乎。
        slider_costs = self.slider_actuator_cost_profile
        self.Q_y = np.diag(
            [
                q_phi,
                q_theta,
                q_psi,
                q_p,
                q_q,
                q_r,
                slider_costs.q_chi,
                slider_costs.q_chi_d,
                slider_costs.q_ups,
                slider_costs.q_ups_d,
            ]
        )
        # R_u 为绝对控制量 [u] 权重，R_delta 为增量 [Δu] 权重
        self.R_u = np.diag([r_tauz, slider_costs.r_chi, slider_costs.r_ups])
        self.R_delta = np.diag([rd_tauz, slider_costs.rd_chi, slider_costs.rd_ups])

        self.Q_aug = np.zeros((self.ny, self.ny), dtype=float)
        self.Q_aug[:self.ny_state, :self.ny_state] = self.Q_y
        self.Q_aug[self.ny_state:, self.ny_state:] = self.R_u
        self._Q_aug_diag = np.diag(self.Q_aug).copy()

        # np.kron(A, B) → 将矩阵 B 放在 A 的每个元素位置上进行复制和缩放
        # 最终得到大小为 (Np*Q_aug行数) × (Np*Q_aug列数) 的块对角矩阵
        self.Q_bar = np.kron(np.eye(self.Np), self.Q_aug)
        self.R_bar = np.kron(np.eye(self.Np), self.R_delta)

        # 约束限幅
        self.tau_z_lim = float(tau_z_lim)
        # 与实体滑轨硬限位保留安全裕度，
        # 避免 MPC 把撞限位当成正常工作点。
        self.u2_lim = min(float(self.P.u2_lim), max(1e-4, float(slider_soft_limit)))
        self.slider_vel_lim = float(self.P.slider_vel_max)
        self.delta_tau_z_lim = float(self.tau_z_lim)
        # 控制输入是“位置设定值”而不是“实际滑块速度”。
        # 若把 Δu 也硬绑到物理速度上限，MPC 会在姿态反转时被 2.5 mm/step 的
        # 设定值爬坡卡死，明明看到大滚转误差却只能一点点挪 setpoint。
        # 这里保留物理滑块速度约束给状态模型和 Gazebo 关节本体处理，同时把
        # setpoint 的单步变化放宽到物理速度上限的 6 倍，避免重复限速。
        self.delta_slider_lim = min(
            self.u2_lim,
            float(max(6.0 * self.P.slider_vel_max * self.dt, self.P.slider_vel_max * self.dt)),
        )

        # 失效保护相关参数
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.decay_factor = 0.95  # 控制量衰减系数

        # 缓存 LPV 矩阵
        self.A_d = np.eye(self.nx)
        self.B_d = np.zeros((self.nx, self.nu))
        self.c_d = np.zeros(self.nx)

        # 缓存二次规划相关矩阵（H、g、Gamma、Phi）
        self.H_mat = None
        self.Phi = None
        self.casadi_get_AB = build_casadi_dynamics(self.P)
        self.last_model_update_dt = 0.0
        self.last_qp_build_dt = 0.0
        self.last_solver_setup_dt = 0.0
        self.last_solver_solve_dt = 0.0
        self.last_model_reused = False
        self.last_lpv_reuse_count = 0
        self.last_ref_sequence_10 = np.zeros((self.Np, self.ny_state), dtype=float)
        self.last_Y_ref = np.zeros(self.ny * self.Np, dtype=float)
        self._n_dec = self.nu * self.Np
        self._constraint_A_sparse = None
        self._du_lb = None
        self._du_ub = None
        self._u_lb = None
        self._u_ub = None
        self._P_rows = np.concatenate(
            [np.arange(col + 1, dtype=int) for col in range(self._n_dec)]
        )
        self._P_cols = np.concatenate(
            [np.full(col + 1, col, dtype=int) for col in range(self._n_dec)]
        )
        self._cached_P_upper = None
        self._cached_P_upper_data = None
        self._cached_Y_const = np.zeros(self.ny * self.Np, dtype=float)
        self._osqp_prob = None
        self._osqp_is_ready = False
        self._lpv_reuse_max_skips = 0  # 兼容诊断字段：表示 LPV reuse 已禁用
        self.last_refresh_reason_mask = 0
        self._x_aug0_buffer = np.zeros(self.nx, dtype=float)
        self._u_prev_stack_buffer = np.zeros(self.nu * self.Np, dtype=float)
        self._l_vec_buffer = np.zeros(self._n_dec + self.nu * self.Np, dtype=float)
        self._u_vec_buffer = np.zeros(self._n_dec + self.nu * self.Np, dtype=float)
        self._g_vec_buffer = np.zeros(self._n_dec, dtype=float)
        self._H_buffer = np.zeros((self._n_dec, self._n_dec), dtype=float)
        self._tracking_error_blocks = np.zeros((self.Np, self.ny), dtype=float)
        self._Phi_buffer = np.zeros((self.ny * self.Np, self.nx), dtype=float)
        self._Gamma_buffer = np.zeros((self.ny * self.Np, self.nu * self.Np), dtype=float)
        self._CA_power_blocks = np.zeros((self.Np, self.ny, self.nx), dtype=float)
        self._CAB_power_blocks = np.zeros((self.Np, self.ny, self.nu), dtype=float)
        self._y_const_buffer = np.zeros(self.ny * self.Np, dtype=float)
        self._curr_c_accum_buffer = np.zeros(self.nx, dtype=float)
        self._A_d_base_buffer = np.zeros((self.nx_state, self.nx_state), dtype=float)
        self._B_d_base_buffer = np.zeros((self.nx_state, self.nu), dtype=float)
        self._c_d_base_buffer = np.zeros(self.nx_state, dtype=float)
        self._Ax_buffer = np.zeros(self.nx_state, dtype=float)
        self._Bu_buffer = np.zeros(self.nx_state, dtype=float)
        self._eye_nx_state = np.eye(self.nx_state, dtype=float)
        self._eye_nu = np.eye(self.nu, dtype=float)
        self._H_diag_indices = np.diag_indices(self._n_dec)
        self._build_static_qp_structures()

    def _state6_to_vec(self, s: State6) -> np.ndarray:
        x = np.zeros(self.nx_state, dtype=float)
        x[0], x[1], x[2] = s.x, s.y, s.z
        x[3], x[4], x[5] = s.vx, s.vy, s.vz
        x[6], x[7], x[8] = s.phi, s.theta, s.psi
        x[9], x[10], x[11] = s.p, s.q, s.r
        x[12], x[13] = s.chi, s.chi_d
        x[14], x[15] = s.ups, s.ups_d
        return x

    def _f_continuous(self, x_vec: np.ndarray, u_vec: np.ndarray, thrust_cmd: float) -> np.ndarray:
        # 辅助函数：用于数值线性化
        # 将状态向量转回 State6 对象进行计算
        s = State6(
            x=x_vec[0], y=x_vec[1], z=x_vec[2],
            vx=x_vec[3], vy=x_vec[4], vz=x_vec[5],
            phi=x_vec[6], theta=x_vec[7], psi=x_vec[8],
            p=x_vec[9], q=x_vec[10], r=x_vec[11],
            chi=x_vec[12], chi_d=x_vec[13],
            ups=x_vec[14], ups_d=x_vec[15]
        )
        inp = Input6(thrust=thrust_cmd, tau_z=u_vec[0], chi_cmd=u_vec[1], ups_cmd=u_vec[2])
        dx = update_6dof_model(self.P, s, inp)

        # 将 dx（State6）转回状态向量
        dx_vec = np.zeros(self.nx_state)
        dx_vec[0:3] = dx.x, dx.y, dx.z
        dx_vec[3:6] = dx.vx, dx.vy, dx.vz
        dx_vec[6:9] = dx.phi, dx.theta, dx.psi
        dx_vec[9:12] = dx.p, dx.q, dx.r
        dx_vec[12:14] = dx.chi, dx.chi_d
        dx_vec[14:16] = dx.ups, dx.ups_d
        return dx_vec

    def _linearize_AB(self, x0: np.ndarray, u_op: np.ndarray, thrust_cmd: float):
        A_ca, B_ca, f0_ca = self.casadi_get_AB(x0, u_op, thrust_cmd)
        return np.array(A_ca, dtype=float), np.array(B_ca, dtype=float), np.array(f0_ca, dtype=float).flatten()

    def _build_static_qp_structures(self):
        n_dec = self._n_dec
        nu = self.nu
        Np = self.Np
        du_lb_single = np.array(
            [-self.delta_tau_z_lim, -self.delta_slider_lim, -self.delta_slider_lim],
            dtype=float,
        )
        du_ub_single = np.array(
            [self.delta_tau_z_lim, self.delta_slider_lim, self.delta_slider_lim],
            dtype=float,
        )
        self._du_lb = np.tile(du_lb_single, Np)
        self._du_ub = np.tile(du_ub_single, Np)

        u_lb_single = np.array([-self.tau_z_lim, -self.u2_lim, -self.u2_lim], dtype=float)
        u_ub_single = np.array([self.tau_z_lim, self.u2_lim, self.u2_lim], dtype=float)
        self._u_lb = np.tile(u_lb_single, Np)
        self._u_ub = np.tile(u_ub_single, Np)

        I_dec = sparse.eye(n_dec, format='csc')
        L_scalar = sparse.tril(sparse.csc_matrix(np.ones((Np, Np), dtype=float)), format='csc')
        L_accum = sparse.kron(L_scalar, sparse.eye(nu, format='csc'), format='csc')
        self._constraint_A_sparse = sparse.vstack([I_dec, L_accum], format='csc')

    def _build_upper_triangular_hessian(self):
        upper_values = self._build_upper_triangular_hessian_data()
        return sparse.csc_matrix(
            (upper_values, (self._P_rows, self._P_cols)),
            shape=(self._n_dec, self._n_dec),
        )

    def _build_upper_triangular_hessian_data(self) -> np.ndarray:
        return 2.0 * self.H_mat[self._P_rows, self._P_cols]

    def _build_constant_response_vector(self) -> np.ndarray:
        y_const = self._y_const_buffer
        curr_c_accum = self._curr_c_accum_buffer
        y_const.fill(0.0)
        curr_c_accum.fill(0.0)
        for k in range(self.Np):
            curr_c_accum = self.A_d @ curr_c_accum + self.c_d
            idx = k * self.ny
            y_const[idx: idx + self.ny] = self.C @ curr_c_accum
        return y_const

    def _assemble_hessian_from_gamma(self) -> np.ndarray:
        H = self._H_buffer
        H[:, :] = self.R_bar
        q_diag = self._Q_aug_diag
        for k in range(self.Np):
            row = self.Gamma[k * self.ny: (k + 1) * self.ny, :]
            H += row.T @ (q_diag[:, None] * row)
        H[:] = 0.5 * (H + H.T)
        H[self._H_diag_indices] += 1e-4
        return H

    def _build_qp_gradient(self, tracking_error: np.ndarray) -> np.ndarray:
        error_blocks = self._tracking_error_blocks
        error_blocks[:, :] = tracking_error.reshape(self.Np, self.ny)
        error_blocks *= self._Q_aug_diag
        g_vec = self._g_vec_buffer
        g_vec.fill(0.0)
        for k in range(self.Np):
            row = self.Gamma[k * self.ny: (k + 1) * self.ny, :]
            g_vec += row.T @ error_blocks[k]
        return g_vec

    def _ensure_solver_ready(
        self,
        q_vec: np.ndarray,
        l_vec: np.ndarray,
        u_vec: np.ndarray,
    ):
        setup_start = time.perf_counter()
        if not self._osqp_is_ready or self._osqp_prob is None:
            if self._cached_P_upper is None:
                self._cached_P_upper = self._build_upper_triangular_hessian()
                self._cached_P_upper_data = self._cached_P_upper.data.copy()
            self._osqp_prob = osqp.OSQP()
            self._osqp_prob.setup(
                P=self._cached_P_upper,
                q=q_vec,
                A=self._constraint_A_sparse,
                l=l_vec,
                u=u_vec,
                warm_starting=True,
                verbose=False,
            )
            self._osqp_is_ready = True
        else:
            update_kwargs = {
                "q": q_vec,
                "l": l_vec,
                "u": u_vec,
            }
            self._cached_P_upper_data = self._build_upper_triangular_hessian_data()
            update_kwargs["Px"] = self._cached_P_upper_data
            self._osqp_prob.update(**update_kwargs)
        self.last_solver_setup_dt = time.perf_counter() - setup_start
        return self._osqp_prob

    def _update_prediction_rollout_blocks(self, A_d, B_d):
        ny, Np = self.ny, self.Np
        C = self.C
        Phi = self._Phi_buffer
        Phi.fill(0.0)

        ca_blocks = self._CA_power_blocks
        cab_blocks = self._CAB_power_blocks
        a_power = A_d.copy()
        ab_power = B_d.copy()
        for k in range(Np):
            ca_block = C @ a_power
            cab_block = C @ ab_power
            ca_blocks[k, :, :] = ca_block
            cab_blocks[k, :, :] = cab_block
            row = k * ny
            Phi[row:row + ny, :] = ca_block
            a_power = A_d @ a_power
            ab_power = A_d @ ab_power

        Gamma = self._Gamma_buffer
        Gamma.fill(0.0)
        for row_idx in range(Np):
            row = row_idx * ny
            for col_idx in range(row_idx + 1):
                col = col_idx * self.nu
                Gamma[row:row + ny, col:col + self.nu] = cab_blocks[row_idx - col_idx]
        return Phi, Gamma

    def _build_prediction_matrices(self, A_d, B_d):
        Phi, Gamma = self._update_prediction_rollout_blocks(A_d, B_d)
        return Phi, Gamma

    def _update_lpv_model(self, x0: np.ndarray, thrust_cmd: float, last_u: np.ndarray):
        # 1) 基础16维系统线性化与离散化
        A_c, B_c, f0 = self._linearize_AB(x0, last_u, thrust_cmd)
        dt = self.dt
        A_d_base = self._A_d_base_buffer
        np.multiply(A_c, dt, out=A_d_base)
        A_d_base += self._eye_nx_state

        B_d_base = self._B_d_base_buffer
        np.multiply(B_c, dt, out=B_d_base)

        c_d_base = self._c_d_base_buffer
        np.dot(A_c, x0, out=self._Ax_buffer)
        np.dot(B_c, last_u, out=self._Bu_buffer)
        c_d_base[:] = f0
        c_d_base -= self._Ax_buffer
        c_d_base -= self._Bu_buffer
        c_d_base *= dt

        # 2) 扩维系统：x_aug=[x;u_{k-1}]，控制变量为 Δu
        self.A_d[:self.nx_state, :self.nx_state] = A_d_base
        self.A_d[:self.nx_state, self.nx_state:] = B_d_base
        self.A_d[self.nx_state:, :self.nx_state] = 0.0
        self.A_d[self.nx_state:, self.nx_state:] = self._eye_nu
        self.B_d[:self.nx_state, :] = B_d_base
        self.B_d[self.nx_state:, :] = self._eye_nu
        self.c_d[:self.nx_state] = c_d_base
        self.c_d[self.nx_state:] = 0.0

        # 3) 预测矩阵与 Hessian 矩阵
        self.Phi, self.Gamma = self._build_prediction_matrices(self.A_d, self.B_d)
        self.H_mat = self._assemble_hessian_from_gamma()
        self._cached_P_upper = None
        self._cached_P_upper_data = None
        self._cached_Y_const = self._build_constant_response_vector()

    def _pack_reference_sequence(self, ref_seq: np.ndarray) -> np.ndarray:
        if ref_seq is None:
            src = np.zeros((1, self.ny_state), dtype=float)
        else:
            src = np.asarray(ref_seq, dtype=float)
            if src.ndim == 1:
                src = src.reshape(1, -1)
            if src.shape[0] == 0:
                src = np.zeros((1, self.ny_state), dtype=float)
            if src.shape[1] != self.ny_state:
                raise ValueError(
                    f"AttitudeMPC expects reference width {self.ny_state}, got {src.shape[1]}"
                )

        packed = self.last_ref_sequence_10
        copy_rows = min(src.shape[0], self.Np)
        packed[:copy_rows, :] = src[:copy_rows, :]
        if copy_rows < self.Np:
            packed[copy_rows:, :] = src[copy_rows - 1, :]
        return packed

    def compute(self, t: float, s: State6, thrust_cmd: float, ref_att_seq: np.ndarray,
                last_u_opt: np.ndarray = None):
        """
        求解带约束的二次规划问题（支持预测视域序列）
        :param last_u_opt: 上一时刻的控制输出 [tau_z, chi, ups]，用于失效保护时保底
        """
        x0 = self._state6_to_vec(s)

        if last_u_opt is None:
            last_u_opt = np.zeros(self.nu, dtype=float)
        else:
            last_u_opt = np.asarray(last_u_opt, dtype=float).reshape(self.nu)

        # 1) 更新 LPV 扩维模型
        model_update_start = time.perf_counter()
        self._update_lpv_model(x0, thrust_cmd, last_u_opt)
        self.last_model_reused = False
        self.last_lpv_reuse_count = 0
        self.last_refresh_reason_mask = 0
        self.last_model_update_dt = time.perf_counter() - model_update_start

        ny, nx, Np, nu = self.ny, self.nx, self.Np, self.nu
        qp_build_start = time.perf_counter()
        # 2) 扩维初值 x_aug0 = [x0; u_{k-1}]
        self._x_aug0_buffer[: self.nx_state] = x0
        self._x_aug0_buffer[self.nx_state:] = last_u_opt
        x_aug0 = self._x_aug0_buffer

        # 3) 预测自由响应 Y0 = Phi*x_aug0 + 常数项累积响应
        Y0 = self.Phi @ x_aug0
        Y0 += self._cached_Y_const

        # 4) 参考输出：
        # 前 10 维为状态跟踪项，后 3 维为绝对控制量参考（设为 0，滑块归中）
        packed_ref = self._pack_reference_sequence(ref_att_seq)
        Y_ref = self.last_Y_ref
        Y_ref.fill(0.0)
        Y_ref_matrix = Y_ref.reshape(Np, ny)
        Y_ref_matrix[:, : self.ny_state] = packed_ref

        # 5) 构建 QP 一次项
        tracking_error = Y0 - Y_ref
        g_vec = self._build_qp_gradient(tracking_error)

        # 6) OSQP 目标函数
        q_vec = 2.0 * g_vec

        # 7) 约束 A * ΔU ∈ [l, u]
        #    a) 增量限幅：I * ΔU
        #    b) 绝对限幅：L * ΔU，其中 U = 1⊗u_{-1} + LΔU
        self._u_prev_stack_buffer.reshape(Np, nu)[:, :] = last_u_opt
        u_prev_stack = self._u_prev_stack_buffer
        l_vec = self._l_vec_buffer
        u_vec = self._u_vec_buffer
        l_vec[: self._n_dec] = self._du_lb
        l_vec[self._n_dec:] = self._u_lb - u_prev_stack
        u_vec[: self._n_dec] = self._du_ub
        u_vec[self._n_dec:] = self._u_ub - u_prev_stack
        self.last_qp_build_dt = time.perf_counter() - qp_build_start

        # 8) 求解 OSQP
        prob = self._ensure_solver_ready(q_vec, l_vec, u_vec)
        solve_start = time.perf_counter()
        res = prob.solve()
        self.last_solver_solve_dt = time.perf_counter() - solve_start

        # 9) 提取首个增量并恢复绝对控制量
        if res.info.status_val == 1 or res.info.status_val == 2:
            self.consecutive_failures = 0
            delta_u0 = res.x[0:nu]
            u_opt_now = last_u_opt + delta_u0
            return u_opt_now[0], u_opt_now[1], u_opt_now[2]
        else:
            self.consecutive_failures += 1
            print(f"[Safe] OSQP failed (status: {res.info.status}, count={self.consecutive_failures}) at t={t:.2f}")

            if self.consecutive_failures >= self.max_consecutive_failures:
                print(f"[Emergency] Engaging hover mode: sliders to neutral")
                return 0.0, 0.0, 0.0
            else:
                safe_u = last_u_opt * self.decay_factor
                return safe_u[0], safe_u[1], safe_u[2]


class NDO_Observer:
    """
    双通道非线性干扰观测器 (Dual-Channel NDO) - 自适应增益版本

    理论框架
    --------
    基于论文中的NDO理论，针对变质心共轴双旋翼无人机设计双通道架构

    核心方程:
    - 平动NDO: d̂_f = z_f + L_f·v
              ż_f = -L_f·[a_known + d̂_f]
    - 转动NDO: d̂_τ = z_τ + L_τ·ω
              ż_τ = -L_τ·[α_known + d̂_τ]

    其中:
    - a_known = (F_thrust + F_gravity + F_drag) / M
    - α_known = J^(-1)·(τ_control + τ_gyro + τ_damping)
    - L_f, L_τ为观测器增益矩阵

    自适应增益机制
    --------------
    变质心特性导致转动惯量J随滑块位置变化，需要自适应增益:

    L_τ(χ, υ) = k_τ · J^(-1)(χ, υ)

    其中:
    - k_τ为基准增益系数
    - J^(-1)为当前转动惯量的逆矩阵
    - 增益随惯量增大而减小，保持观测器动态特性一致

    通道耦合处理
    ------------
    平动和转动之间存在动态耦合:
    1. 质心偏移导致推力产生附加力矩
    2. 滑块运动产生惯性力矩
    3. 需要在转动NDO中补偿这些耦合项

    控制权限管理
    ------------
    - NDO模块: 仅输出姿态角补偿，不直接输出滑块位移
    - MPC控制器: 唯一具备滑块位移命令输出权限的模块
    - 外环PID: 输出推力和期望姿态角
    """

    def __init__(self, P: Params,
                 base_gain_force: float = 15.0,
                 base_gain_torque: float = 15.0,
                 lpf_alpha: float = 0.4,
                 attitude_gain: float = 10.0,
                 enable_adaptive: bool = True,
                 coupling_compensation: bool = True):
        """
        初始化双通道NDO观测器

        参数说明
        --------
        P: 无人机参数对象
        base_gain_force: 平动NDO基准增益 [1/s]，影响观测速度
        base_gain_torque: 转动NDO基准增益 [1/s]，影响观测速度
        lpf_alpha: 低通滤波系数 [0-1]，越大响应越快但噪声越大
        attitude_gain: 姿态补偿增益，将干扰转换为姿态角的系数
        enable_adaptive: 是否启用自适应增益调节
        coupling_compensation: 是否启用通道耦合补偿

        增益整定建议
        ------------
        1. 基准增益范围: 10-20 [1/s]
           - 增益过小: 观测响应慢，补偿滞后
           - 增益过大: 观测噪声放大，系统振荡
           
        2. 自适应增益: 根据转动惯量变化自动调节
           - 惯量增大时增益减小，保持观测器带宽一致
           
        3. 低通滤波系数: 0.3-0.6
           - 系数小: 平滑性好但响应慢
           - 系数大: 响应快但噪声大
        """
        self.P = P

        # 基础参数
        self.base_gain_force = base_gain_force
        self.base_gain_torque = base_gain_torque
        self.lpf_alpha = lpf_alpha
        self.attitude_gain = attitude_gain
        self.enable_adaptive = enable_adaptive
        self.coupling_compensation = coupling_compensation

        # 通道 1：平动 NDO（力观测器）
        self.L_force = np.diag([base_gain_force] * 3)
        self.z_force = np.zeros(3)
        self.d_force_hat = np.zeros(3)

        # 通道 2：转动 NDO（力矩观测器）
        self.L_torque = np.diag([base_gain_torque] * 3)
        self.z_torque = np.zeros(3)
        self.d_torque_hat = np.zeros(3)

        # 滤波与状态缓存
        self.last_comp_phi_f = 0.0
        self.last_comp_theta_f = 0.0
        self.last_comp_phi_r = 0.0
        self.last_comp_theta_r = 0.0

        # 增益调节缓存
        self.last_L_torque = self.L_torque.copy()
        self.gain_adaptation_factor = 1.0

        # 观测值限幅保护，防止极端干扰时产生过激补偿
        self.d_force_max = 8.0      # 最大加速度干扰估计 [m/s^2]
        self.d_torque_max = 20.0     # 最大角加速度干扰估计 [rad/s^2]

        # 通道耦合补偿缓存
        self.last_coupling_torque = np.zeros(3)

    def update(self, x: State6, u: Input6, dt: float):
        """
        同时更新两个观测器
        """
        # ==========================================
        # 第一部分：力观测器
        # ==========================================
        v = np.array([x.vx, x.vy, x.vz])
        Tb_v, Tv_b = T_transformation(x)

        # 1.1 计算已知加速度（推力 + 重力 + 阻力）
        F_thrust_b = np.array([0.0, 0.0, u.thrust])
        acc_thrust_inertial = (Tb_v @ F_thrust_b) / self.P.M
        acc_gravity = np.array([0.0, 0.0, -self.P.g])

        # 阻力
        v_body = Tv_b @ v
        vx_b, vy_b, vz_b = v_body[0], v_body[1], v_body[2]
        Fx_drag = -0.5 * self.P.rho * self.P.S_side * self.P.Cd_side * vx_b * abs(vx_b)
        Fy_drag = -0.5 * self.P.rho * self.P.S_side * self.P.Cd_side * vy_b * abs(vy_b)
        Fz_drag = -0.5 * self.P.rho * self.P.S_top * self.P.Cd_top * vz_b * abs(vz_b)
        acc_drag_inertial = (Tb_v @ np.array([Fx_drag, Fy_drag, Fz_drag])) / self.P.M

        acc_known = acc_thrust_inertial + acc_gravity + acc_drag_inertial

        # 1.2 NDO 积分
        Lv = self.L_force @ v
        z_force_dot = -self.L_force @ self.z_force - self.L_force @ (acc_known + Lv)
        self.z_force += z_force_dot * dt
        self.d_force_hat = self.z_force + Lv

        # 平动 NDO 观测值限幅保护
        self.d_force_hat = np.clip(self.d_force_hat, -self.d_force_max, self.d_force_max)

        # ==========================================
        # 第二部分：力矩观测器
        # ==========================================
        omega = np.array([x.p, x.q, x.r])  # 机体系角速度

        # 2.1 获取当前转动惯量 J（随滑块位置变化）
        S = cal_sigma(self.P, x.chi, x.ups, x.chi_d, x.ups_d)
        I_obj = cal_inertia(self.P, S)
        J = np.array([
            [I_obj.Ixx, I_obj.Ixy, 0.0],
            [I_obj.Ixy, I_obj.Iyy, 0.0],
            [0.0, 0.0, I_obj.Izz]
        ]) + np.eye(3) * 1e-6

        # 尝试求逆（加保护，防止奇异）
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            J_inv = np.eye(3)

        # ==========================================
        # 根据转动惯量动态调整增益
        if self.enable_adaptive:
            # 基准转动惯量（使用基础机体惯量）
            J_base = np.array([
                [self.P.Ib_x, 0.0, 0.0],
                [0.0, self.P.Ib_y, 0.0],
                [0.0, 0.0, self.P.Ib_z]
            ])
            
            # 数值稳定性保护：设置 J_current 的最小值阈值（不低于 J_base 的 10%）
            J_min = J_base * 0.1
            J_current_safe = np.maximum(J, J_min)
            
            # 自适应增益计算：L_new = L_base * (J_base / J_current)
            # 原理：惯量增大时增益减小，保持观测器带宽一致
            # 使用对角矩阵简化计算，只更新对角元素
            L_new = np.zeros((3, 3))
            for i in range(3):
                L_new[i, i] = self.base_gain_torque * (J_base[i, i] / J_current_safe[i, i])
            
            # 更新转动 NDO 增益矩阵
            self.L_torque = L_new
            
            # 缓存增益调节因子用于调试
            self.gain_adaptation_factor = np.mean(np.diag(L_new) / self.base_gain_torque)

        # 2.2 计算已知力矩
        # 这里用当前滑块位置和总推力近似 MMC 的已知滚转/俯仰力矩，
        # 再叠加偏航控制力矩与角速度阻尼项。
        mu = self.P.mu
        # 加上偏航力矩
        M_known = np.array([
            -2.0 * mu * x.ups * u.thrust,  # Mx（滚转力矩）
            2.0 * mu * x.chi * u.thrust,  # My（俯仰力矩）
            u.tau_z  # Mz（偏航力矩）
        ])

        # 加上气动阻尼力矩（参考 update_6dof_model 中的 Md）
        Md = -np.array([
            self.P.kp * x.p,
            self.P.kq * x.q,
            self.P.kr * x.r
        ])

        Tau_total_known = M_known + Md

        # 2.3 计算已知角加速度
        # 刚体转动方程：J*alpha = Tau - w x Jw
        # 因此：alpha = J_inv * (Tau - w x Jw)
        gyroscopic = np.cross(omega, J @ omega)
        alpha_known = J_inv @ (Tau_total_known - gyroscopic)

        # 2.4 NDO 积分（结构与平动通道相同）
        # 观测器状态方程：dot_z = -L*z - L*(alpha_known + L*w)
        Lw = self.L_torque @ omega
        z_torque_dot = -self.L_torque @ self.z_torque - self.L_torque @ (alpha_known + Lw)

        self.z_torque += z_torque_dot * dt
        self.d_torque_hat = self.z_torque + Lw  # 这是干扰角加速度，单位 rad/s^2

        # 转动 NDO 观测值限幅保护
        self.d_torque_hat = np.clip(self.d_torque_hat, -self.d_torque_max, self.d_torque_max)

    def get_force_compensation(self, x: State6, current_thrust: float):
        """
        获取平动NDO的姿态角前馈补偿
        
        原理：将观测到的加速度干扰转换为需要的倾斜角
        使得推力分量能够抵消外部力干扰
        
        符号推导：
        - 风向东吹(wind_x > 0)，无人机被推向东，d_force_hat[0] > 0
        - 为了抵抗，无人机需要向西倾斜(theta < 0)
        - val_theta = -dx_head / acc_thrust，当dx_head > 0时，theta < 0 ✓
        
        返回: (comp_phi, comp_theta) 姿态角补偿量 [rad]
        """
        cpsi = math.cos(x.psi)
        spsi = math.sin(x.psi)

        dx_head = cpsi * self.d_force_hat[0] + spsi * self.d_force_hat[1]
        dy_head = -spsi * self.d_force_hat[0] + cpsi * self.d_force_hat[1]

        acc_thrust = current_thrust / self.P.M
        if acc_thrust < 1.0:
            acc_thrust = 9.81

        val_theta = -dx_head / acc_thrust
        # ENU 下正滚转对应负 Y 加速度，因此 Y 向补偿应与扰动同号：
        # 若扰动把机体往 +Y 推，则应给出正 phi 去产生 -Y 加速度；
        # 若扰动把机体往 -Y 推，则应给出负 phi 去产生 +Y 加速度。
        val_phi = dy_head / acc_thrust

        limit = math.sin(math.radians(25))
        val_theta = clamp(val_theta, -limit, limit)
        val_phi = clamp(val_phi, -limit, limit)

        theta_raw = math.asin(val_theta)
        phi_raw = math.asin(val_phi)

        comp_theta = self.lpf_alpha * theta_raw + (1 - self.lpf_alpha) * self.last_comp_theta_f
        comp_phi = self.lpf_alpha * phi_raw + (1 - self.lpf_alpha) * self.last_comp_phi_f

        self.last_comp_theta_f = comp_theta
        self.last_comp_phi_f = comp_phi

        return comp_phi, comp_theta

    def get_torque_compensation(self, x: State6, current_thrust: float):
        """
        获取转动NDO的姿态角补偿 (统一输出类型)

        设计原理：
        1. d_torque_hat 是观测到的角加速度干扰 [rad/s^2]
        2. 转换为力矩干扰: M_dist = J * d_torque_hat
        3. 将力矩干扰转换为姿态角补偿：
           - 外部力矩需要通过姿态调整来抵抗
           - 姿态调整会产生重力分量偏移，从而产生反向力矩
           - 同时姿态调整会改变推力方向，影响位置控制
        
        与平动NDO的区别：
        - 平动NDO：检测速度偏差 -> 输出姿态角补偿 -> 抵抗位置扰动
        - 转动NDO：检测角速度偏差 -> 输出姿态角补偿 -> 抵抗姿态扰动
        
        统一架构优势：
        - 两个NDO输出类型一致 (姿态角补偿)
        - 补偿信号叠加到MPC参考姿态角
        - MPC统一计算滑块位移，避免过补偿
        
        返回: (comp_phi_r, comp_theta_r) 姿态角补偿量 [rad]
        """
        S = cal_sigma(self.P, x.chi, x.ups, x.chi_d, x.ups_d)
        I_obj = cal_inertia(self.P, S)
        J = np.array([
            [I_obj.Ixx, I_obj.Ixy, 0.0],
            [I_obj.Ixy, I_obj.Iyy, 0.0],
            [0.0, 0.0, I_obj.Izz]
        ])
        
        M_dist = J @ self.d_torque_hat
        
        safe_thrust = max(current_thrust, 0.5 * self.P.M * self.P.g)
        acc_thrust = safe_thrust / self.P.M
        
        Mx_dist, My_dist, _ = M_dist[0], M_dist[1], M_dist[2]
        
        cpsi = math.cos(x.psi)
        spsi = math.sin(x.psi)
        
        Mx_head = cpsi * Mx_dist + spsi * My_dist
        My_head = -spsi * Mx_dist + cpsi * My_dist
        
        if acc_thrust < 1.0:
            acc_thrust = 9.81
        
        val_phi_r = -My_head / (self.P.M * acc_thrust * self.attitude_gain)
        val_theta_r = Mx_head / (self.P.M * acc_thrust * self.attitude_gain)
        
        max_angle = math.radians(15)
        val_phi_r = clamp(val_phi_r, -max_angle, max_angle)
        val_theta_r = clamp(val_theta_r, -max_angle, max_angle)
        
        comp_phi_r = self.lpf_alpha * val_phi_r + (1 - self.lpf_alpha) * self.last_comp_phi_r
        comp_theta_r = self.lpf_alpha * val_theta_r + (1 - self.lpf_alpha) * self.last_comp_theta_r
        
        self.last_comp_phi_r = comp_phi_r
        self.last_comp_theta_r = comp_theta_r
        
        return comp_phi_r, comp_theta_r

    def get_combined_compensation(self, x: State6, current_thrust: float):
        """
        [重构] 获取双NDO的联合姿态角补偿
        
        统一架构：
        - 平动NDO：输出姿态角补偿 (comp_phi_f, comp_theta_f)
        - 转动NDO：输出姿态角补偿 (comp_phi_r, comp_theta_r)
        - 两者叠加，统一由MPC计算滑块位移
        
        返回:
            (comp_phi_total, comp_theta_total): 总姿态角补偿 [rad]
            (comp_phi_f, comp_theta_f): 平动NDO分量 [rad]
            (comp_phi_r, comp_theta_r): 转动NDO分量 [rad]
        """
        comp_phi_f, comp_theta_f = self.get_force_compensation(x, current_thrust)
        if self.coupling_compensation:
            comp_phi_r, comp_theta_r = self.get_torque_compensation(x, current_thrust)
        else:
            comp_phi_r, comp_theta_r = 0.0, 0.0

        comp_phi_total = comp_phi_f + comp_phi_r
        comp_theta_total = comp_theta_f + comp_theta_r

        return (comp_phi_total, comp_theta_total), (comp_phi_f, comp_theta_f), (comp_phi_r, comp_theta_r)

    def get_disturbance_estimates(self):
        """
        获取当前的干扰估计值 (用于调试和可视化)
        
        返回:
            d_force: 平动加速度干扰估计 [m/s^2]
            d_torque: 转动角加速度干扰估计 [rad/s^2]
        """
        return self.d_force_hat.copy(), self.d_torque_hat.copy()

# ===================== ROS 2 与 Gazebo Harmonic 集成 =====================
class MMCUAVROS2Controller(Node):
    """
    [ROS2集成层]
    保持原有算法模块不变（外环PID、姿态MPC、NDO、混控器），
    （涉及规划器和动力学数学），并且仅用新的模拟驱动器替换了旧的RK4模拟驱动器
    ROS 2订阅 + 双速率定时器 + Gazebo主题发布。
    """

    def __init__(self):
        super().__init__("mmc_uav_controller")

        # ===== 保留的原始算法模块 =====
        self.P = Params()
        default_params = Params()
        self.P.m_b = max(
            1e-6,
            float(self.declare_parameter("body_mass_kg", self.P.m_b).value),
        )
        self.P.m = max(
            1e-6,
            float(self.declare_parameter("moving_mass_kg", self.P.m).value),
        )
        # 质量比扫描时，控制模型中的单个移动质量惯量随质量线性缩放。
        # 几何尺寸保持不变，因此这是对同一滑块外形填充密度变化的保守近似。
        self.P.Im_x = scaled_inertia(default_params.Im_x, self.P.m, default_params.m)
        self.P.Im_y = scaled_inertia(default_params.Im_y, self.P.m, default_params.m)
        self.P.Im_z = scaled_inertia(default_params.Im_z, self.P.m, default_params.m)
        self.P.wn_mass = max(
            1.0,
            float(self.declare_parameter("slider_wn_mass", self.P.wn_mass).value),
        )
        self.P.zeta_mass = max(
            0.05,
            float(self.declare_parameter("slider_zeta_mass", self.P.zeta_mass).value),
        )
        self.state = State6()
        self.outer = OuterPosController(self.P, dt_outer=1.0 / 25.0)
        outer_xy_kp = max(0.0, float(self.declare_parameter("outer_xy_kp", self.outer.kp_x).value))
        outer_xy_ki = max(0.0, float(self.declare_parameter("outer_xy_ki", self.outer.ki_x).value))
        outer_xy_kd = max(0.0, float(self.declare_parameter("outer_xy_kd", self.outer.kd_x).value))
        self.outer.kp_x = outer_xy_kp
        self.outer.kp_y = outer_xy_kp
        self.outer.ki_x = outer_xy_ki
        self.outer.ki_y = outer_xy_ki
        self.outer.kd_x = outer_xy_kd
        self.outer.kd_y = outer_xy_kd
        self.outer.terminal_hover_velocity_damping = max(
            0.0,
            float(
                self.declare_parameter(
                    "outer_terminal_hover_velocity_damping",
                    self.outer.terminal_hover_velocity_damping,
                ).value
            ),
        )
        self.att_mpc = AttitudeMPC(
            self.P,
            dt_mpc=1.0 / 100.0,
        )
        self.pitch_axis_coordinator = AxisManeuverCoordinator(
            AxisCoordinatorConfig.for_pitch(self.P, prediction_horizon=self.att_mpc.Np, dt=self.att_mpc.dt)
        )
        self.roll_axis_coordinator = AxisManeuverCoordinator(
            AxisCoordinatorConfig.for_roll(self.P, prediction_horizon=self.att_mpc.Np, dt=self.att_mpc.dt)
        )
        self.ndo_enabled = bool(self.declare_parameter("ndo_enabled", True).value)
        self.ndo_compensation_limit = math.radians(
            max(
                0.0,
                float(self.declare_parameter("ndo_compensation_limit_deg", 20.0).value),
            )
        )
        self.ndo_compensation_limit_schedule_enabled = bool(
            self.declare_parameter("ndo_compensation_limit_schedule_enabled", False).value
        )
        self.ndo_compensation_limit_low_speed = max(
            0.0,
            float(self.declare_parameter("ndo_compensation_limit_low_speed", 3.0).value),
        )
        self.ndo_compensation_limit_high_speed = max(
            0.0,
            float(self.declare_parameter("ndo_compensation_limit_high_speed", 5.0).value),
        )
        self.ndo_compensation_limit_low = math.radians(
            max(
                0.0,
                float(self.declare_parameter("ndo_compensation_limit_low_deg", 12.0).value),
            )
        )
        self.ndo_compensation_limit_high = math.radians(
            max(
                0.0,
                float(self.declare_parameter("ndo_compensation_limit_high_deg", 18.0).value),
            )
        )
        self.ndo_feedback_relief_enabled = bool(
            self.declare_parameter("ndo_feedback_relief_enabled", False).value
        )
        self.ndo_feedback_relief_gain = max(
            0.0,
            float(self.declare_parameter("ndo_feedback_relief_gain", 1.0).value),
        )
        self.ndo_feedback_relief_deadband = math.radians(
            max(
                0.0,
                float(self.declare_parameter("ndo_feedback_relief_deadband_deg", 1.5).value),
            )
        )
        self.ndo_feedback_relief_max_fraction = clamp(
            float(self.declare_parameter("ndo_feedback_relief_max_fraction", 0.65).value),
            0.0,
            0.95,
        )
        self.ndo_compensated_attitude_limit = math.radians(
            max(
                0.0,
                float(self.declare_parameter("ndo_compensated_attitude_limit_deg", 25.0).value),
            )
        )
        self.ndo_transient_attitude_boost_enabled = bool(
            self.declare_parameter("ndo_transient_attitude_boost_enabled", False).value
        )
        self.ndo_transient_attitude_limit = math.radians(
            max(
                0.0,
                float(self.declare_parameter("ndo_transient_attitude_limit_deg", 28.0).value),
            )
        )
        self.ndo_transient_attitude_boost_duration = max(
            0.0,
            float(self.declare_parameter("ndo_transient_attitude_boost_duration", 4.5).value),
        )
        self.ndo_transient_attitude_boost_fade = max(
            0.0,
            float(self.declare_parameter("ndo_transient_attitude_boost_fade", 1.0).value),
        )
        self.ndo = NDO_Observer(
            self.P,
            base_gain_force=max(
                0.0,
                float(self.declare_parameter("ndo_base_gain_force", 15.0).value),
            ),
            base_gain_torque=max(
                0.0,
                float(self.declare_parameter("ndo_base_gain_torque", 15.0).value),
            ),
            lpf_alpha=clamp(
                float(self.declare_parameter("ndo_lpf_alpha", 0.4).value),
                0.0,
                1.0,
            ),
            attitude_gain=max(
                1e-6,
                float(self.declare_parameter("ndo_attitude_gain", 10.0).value),
            ),
            enable_adaptive=bool(self.declare_parameter("ndo_adaptive_gain_enabled", True).value),
            coupling_compensation=bool(
                self.declare_parameter("ndo_coupling_compensation_enabled", True).value
            ),
        )
        self.mixer = CoaxialMixer(self.P)
        self.planner = UniversalTrajectoryPlanner()
        self.rotor_upper_force_constant = float(self.P.b_thrust)
        self.rotor_lower_force_constant = float(self.P.b_thrust * self.mixer.lower_rotor_eff)
        self.rotor_moment_constant = float(self.P.d_yaw / max(self.P.b_thrust, 1e-9))

        # ===== 回路频率：外环 25 Hz，内环 100 Hz =====
        self.outer_dt = 1.0 / 25.0
        self.inner_dt = 1.0 / 100.0

        # ===== 控制状态缓存 =====
        self.ref_pos_now = RefPos()
        self.thrust_cmd = self.P.M * self.P.g
        self.thrust_cmd_outer = self.thrust_cmd
        self.thrust_retarget_ratio = 1.0
        self.phi_ref = 0.0
        self.theta_ref = 0.0
        self.psi_ref = 0.0
        self.tau_z_cmd = 0.0
        self.chi_cmd = 0.0
        self.ups_cmd = 0.0
        self.raw_chi_cmd = 0.0
        self.raw_ups_cmd = 0.0
        self.actuation_backend = "rotor_physics"
        self.yaw_control_mode = "rotor_only"
        self.auto_scene_figure_eight_hover_start_time = None
        self.auto_scene_figure_eight_trigger_time = None
        self.auto_scene_figure_eight_logged = False
        self.auto_scene_figure_eight_base_psi = None
        self.auto_scene_figure_eight_last_yaw_ref = None
        self.auto_scene_figure_eight_last_yaw_ref_time = None
        self.auto_scene_figure_eight_entry_replanned = False
        self.auto_scene_figure_eight_entry_end_time = None
        self.auto_scene_figure_eight_phase_bias = 0.0
        self.auto_scene_figure_eight_phase_bias_locked = False
        self.rviz_actual_path_msg = None
        self.rviz_reference_path_msg = None
        self.rviz_reference_path_published = False
        self.rviz_vehicle_marker_array = None
        self.rviz_vehicle_marker_yaw = None
        self.auto_scene_yaw_step_hover_start_time = None
        self.auto_scene_yaw_step_base_psi = None
        self.auto_scene_yaw_step_target_psi = None
        self.auto_scene_yaw_step_trigger_time = None
        self.auto_scene_yaw_step_hold_x = None
        self.auto_scene_yaw_step_hold_y = None
        self.auto_scene_yaw_step_hold_z = None
        self.auto_scene_yaw_step_logged = False
        self.upper_rotor_cmd = 0.0
        self.lower_rotor_cmd = 0.0
        self.upper_rotor_actual = 0.0
        self.lower_rotor_actual = 0.0
        self.upper_rotor_cmd_target = 0.0
        self.lower_rotor_cmd_target = 0.0
        self.rotor_max_speed_rad_s = 450.0
        self.rotor_motor_time_constant_up = 0.0
        self.rotor_motor_time_constant_down = 0.0
        self.rotor_motor_rate_limit_rad_s2 = 0.0
        self.filtered_tau_z_cmd = 0.0
        self.rotor_total_thrust_est = 0.0
        self.rotor_tau_z_est = 0.0
        self.raw_phi_ref = 0.0
        self.raw_theta_ref = 0.0
        self.shaped_phi_ref = 0.0
        self.shaped_theta_ref = 0.0
        self.shaped_p_ref = 0.0
        self.shaped_q_ref = 0.0
        self.inner_phi_ref = 0.0
        self.inner_theta_ref = 0.0
        self.inner_psi_ref = 0.0
        self.coordinator_chi_ref = 0.0
        self.coordinator_ups_ref = 0.0
        self.last_u_mpc = np.zeros(3, dtype=float)
        self.slider_command_shaping_enabled = bool(
            self.declare_parameter("slider_command_shaping_enabled", False).value
        )
        slider_command_smoothing_time = float(self.declare_parameter("slider_command_smoothing_time", 0.06).value)
        slider_command_shape_tau = float(
            self.declare_parameter("slider_command_shape_tau", slider_command_smoothing_time).value
        )
        self.slider_command_shape_tau = max(0.0, slider_command_shape_tau)
        slider_command_rate_limit = float(self.declare_parameter("slider_command_rate_limit", 0.45).value)
        self.slider_command_rate_limit = max(0.0, slider_command_rate_limit)
        self.inner_thrust_retarget_enabled = bool(
            self.declare_parameter("inner_thrust_retarget_enabled", True).value
        )

        # ===== 传感器就绪状态 =====
        self.odom_ready = False
        self.imu_ready = False
        self.outer_ran_once = False
        self.mission_initialized = False
        self.preflight_centering_started = False
        self.preflight_centering_confirmed = False
        self.center_hold_start_ns: Optional[int] = None
        self.center_pos_tol = 0.003
        self.center_vel_tol = 0.03
        self.center_hold_time = 0.5
        self.preflight_centering_degraded_mode = False
        self.preflight_centering_degraded_logged = False

        # 可选的关节反馈时间戳
        self.last_joint_update_ns: Optional[int] = None

        # ===== 时间基准 =====
        self.start_time = self.get_clock().now()
        self.last_inner_tick = self.start_time
        self.wrench_dt_for_scale = self.inner_dt
        self.last_inner_dt = self.inner_dt
        self.last_inner_dt_raw = self.inner_dt
        self.last_inner_exec_dt = 0.0
        self.last_inner_mpc_dt = 0.0
        self.last_inner_observer_dt = 0.0
        self.last_inner_drag_dt = 0.0
        self.last_inner_publish_dt = 0.0
        self.last_inner_ref_build_dt = 0.0
        self.last_inner_log_dt = 0.0
        self.last_wrench_dt_for_scale = self.inner_dt
        self.last_wrench_scale = 0.0
        self.last_force_z_world_total = 0.0
        self.last_force_z_world_published = 0.0
        self.last_scene_hover_ready = False
        self.last_large_inner_dt_warn_time = -math.inf
        self._zero_ndo_log_row = (0.0,) * 12
        self._last_ndo_log_row = self._zero_ndo_log_row
        self._last_ndo_warning_time = -math.inf
        self._ndo_current_cycle_valid = False
        self._body_velocity_buffer = np.zeros(3, dtype=float)

        # ===== 话题参数（默认采用当前约定映射） =====
        self.imu_topic = self.declare_parameter("imu_topic", "/imu").value
        self.odom_topic = self.declare_parameter("odom_topic", "/model/mmc_uav/odometry").value
        self.joint_state_topic = self.declare_parameter("joint_state_topic", "/joint_states").value
        self.wrench_topic = self.declare_parameter("wrench_topic", "/world/mmc_world/wrench").value
        self.motor_speed_topic = self.declare_parameter("motor_speed_topic", "/mmc_uav/command/motor_speed").value
        self.rotor_lower_command_scale = max(
            0.0,
            float(self.declare_parameter("rotor_lower_command_scale", 1.0).value),
        )
        self.rotor_lower_command_scale_after_hover = max(
            0.0,
            float(
                self.declare_parameter(
                    "rotor_lower_command_scale_after_hover",
                    self.rotor_lower_command_scale,
                ).value
            ),
        )
        self.rotor_lower_command_scale_after_hover_hold_time = max(
            0.0,
            float(self.declare_parameter("rotor_lower_command_scale_after_hover_hold_time", 0.0).value),
        )
        self.rotor_lower_command_scale_hover_start_time: Optional[float] = None
        self.rotor_lower_command_scale_event_active = False
        self.rotor_lower_command_scale_event_logged = False
        self.actuation_backend = str(
            self.declare_parameter("actuation_backend", "rotor_physics").value
        ).strip().lower()
        if self.actuation_backend not in ("direct_wrench", "rotor_physics"):
            self.actuation_backend = "rotor_physics"
        self.yaw_control_mode = str(
            self.declare_parameter("yaw_control_mode", "rotor_only").value
        ).strip().lower()
        if self.yaw_control_mode not in ("rotor_only",):
            self.yaw_control_mode = "rotor_only"
        self.manual_xy_topic = self.declare_parameter("manual_xy_topic", "/mmc/manual_xy_cmd").value
        self.world_sdf_path = self.declare_parameter("world_sdf_path", "").value
        legacy_pitch_step_enabled = bool(self.declare_parameter("pitch_step_test_enabled", False).value)
        legacy_pitch_step_hover_hold_time = max(
            0.0,
            float(self.declare_parameter("pitch_step_hover_hold_time", 4.0).value),
        )
        legacy_pitch_step_theta_deg = float(self.declare_parameter("pitch_step_theta_deg", 20.0).value)
        legacy_pitch_step_hover_z_tol = max(
            0.0,
            float(self.declare_parameter("pitch_step_hover_z_tol", 0.15).value),
        )
        legacy_pitch_step_hover_speed_tol = max(
            0.0,
            float(self.declare_parameter("pitch_step_hover_speed_tol", 0.15).value),
        )

        self.attitude_step_test_enabled = bool(
            self.declare_parameter("attitude_step_test_enabled", legacy_pitch_step_enabled).value
        )
        self.attitude_step_test_axis = str(
            self.declare_parameter("attitude_step_test_axis", "pitch").value
        ).strip().lower()
        if self.attitude_step_test_axis not in ("pitch", "roll", "yaw"):
            self.attitude_step_test_axis = "pitch"
        self.attitude_step_hover_hold_time = max(
            0.0,
            float(self.declare_parameter("attitude_step_hover_hold_time", legacy_pitch_step_hover_hold_time).value),
        )
        self.attitude_step_hold_time = max(
            1e-6,
            float(self.declare_parameter("attitude_step_hold_time", 3.0).value),
        )
        self.attitude_step_recovery_time = max(
            1e-6,
            float(self.declare_parameter("attitude_step_recovery_time", 2.0).value),
        )
        self.attitude_step_test_angle_ref = math.radians(
            float(self.declare_parameter("attitude_step_angle_deg", legacy_pitch_step_theta_deg).value)
        )
        self.attitude_step_hover_z_tol = max(
            0.0,
            float(self.declare_parameter("attitude_step_hover_z_tol", legacy_pitch_step_hover_z_tol).value),
        )
        self.attitude_step_hover_speed_tol = max(
            0.0,
            float(self.declare_parameter("attitude_step_hover_speed_tol", legacy_pitch_step_hover_speed_tol).value),
        )
        self.auto_scene_mode = str(
            self.declare_parameter("auto_scene_mode", "hover_only").value
        ).strip().lower()
        if self.auto_scene_mode not in (
            "hover_only",
            "hover_to_point_hold",
            "hover_to_point_yaw_step_hold",
            "hover_to_yz_figure_eight",
            "hover_to_xz_figure_eight",
            "hover_to_yaw_step_hold",
            "hover_to_open_loop_rotor_diff",
        ):
            self.auto_scene_mode = "hover_only"
        self.auto_scene_takeoff_transition_time = max(
            1e-6,
            float(self.declare_parameter("auto_scene_takeoff_transition_time", 5.0).value),
        )
        self.auto_scene_hover_hold_time = max(
            0.0,
            float(self.declare_parameter("auto_scene_hover_hold_time", 4.0).value),
        )
        self.auto_scene_move_duration = max(
            1e-6,
            float(self.declare_parameter("auto_scene_move_duration", 4.0).value),
        )
        self.auto_scene_horizontal_accel_limit = max(
            1e-6,
            float(self.declare_parameter("auto_scene_horizontal_accel_limit", 0.8).value),
        )
        self.auto_scene_target_x = float(self.declare_parameter("auto_scene_target_x", 0.0).value)
        self.auto_scene_target_y = float(self.declare_parameter("auto_scene_target_y", 3.0).value)
        self.auto_scene_target_z = float(self.declare_parameter("auto_scene_target_z", 1.5).value)
        self.auto_scene_figure_eight_x_amplitude = max(
            0.0,
            float(self.declare_parameter("auto_scene_figure_eight_x_amplitude", 1.8).value),
        )
        self.auto_scene_figure_eight_y_amplitude = max(
            0.0,
            float(self.declare_parameter("auto_scene_figure_eight_y_amplitude", 1.8).value),
        )
        self.auto_scene_figure_eight_z_amplitude = max(
            0.0,
            float(self.declare_parameter("auto_scene_figure_eight_z_amplitude", 0.675).value),
        )
        self.auto_scene_figure_eight_forward_tilt_deg = clamp(
            float(self.declare_parameter("auto_scene_figure_eight_forward_tilt_deg", 0.0).value),
            -85.0,
            85.0,
        )
        self.auto_scene_figure_eight_period = max(
            1e-6,
            float(self.declare_parameter("auto_scene_figure_eight_period", 16.0).value),
        )
        self.auto_scene_figure_eight_ramp_duration = max(
            1e-6,
            float(self.declare_parameter("auto_scene_figure_eight_ramp_duration", 5.0).value),
        )
        self.auto_scene_figure_eight_entry_phase_ratio = clamp(
            float(self.declare_parameter("auto_scene_figure_eight_entry_phase_ratio", 1.0).value),
            0.20,
            1.0,
        )
        self.auto_scene_adaptive_phase_enabled = bool(
            self.declare_parameter("auto_scene_adaptive_phase_enabled", True).value
        )
        self.auto_scene_adaptive_phase_min_rate = clamp(
            float(self.declare_parameter("auto_scene_adaptive_phase_min_rate", 0.45).value),
            0.05,
            1.0,
        )
        self.auto_scene_adaptive_phase_filter_time_constant = max(
            0.0,
            float(self.declare_parameter("auto_scene_adaptive_phase_filter_time_constant", 0.15).value),
        )
        self.auto_scene_adaptive_phase_along_track_window = max(
            1e-3,
            float(self.declare_parameter("auto_scene_adaptive_phase_along_track_window", 1.00).value),
        )
        self.auto_scene_adaptive_phase_cross_track_window = max(
            1e-3,
            float(self.declare_parameter("auto_scene_adaptive_phase_cross_track_window", 0.70).value),
        )
        self.auto_scene_adaptive_phase_speed_floor = max(
            1e-3,
            float(self.declare_parameter("auto_scene_adaptive_phase_speed_floor", 0.25).value),
        )
        self.auto_scene_adaptive_phase_position_floor = max(
            1e-3,
            float(self.declare_parameter("auto_scene_adaptive_phase_position_floor", 0.25).value),
        )
        self.auto_scene_adaptive_phase_velocity_floor = max(
            1e-3,
            float(self.declare_parameter("auto_scene_adaptive_phase_velocity_floor", 0.25).value),
        )
        self.auto_scene_adaptive_phase_lag_weight = max(
            0.0,
            float(self.declare_parameter("auto_scene_adaptive_phase_lag_weight", 0.95).value),
        )
        self.auto_scene_adaptive_phase_cross_track_weight = max(
            0.0,
            float(self.declare_parameter("auto_scene_adaptive_phase_cross_track_weight", 1.15).value),
        )
        self.auto_scene_adaptive_phase_velocity_weight = max(
            0.0,
            float(self.declare_parameter("auto_scene_adaptive_phase_velocity_weight", 0.30).value),
        )
        self.auto_scene_adaptive_phase_projection_align_time_constant = max(
            0.0,
            float(
                self.declare_parameter(
                    "auto_scene_adaptive_phase_projection_align_time_constant",
                    0.30,
                ).value
            ),
        )
        self.auto_scene_adaptive_phase_projection_deadband = max(
            0.0,
            float(
                self.declare_parameter(
                    "auto_scene_adaptive_phase_projection_deadband",
                    0.02,
                ).value
            ),
        )
        self.auto_scene_adaptive_phase_projection_max_correction = max(
            0.0,
            float(
                self.declare_parameter(
                    "auto_scene_adaptive_phase_projection_max_correction",
                    0.10,
                ).value
            ),
        )
        self.auto_scene_yaw_ref_mode = str(
            self.declare_parameter("auto_scene_yaw_ref_mode", "fixed").value
        ).strip().lower()
        if self.auto_scene_yaw_ref_mode not in ("fixed", "path_tangent_xy"):
            self.auto_scene_yaw_ref_mode = "fixed"
        self.auto_scene_yaw_ref_speed_floor = max(
            1e-6,
            float(self.declare_parameter("auto_scene_yaw_ref_speed_floor", 0.05).value),
        )
        self.auto_scene_yaw_ref_rate_limit_rad_s = math.radians(
            max(
                0.0,
                float(
                    self.declare_parameter(
                        "auto_scene_yaw_ref_rate_limit_deg_s",
                        25.0,
                    ).value
                ),
            )
        )
        self.rviz_trajectory_enabled = bool(
            self.declare_parameter("rviz_trajectory_enabled", True).value
        )
        self.rviz_trajectory_frame_id = str(
            self.declare_parameter("rviz_trajectory_frame_id", "mmc_world").value
        ).strip() or "mmc_world"
        self.rviz_actual_path_topic = str(
            self.declare_parameter("rviz_actual_path_topic", "/mmc/trajectory/actual").value
        ).strip() or "/mmc/trajectory/actual"
        self.rviz_reference_path_topic = str(
            self.declare_parameter("rviz_reference_path_topic", "/mmc/trajectory/reference").value
        ).strip() or "/mmc/trajectory/reference"
        self.rviz_actual_path_max_points = max(
            1,
            int(self.declare_parameter("rviz_actual_path_max_points", 5000).value),
        )
        self.rviz_reference_path_dt = clamp(
            float(self.declare_parameter("rviz_reference_path_dt", 0.05).value),
            0.01,
            0.50,
        )
        self.rviz_reference_path_cycles = max(
            0.25,
            float(self.declare_parameter("rviz_reference_path_cycles", 1.0).value),
        )
        self.rviz_vehicle_marker_enabled = bool(
            self.declare_parameter("rviz_vehicle_marker_enabled", True).value
        )
        self.rviz_vehicle_marker_topic = str(
            self.declare_parameter("rviz_vehicle_marker_topic", "/mmc/vehicle_marker").value
        ).strip() or "/mmc/vehicle_marker"
        self.rviz_vehicle_sphere_diameter = clamp(
            float(self.declare_parameter("rviz_vehicle_sphere_diameter", 0.18).value),
            0.02,
            2.0,
        )
        self.rviz_vehicle_arrow_length = clamp(
            float(self.declare_parameter("rviz_vehicle_arrow_length", 0.32).value),
            0.05,
            5.0,
        )
        self.rviz_vehicle_arrow_shaft_diameter = clamp(
            float(self.declare_parameter("rviz_vehicle_arrow_shaft_diameter", 0.035).value),
            0.005,
            1.0,
        )
        self.rviz_vehicle_arrow_head_diameter = clamp(
            float(self.declare_parameter("rviz_vehicle_arrow_head_diameter", 0.08).value),
            0.01,
            1.5,
        )
        self.rviz_vehicle_arrow_z_offset = clamp(
            float(self.declare_parameter("rviz_vehicle_arrow_z_offset", 0.03).value),
            -1.0,
            1.0,
        )
        self.auto_scene_yaw_step_deg = float(self.declare_parameter("auto_scene_yaw_step_deg", 90.0).value)
        self.auto_scene_yaw_ramp_duration = max(
            1e-6,
            float(self.declare_parameter("auto_scene_yaw_ramp_duration", 4.0).value),
        )
        self.auto_scene_open_loop_tau_z_step = float(
            self.declare_parameter("auto_scene_open_loop_tau_z_step", -0.03).value
        )
        self.rotor_min_speed_ratio = clamp(
            float(self.declare_parameter("rotor_min_speed_ratio", 0.20).value),
            0.0,
            0.95,
        )
        self.rotor_tau_z_filter_time_constant = max(
            0.0,
            float(self.declare_parameter("rotor_tau_z_filter_time_constant", 0.12).value),
        )
        self.rotor_max_speed_rad_s = max(
            1.0,
            float(self.declare_parameter("rotor_max_speed_rad_s", 450.0).value),
        )
        self.rotor_motor_time_constant_up = max(
            0.0,
            float(self.declare_parameter("rotor_motor_time_constant_up", 0.0).value),
        )
        self.rotor_motor_time_constant_down = max(
            0.0,
            float(self.declare_parameter("rotor_motor_time_constant_down", 0.0).value),
        )
        self.rotor_motor_rate_limit_rad_s2 = max(
            0.0,
            float(self.declare_parameter("rotor_motor_rate_limit_rad_s2", 0.0).value),
        )
        self.mixer.set_min_speed_ratio(self.rotor_min_speed_ratio)
        self.auto_scene_open_loop_rotor_hover_start_time: Optional[float] = None
        self.auto_scene_open_loop_rotor_active = False
        self.auto_scene_open_loop_rotor_logged = False
        self.auto_scene_open_loop_w1_sq = 0.0
        self.auto_scene_open_loop_w2_sq = 0.0
        self.auto_scene_open_loop_thrust_real = 0.0
        self.auto_scene_open_loop_tau_z_real = 0.0
        self.auto_scene_adaptive_phase_cfg = AdaptivePhaseScheduleConfig(
            enabled=bool(self.auto_scene_adaptive_phase_enabled),
            min_rate=float(self.auto_scene_adaptive_phase_min_rate),
            filter_time_constant=float(self.auto_scene_adaptive_phase_filter_time_constant),
            along_track_window=float(self.auto_scene_adaptive_phase_along_track_window),
            cross_track_window=float(self.auto_scene_adaptive_phase_cross_track_window),
            speed_floor=float(self.auto_scene_adaptive_phase_speed_floor),
            position_floor=float(self.auto_scene_adaptive_phase_position_floor),
            velocity_floor=float(self.auto_scene_adaptive_phase_velocity_floor),
            lag_weight=float(self.auto_scene_adaptive_phase_lag_weight),
            cross_track_weight=float(self.auto_scene_adaptive_phase_cross_track_weight),
            velocity_weight=float(self.auto_scene_adaptive_phase_velocity_weight),
            projection_align_time_constant=float(
                self.auto_scene_adaptive_phase_projection_align_time_constant
            ),
            projection_deadband=float(self.auto_scene_adaptive_phase_projection_deadband),
            projection_max_correction=float(self.auto_scene_adaptive_phase_projection_max_correction),
        )
        self.auto_scene_adaptive_phase_state = AdaptivePhaseScheduleState()
        self.last_adaptive_phase_active = False
        self.last_adaptive_phase_time = 0.0
        self.last_adaptive_phase_rate = 1.0
        self.last_adaptive_phase_metric = 0.0
        self.manual_xy_enabled = bool(self.declare_parameter("manual_xy_enabled", False).value)
        manual_xy_requested = self.manual_xy_enabled
        self.manual_xy_enabled = manual_xy_requested and not self.attitude_step_test_enabled
        self.attitude_step_hover_start_time: Optional[float] = None
        self.attitude_step_test_start_time: Optional[float] = None
        self.attitude_step_test_active = False
        self.attitude_step_test_logged = False
        self.attitude_step_test_roll_start = 0.0
        self.attitude_step_test_pitch_start = 0.0

        # Legacy fields are kept as live aliases and synchronized by runtime helpers.
        self.pitch_step_test_enabled = self.attitude_step_test_enabled
        self.pitch_step_hover_hold_time = self.attitude_step_hover_hold_time
        self.pitch_step_test_theta_ref = self.attitude_step_test_angle_ref
        self.pitch_step_hover_z_tol = self.attitude_step_hover_z_tol
        self.pitch_step_hover_speed_tol = self.attitude_step_hover_speed_tol
        self.pitch_step_hover_start_time = self.attitude_step_hover_start_time
        self.pitch_step_test_start_time = self.attitude_step_test_start_time
        self.pitch_step_test_active = self.attitude_step_test_active
        self.pitch_step_test_logged = self.attitude_step_test_logged
        self.pitch_step_test_theta_start = self.attitude_step_test_pitch_start
        self._pitch_step_legacy_shadow = {
            "pitch_step_test_enabled": self.pitch_step_test_enabled,
            "pitch_step_hover_hold_time": self.pitch_step_hover_hold_time,
            "pitch_step_test_theta_ref": self.pitch_step_test_theta_ref,
            "pitch_step_hover_z_tol": self.pitch_step_hover_z_tol,
            "pitch_step_hover_speed_tol": self.pitch_step_hover_speed_tol,
            "pitch_step_hover_start_time": self.pitch_step_hover_start_time,
            "pitch_step_test_start_time": self.pitch_step_test_start_time,
            "pitch_step_test_active": self.pitch_step_test_active,
            "pitch_step_test_logged": self.pitch_step_test_logged,
            "pitch_step_test_theta_start": self.pitch_step_test_theta_start,
        }

        self.upper_rotor_topic = self.declare_parameter(
            "upper_rotor_topic", "/model/mmc_uav/joint/joint_rotor_upper/cmd_vel"
        ).value
        self.lower_rotor_topic = self.declare_parameter(
            "lower_rotor_topic", "/model/mmc_uav/joint/joint_rotor_lower/cmd_vel"
        ).value

        self.slider_green_topic = self.declare_parameter(
            "slider_green_topic", "/model/mmc_uav/joint/joint_slider_green/cmd_pos"
        ).value
        self.slider_purple_topic = self.declare_parameter(
            "slider_purple_topic", "/model/mmc_uav/joint/joint_slider_purple/cmd_pos"
        ).value
        self.slider_blue_topic = self.declare_parameter(
            "slider_blue_topic", "/model/mmc_uav/joint/joint_slider_blue/cmd_pos"
        ).value
        self.slider_red_topic = self.declare_parameter(
            "slider_red_topic", "/model/mmc_uav/joint/joint_slider_red/cmd_pos"
        ).value

        self.manual_xy = ManualXYController(
            unlock_altitude_tol=self.declare_parameter("manual_unlock_altitude_tol", 0.30).value,
            unlock_hold_time=self.declare_parameter("manual_unlock_hold_time", 3.0).value,
            max_tilt_deg=self.declare_parameter("manual_max_tilt_deg", 12.0).value,
            max_speed=self.declare_parameter("manual_max_speed", 1.5).value,
            speed_rise_time=self.declare_parameter("manual_speed_rise_time", 0.45).value,
            speed_fall_time=self.declare_parameter("manual_speed_fall_time", 0.30).value,
            brake_max_tilt_deg=self.declare_parameter("manual_auto_brake_max_tilt_deg", 15.0).value,
            brake_full_speed=self.declare_parameter("manual_auto_brake_full_speed", 2.7).value,
            brake_rise_time=self.declare_parameter("manual_auto_brake_rise_time", 0.75).value,
            brake_stop_speed_threshold=self.declare_parameter("manual_auto_brake_stop_speed_threshold", 0.08).value,
            brake_capture_speed=self.declare_parameter("manual_auto_brake_capture_speed", 0.25).value,
            brake_relock_dwell=self.declare_parameter("manual_auto_brake_relock_dwell", 0.35).value,
            stop_tilt_deg=self.declare_parameter("manual_stop_tilt_deg", 0.9).value,
            cmd_timeout=self.declare_parameter("manual_cmd_timeout", 0.20).value,
        )
        self.manual_tilt_thrust_margin = float(
            self.declare_parameter("manual_tilt_thrust_margin", 0.08).value
        )
        self.outer.z_thrust_scale = clamp(
            float(self.declare_parameter("outer_z_thrust_scale", 0.977).value),
            0.90,
            1.05,
        )
        self.last_manual_ready = self.manual_xy.ready
        self.last_manual_mode = self.manual_xy.mode
        self.manual_status = ManualXYStatus()
        if self.attitude_step_test_enabled:
            self.get_logger().info(
                "Attitude-step test mode enabled. Manual XY input is bypassed; "
                f"axis={self.attitude_step_test_axis}, "
                f"hover-hold {self.attitude_step_hover_hold_time:.1f}s -> "
                f"+/-{math.degrees(self.attitude_step_test_angle_ref):.1f} deg "
                f"hold {self.attitude_step_hold_time:.1f}s / recover {self.attitude_step_recovery_time:.1f}s cycle."
            )

        # ===== 必需订阅 =====
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(WindStatus, "/mmc/wind/status", self.wind_status_cb, 10)
        if self.manual_xy_enabled:
            self.create_subscription(Twist, self.manual_xy_topic, self.manual_xy_cmd_cb, 30)

        # ===== 可选关节状态反馈（缺失时允许估计降级放行） =====
        self.create_subscription(JointState, self.joint_state_topic, self.joint_state_cb, 30)

        # ===== 发布器 =====
        self.pub_wrench = self.create_publisher(EntityWrench, self.wrench_topic, 30)
        self.pub_motor_speed = self.create_publisher(Actuators, self.motor_speed_topic, 30)
        self.pub_upper_rotor = self.create_publisher(Float64, self.upper_rotor_topic, 30)
        self.pub_lower_rotor = self.create_publisher(Float64, self.lower_rotor_topic, 30)

        self.pub_slider_green = self.create_publisher(Float64, self.slider_green_topic, 30)
        self.pub_slider_purple = self.create_publisher(Float64, self.slider_purple_topic, 30)
        self.pub_slider_blue = self.create_publisher(Float64, self.slider_blue_topic, 30)
        self.pub_slider_red = self.create_publisher(Float64, self.slider_red_topic, 30)

        self.pub_euler_deg = self.create_publisher(Vector3, '/model/mmc_uav/euler_degrees', 30)
        self.wind_command_pub = self.create_publisher(WindCommand, "/mmc/wind/command", 10)
        self.pub_actual_path = None
        self.pub_reference_path = None
        self.pub_vehicle_marker = None
        if self.rviz_trajectory_enabled:
            self.pub_actual_path = self.create_publisher(Path, self.rviz_actual_path_topic, 10)
            self.pub_reference_path = self.create_publisher(Path, self.rviz_reference_path_topic, 10)
        if self.rviz_vehicle_marker_enabled:
            self.pub_vehicle_marker = self.create_publisher(MarkerArray, self.rviz_vehicle_marker_topic, 10)
        self._cached_wrench_msg = None
        self._cached_motor_speed_msg = None
        self._cached_upper_rotor_msg = None
        self._cached_lower_rotor_msg = None
        self._cached_slider_green_msg = None
        self._cached_slider_purple_msg = None
        self._cached_slider_blue_msg = None
        self._cached_slider_red_msg = None
        self._cached_euler_msg = None

        # ===== 双频定时器 =====
        self.create_timer(self.outer_dt, self.outer_loop_cb)
        self.create_timer(self.inner_dt, self.inner_loop_cb)

        self.get_logger().info(
            f"MMC controller node started. Waiting for IMU/Odometry... "
            f"attitude_mpc_prediction_horizon={self.att_mpc.Np}, "
            f"attitude_mpc_control_horizon={self.att_mpc.Np}"
        )
        self.get_logger().info(
            f"Actuation backend: {self.actuation_backend}, "
            f"yaw_control_mode={self.yaw_control_mode}, "
            f"motor_speed_topic={self.motor_speed_topic}, "
            f"m_b={self.P.m_b:.3f}kg, m={self.P.m:.3f}kg, M={self.P.M:.3f}kg, "
            f"mu={self.P.mu:.5f}, "
            f"rotor_lower_command_scale={self.rotor_lower_command_scale:.3f}, "
            f"rotor_lower_command_scale_after_hover={self.rotor_lower_command_scale_after_hover:.3f}, "
            f"rotor_lower_command_scale_after_hover_hold_time="
            f"{self.rotor_lower_command_scale_after_hover_hold_time:.3f}s"
        )
        self.get_logger().info(
            "Rotor command dynamics: "
            f"max_speed={self.rotor_max_speed_rad_s:.1f}rad/s, "
            f"tau_up={self.rotor_motor_time_constant_up:.3f}s, "
            f"tau_down={self.rotor_motor_time_constant_down:.3f}s, "
            f"rate_limit={self.rotor_motor_rate_limit_rad_s2:.1f}rad/s^2, "
            f"tau_z_filter={self.rotor_tau_z_filter_time_constant:.3f}s"
        )
        if (
            abs(self.rotor_lower_command_scale_after_hover - self.rotor_lower_command_scale) > 1e-9
            and self.rotor_lower_command_scale_after_hover_hold_time > 1e-9
        ):
            self.get_logger().warning(
                "RotorPhysics validation override armed: "
                f"after {self.rotor_lower_command_scale_after_hover_hold_time:.2f}s of stable hover, "
                f"final lower-rotor speed command scale will switch "
                f"from {self.rotor_lower_command_scale:.3f} "
                f"to {self.rotor_lower_command_scale_after_hover:.3f}."
            )
        elif abs(self.rotor_lower_command_scale - 1.0) > 1e-9:
            self.get_logger().warning(
                "RotorPhysics validation override is active immediately: "
                f"final lower-rotor speed command is scaled by {self.rotor_lower_command_scale:.3f}."
            )
        if not self.manual_xy_enabled:
            self.get_logger().info("Manual XY handoff is disabled for the current auto-scene baseline.")

        # 飞行数据记录
        self.flight_data_log = []
        self.start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wind_config_summary: WindConfigSummary = parse_world_wind_config(self.world_sdf_path)
        self.wind_log_row = self.wind_config_summary.as_log_row()
        self.wind_world_name = self.wind_config_summary.world_name or world_name_from_topic(self.wrench_topic) or "mmc_world"
        self.wind_topic = f"/world/{self.wind_world_name}/wind/"
        self.wind_command_seq = 0
        self.wind_command_source = ""
        self.wind_command_pending = False
        self.wind_status_publish_ok = False
        self.wind_status_detail = ""
        self.wind_runtime_active = bool(
            self.wind_config_summary.enable_wind and self.wind_config_summary.activation_mode == "immediate"
        )
        self.wind_activation_time = 0.0 if self.wind_runtime_active else math.nan
        self.wind_hover_hold_start_time: Optional[float] = None
        self.wind_move_window_start_requested = bool(self.wind_runtime_active)
        self.wind_move_window_stop_requested = False

        if self.wind_config_summary.config_valid:
            self.get_logger().info(
                "Wind config cached from world_sdf_path: "
                f"{self.world_sdf_path} -> "
                f"v=({self.wind_config_summary.wind_vx_world:.2f}, "
                f"{self.wind_config_summary.wind_vy_world:.2f}, "
                f"{self.wind_config_summary.wind_vz_world:.2f}) m/s, "
                f"mode={self.wind_config_summary.activation_mode}, "
                f"wind_topic={self.wind_topic}"
            )
        else:
            self.get_logger().warning(
                "Wind config could not be parsed from world_sdf_path. "
                "Blackbox wind summary will fall back to disabled zero-wind values."
            )

    # ----------------------------
    # 传感器回调
    # ----------------------------
    def odom_cb(self, msg: Odometry):
        with optional_lock(self):
            # Gazebo / Odometry 的 pose 在世界系，但 twist 默认跟随 child_frame_id，
            # 即这里的线速度通常是机体系速度。固定偏航场景下这个问题不明显，
            # 一旦边飞边转头，就会把 body-frame 速度误当成 world-frame 速度，
            # 直接把 XY 位置环、刹车逻辑和终端捕获全部带歪，表现成弧线入场和后段打圈。
            self.state.x = msg.pose.pose.position.x
            self.state.y = msg.pose.pose.position.y
            self.state.z = msg.pose.pose.position.z
            q = msg.pose.pose.orientation
            _, _, self.rviz_vehicle_marker_yaw = quaternion_to_euler_enu(q.x, q.y, q.z, q.w)
            self.state.vx, self.state.vy, self.state.vz = body_velocity_to_world_enu(
                q.x,
                q.y,
                q.z,
                q.w,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            )

            self.odom_ready = True
            self._try_init_mission()

    def imu_cb(self, msg: Imu):
        with optional_lock(self):
            # 四元数转欧拉角（phi、theta、psi）
            q = msg.orientation
            phi, theta, psi = quaternion_to_euler_enu(q.x, q.y, q.z, q.w)
            self.state.phi = phi
            self.state.theta = theta
            self.state.psi = psi

            # 机体系角速度（p、q、r）
            self.state.p = msg.angular_velocity.x
            self.state.q = msg.angular_velocity.y
            self.state.r = msg.angular_velocity.z

            self.imu_ready = True
            self._try_init_mission()

    def wind_status_cb(self, msg: WindStatus):
        with optional_lock(self):
            self.wind_status_publish_ok = bool(msg.publish_ok)
            self.wind_status_detail = str(msg.detail)

            if int(msg.command_seq) != int(getattr(self, "wind_command_seq", 0)):
                return

            self.wind_command_pending = False
            publish_ok = bool(msg.publish_ok)
            status_wind_active = bool(msg.wind_active)
            was_wind_active = bool(getattr(self, "wind_runtime_active", False))
            if publish_ok:
                self.wind_runtime_active = status_wind_active
            if publish_ok and self.wind_runtime_active:
                if not was_wind_active or not math.isfinite(float(getattr(self, "wind_activation_time", math.nan))):
                    self.wind_activation_time = self._elapsed_sec()
                self.get_logger().info(
                    "Wind bridge acknowledged runtime wind command: "
                    f"seq={msg.command_seq}, source={msg.source}, detail={msg.detail}"
                )
                return

            if publish_ok:
                self.get_logger().info(
                    "Wind bridge acknowledged runtime wind deactivation command: "
                    f"seq={msg.command_seq}, source={msg.source}, detail={msg.detail}"
                )
                return

            if str(getattr(msg, "source", "")) == "move_window_start":
                self.wind_move_window_start_requested = False
            elif str(getattr(msg, "source", "")) == "move_window_stop":
                self.wind_move_window_stop_requested = False

            self.get_logger().warning(
                "Wind bridge rejected runtime wind command: "
                f"seq={msg.command_seq}, source={msg.source}, detail={msg.detail}"
            )

    def manual_xy_cmd_cb(self, msg: Twist):
        with optional_lock(self):
            self.manual_xy.set_command(
                forward=msg.linear.x,
                lateral=msg.linear.y,
                now_sec=self._now_sec(),
            )

    def joint_state_cb(self, msg: JointState):
        # 可选：在有真实关节状态反馈时，用其更新 chi/ups
        if not msg.name:
            return

        with optional_lock(self):
            name_to_idx = {name: i for i, name in enumerate(msg.name)}

            def fetch_joint(joint_name):
                idx = name_to_idx.get(joint_name)
                if idx is None:
                    return None, None
                pos = msg.position[idx] if idx < len(msg.position) else None
                vel = msg.velocity[idx] if idx < len(msg.velocity) else None
                return pos, vel

            g_pos, g_vel = fetch_joint("joint_slider_green")
            p_pos, p_vel = fetch_joint("joint_slider_purple")
            b_pos, b_vel = fetch_joint("joint_slider_blue")
            r_pos, r_vel = fetch_joint("joint_slider_red")
            _upper_pos, upper_vel = fetch_joint("joint_rotor_upper")
            _lower_pos, lower_vel = fetch_joint("joint_rotor_lower")

            # X 轴两侧滑块：
            # 在 URDF 中的轴向均为 (-1, 0, 0)。
            # 指令映射对两个滑块都发布 -chi_cmd。
            # 因此 chi 由关节位置负平均值重建。
            if g_pos is not None and p_pos is not None:
                self.state.chi = -0.5 * (g_pos + p_pos)
            elif g_pos is not None:
                self.state.chi = -g_pos
            elif p_pos is not None:
                self.state.chi = -p_pos

            if g_vel is not None and p_vel is not None:
                self.state.chi_d = -0.5 * (g_vel + p_vel)
            elif g_vel is not None:
                self.state.chi_d = -g_vel
            elif p_vel is not None:
                self.state.chi_d = -p_vel

            # Y 轴两侧滑块：
            # 在 URDF 中的轴向均为 (0, 1, 0)。
            # 内部 ups 与关节坐标保持同号，避免反馈重建与真实滚转效应错位。
            if b_pos is not None and r_pos is not None:
                self.state.ups = 0.5 * (b_pos + r_pos)
            elif b_pos is not None:
                self.state.ups = b_pos
            elif r_pos is not None:
                self.state.ups = r_pos

            if b_vel is not None and r_vel is not None:
                self.state.ups_d = 0.5 * (b_vel + r_vel)
            elif b_vel is not None:
                self.state.ups_d = b_vel
            elif r_vel is not None:
                self.state.ups_d = r_vel

            params = getattr(self, "P", P)
            self.state.chi, self.state.chi_d = clamp_slider_state(
                self.state.chi,
                self.state.chi_d,
                params.u2_lim,
                params.slider_vel_max,
            )
            self.state.ups, self.state.ups_d = clamp_slider_state(
                self.state.ups,
                self.state.ups_d,
                params.u2_lim,
                params.slider_vel_max,
            )

            if upper_vel is not None:
                self.upper_rotor_actual = float(upper_vel)
            if lower_vel is not None:
                self.lower_rotor_actual = float(lower_vel)

            self.last_joint_update_ns = self.get_clock().now().nanoseconds

    # ----------------------------
    # 任务与辅助方法
    # ----------------------------
    def _elapsed_sec(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _sensors_ready(self) -> bool:
        with optional_lock(self):
            return self.imu_ready and self.odom_ready

    def _resolved_auto_scene_move_duration(self) -> float:
        requested_duration = max(
            float(getattr(self, "auto_scene_move_duration", 4.0)),
            1e-6,
        )
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if scene_mode not in ("hover_to_point_hold", "hover_to_point_yaw_step_hold"):
            return requested_duration

        target_x = abs(float(getattr(self, "auto_scene_target_x", 0.0)))
        target_y = abs(float(getattr(self, "auto_scene_target_y", 3.0)))
        accel_limit = max(
            float(getattr(self, "auto_scene_horizontal_accel_limit", 0.8)),
            1e-6,
        )

        min_duration_x = math.sqrt(QUINTIC_BLEND_MAX_ABS_ACCEL * target_x / accel_limit) if target_x > 1e-9 else 0.0
        min_duration_y = math.sqrt(QUINTIC_BLEND_MAX_ABS_ACCEL * target_y / accel_limit) if target_y > 1e-9 else 0.0

        resolved_duration = max(requested_duration, min_duration_x, min_duration_y)
        if scene_mode == "hover_to_point_yaw_step_hold":
            yaw_ramp_duration = max(
                float(getattr(self, "auto_scene_yaw_ramp_duration", 4.0)),
                1e-6,
            )
            # 在“平移 + 同步偏航”的耦合场景里，若平移比偏航更早完成，
            # 飞机会在已经冻结到目标点之后继续转头，进而把本应接近直线的
            # 入场轨迹拖成弧线，并给后段打圈埋下初始侧偏。
            # 因此这里强制让平移相位不短于偏航相位；若用户反过来把平移设得更长，
            # 偏航也会在后续通过同一 resolved_duration 与之保持同步结束。
            resolved_duration = max(resolved_duration, yaw_ramp_duration)

        return resolved_duration

    def _adaptive_phase_config(self) -> AdaptivePhaseScheduleConfig:
        cfg = getattr(self, "auto_scene_adaptive_phase_cfg", None)
        if isinstance(cfg, AdaptivePhaseScheduleConfig):
            return cfg
        cfg = AdaptivePhaseScheduleConfig(
            enabled=bool(getattr(self, "auto_scene_adaptive_phase_enabled", False)),
            min_rate=clamp(float(getattr(self, "auto_scene_adaptive_phase_min_rate", 0.45)), 0.05, 1.0),
            filter_time_constant=max(
                0.0,
                float(getattr(self, "auto_scene_adaptive_phase_filter_time_constant", 0.15)),
            ),
            along_track_window=max(
                1e-3,
                float(getattr(self, "auto_scene_adaptive_phase_along_track_window", 1.00)),
            ),
            cross_track_window=max(
                1e-3,
                float(getattr(self, "auto_scene_adaptive_phase_cross_track_window", 0.70)),
            ),
            speed_floor=max(
                1e-3,
                float(getattr(self, "auto_scene_adaptive_phase_speed_floor", 0.25)),
            ),
            position_floor=max(
                1e-3,
                float(getattr(self, "auto_scene_adaptive_phase_position_floor", 0.25)),
            ),
            velocity_floor=max(
                1e-3,
                float(getattr(self, "auto_scene_adaptive_phase_velocity_floor", 0.25)),
            ),
            lag_weight=max(0.0, float(getattr(self, "auto_scene_adaptive_phase_lag_weight", 0.95))),
            cross_track_weight=max(
                0.0,
                float(getattr(self, "auto_scene_adaptive_phase_cross_track_weight", 1.15)),
            ),
            velocity_weight=max(
                0.0,
                float(getattr(self, "auto_scene_adaptive_phase_velocity_weight", 0.30)),
            ),
            projection_align_time_constant=max(
                0.0,
                float(
                    getattr(
                        self,
                        "auto_scene_adaptive_phase_projection_align_time_constant",
                        0.30,
                    )
                ),
            ),
            projection_deadband=max(
                0.0,
                float(getattr(self, "auto_scene_adaptive_phase_projection_deadband", 0.02)),
            ),
            projection_max_correction=max(
                0.0,
                float(
                    getattr(
                        self,
                        "auto_scene_adaptive_phase_projection_max_correction",
                        0.10,
                    )
                ),
            ),
        )
        setattr(self, "auto_scene_adaptive_phase_cfg", cfg)
        return cfg

    def _ensure_auto_scene_adaptive_phase_state(self) -> AdaptivePhaseScheduleState:
        state = getattr(self, "auto_scene_adaptive_phase_state", None)
        if isinstance(state, AdaptivePhaseScheduleState):
            return state
        state = AdaptivePhaseScheduleState()
        setattr(self, "auto_scene_adaptive_phase_state", state)
        return state

    def _reset_auto_scene_adaptive_phase_state(self):
        state = AdaptivePhaseScheduleState()
        setattr(self, "auto_scene_adaptive_phase_state", state)
        self.last_adaptive_phase_active = False
        self.last_adaptive_phase_time = 0.0
        self.last_adaptive_phase_rate = 1.0
        self.last_adaptive_phase_metric = 0.0

    @staticmethod
    def _is_auto_scene_figure_eight_mode(scene_mode: str) -> bool:
        return str(scene_mode).strip().lower() in ("hover_to_yz_figure_eight", "hover_to_xz_figure_eight")

    @staticmethod
    def _auto_scene_figure_eight_lateral_axis(scene_mode: str) -> Optional[str]:
        normalized = str(scene_mode).strip().lower()
        if normalized == "hover_to_yz_figure_eight":
            return "y"
        if normalized == "hover_to_xz_figure_eight":
            return "x"
        return None

    @staticmethod
    def _auto_scene_figure_eight_plane_label(scene_mode: str) -> str:
        lateral_axis = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_axis(scene_mode)
        if lateral_axis == "x":
            return "XZ"
        return "YZ"

    @staticmethod
    def _auto_scene_figure_eight_forward_tilt_deg(self, scene_mode: Optional[str] = None) -> float:
        normalized = str(
            getattr(self, "auto_scene_mode", "hover_only") if scene_mode is None else scene_mode
        ).strip().lower()
        if normalized != "hover_to_yz_figure_eight":
            return 0.0
        return clamp(
            float(getattr(self, "auto_scene_figure_eight_forward_tilt_deg", 0.0)),
            -85.0,
            85.0,
        )

    @staticmethod
    def _auto_scene_yaw_ref_mode(self) -> str:
        mode = str(getattr(self, "auto_scene_yaw_ref_mode", "fixed")).strip().lower()
        if mode not in ("fixed", "path_tangent_xy"):
            return "fixed"
        return mode

    def _auto_scene_figure_eight_lateral_amplitude(self, scene_mode: Optional[str] = None) -> float:
        lateral_axis = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_axis(
            getattr(self, "auto_scene_mode", "hover_only") if scene_mode is None else scene_mode
        )
        default_amp = float(getattr(self, "auto_scene_figure_eight_y_amplitude", 1.8))
        if lateral_axis == "x":
            return float(getattr(self, "auto_scene_figure_eight_x_amplitude", default_amp))
        return float(getattr(self, "auto_scene_figure_eight_y_amplitude", default_amp))

    def _auto_scene_figure_eight_entry_duration(self) -> float:
        return max(float(getattr(self, "auto_scene_figure_eight_ramp_duration", 5.0)), 1e-3)

    def _auto_scene_figure_eight_entry_phase_target(self) -> float:
        entry_duration = MMCUAVROS2Controller._auto_scene_figure_eight_entry_duration(self)
        phase_ratio = clamp(
            float(getattr(self, "auto_scene_figure_eight_entry_phase_ratio", 1.0)),
            0.20,
            1.0,
        )
        return float(entry_duration * phase_ratio)

    def _resolve_auto_scene_figure_eight_entry_end_time(self) -> Optional[float]:
        trigger_time = getattr(self, "auto_scene_figure_eight_trigger_time", None)
        if trigger_time is None:
            return None
        entry_end_time = getattr(self, "auto_scene_figure_eight_entry_end_time", None)
        if entry_end_time is not None:
            return float(entry_end_time)
        return float(trigger_time) + MMCUAVROS2Controller._auto_scene_figure_eight_entry_duration(self)

    def _resolve_auto_scene_adaptive_phase_start_time(
        self,
        t: float,
        *,
        mutate: bool = True,
    ) -> Optional[float]:
        cfg = MMCUAVROS2Controller._adaptive_phase_config(self)
        if not cfg.enabled:
            return None
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
            trigger_time = MMCUAVROS2Controller._resolve_auto_scene_figure_eight_start_time(
                self,
                t,
                mutate=mutate,
            )
            if trigger_time is None:
                return None
            if bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False)):
                entry_end_time = MMCUAVROS2Controller._resolve_auto_scene_figure_eight_entry_end_time(self)
                if entry_end_time is not None:
                    return float(entry_end_time)
            return float(trigger_time)
        return None

    def _query_auto_scene_adaptive_phase_time(
        self,
        t: float,
        *,
        activation_time: Optional[float] = None,
    ) -> float:
        query_t = float(t)
        cfg = MMCUAVROS2Controller._adaptive_phase_config(self)
        if int(getattr(self, "_adaptive_phase_bypass_depth", 0)) > 0:
            return query_t
        if not cfg.enabled:
            return query_t
        if activation_time is None:
            activation_time = MMCUAVROS2Controller._resolve_auto_scene_adaptive_phase_start_time(
                self,
                query_t,
                mutate=False,
            )
        if activation_time is None or query_t <= float(activation_time):
            return query_t
        phase_state = MMCUAVROS2Controller._ensure_auto_scene_adaptive_phase_state(self)
        if phase_state.last_update_time is None:
            return query_t
        effective_time = float(phase_state.effective_time) + float(phase_state.phase_rate) * (
            query_t - float(phase_state.last_update_time)
        )
        return max(float(activation_time), float(effective_time))

    def _planner_get_ref(
        self,
        t: float,
        *,
        apply_adaptive_phase: bool = True,
    ) -> RefPos:
        if apply_adaptive_phase or not hasattr(self, "planner"):
            return self.planner.get_ref(float(t))
        depth = int(getattr(self, "_adaptive_phase_bypass_depth", 0))
        setattr(self, "_adaptive_phase_bypass_depth", depth + 1)
        try:
            return self.planner.get_ref(float(t))
        finally:
            setattr(self, "_adaptive_phase_bypass_depth", depth)

    def _rviz_stamp_msg(self):
        if not hasattr(self, "get_clock"):
            return None
        try:
            return self.get_clock().now().to_msg()
        except Exception:
            return None

    def _new_rviz_path_message(self, stamp_msg=None) -> Path:
        path_msg = Path()
        if hasattr(path_msg, "header"):
            path_msg.header.frame_id = str(getattr(self, "rviz_trajectory_frame_id", "mmc_world"))
            if stamp_msg is not None:
                path_msg.header.stamp = stamp_msg
        if not hasattr(path_msg, "poses") or path_msg.poses is None:
            path_msg.poses = []
        return path_msg

    def _make_rviz_pose_stamped(self, x: float, y: float, z: float, stamp_msg=None) -> PoseStamped:
        pose_msg = PoseStamped()
        if hasattr(pose_msg, "header"):
            pose_msg.header.frame_id = str(getattr(self, "rviz_trajectory_frame_id", "mmc_world"))
            if stamp_msg is not None:
                pose_msg.header.stamp = stamp_msg
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        return pose_msg

    def _quaternion_from_yaw(self, yaw: float) -> tuple[float, float, float, float]:
        half_yaw = 0.5 * float(yaw)
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def _rviz_reference_path_end_time(self) -> float:
        takeoff_transition_time = max(
            float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
            1e-6,
        )
        hover_hold_time = max(
            float(getattr(self, "auto_scene_hover_hold_time", 4.0)),
            0.0,
        )
        nominal_trigger_time = takeoff_transition_time + hover_hold_time
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
            figure_period = max(float(getattr(self, "auto_scene_figure_eight_period", 16.0)), 1e-6)
            ramp_duration = max(float(getattr(self, "auto_scene_figure_eight_ramp_duration", 5.0)), 1e-3)
            cycles = max(float(getattr(self, "rviz_reference_path_cycles", 1.0)), 0.25)
            return float(nominal_trigger_time + ramp_duration + figure_period * cycles)
        if scene_mode in ("hover_to_point_hold", "hover_to_point_yaw_step_hold"):
            return float(
                nominal_trigger_time
                + max(
                    MMCUAVROS2Controller._resolved_auto_scene_move_duration(self),
                    float(getattr(self, "auto_scene_yaw_ramp_duration", 4.0)),
                )
                + 1.0
            )
        return float(nominal_trigger_time + max(MMCUAVROS2Controller._resolved_auto_scene_move_duration(self), 1.0))

    def _build_rviz_reference_path_message(self) -> Path:
        stamp_msg = MMCUAVROS2Controller._rviz_stamp_msg(self)
        path_msg = MMCUAVROS2Controller._new_rviz_path_message(self, stamp_msg)
        if not hasattr(self, "planner"):
            return path_msg

        sample_dt = max(float(getattr(self, "rviz_reference_path_dt", 0.05)), 0.01)
        end_time = max(MMCUAVROS2Controller._rviz_reference_path_end_time(self), sample_dt)
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        restore_trigger = False
        old_trigger_time = getattr(self, "auto_scene_figure_eight_trigger_time", None)
        if (
            MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode)
            and old_trigger_time is None
        ):
            self.auto_scene_figure_eight_trigger_time = (
                float(getattr(self, "auto_scene_takeoff_transition_time", 5.0))
                + float(getattr(self, "auto_scene_hover_hold_time", 4.0))
            )
            restore_trigger = True
        try:
            sample_times = np.arange(0.0, end_time + 0.5 * sample_dt, sample_dt, dtype=float)
            for sample_t in sample_times:
                ref = MMCUAVROS2Controller._planner_get_ref(
                    self,
                    float(sample_t),
                    apply_adaptive_phase=False,
                )
                path_msg.poses.append(
                    MMCUAVROS2Controller._make_rviz_pose_stamped(
                        self,
                        ref.x,
                        ref.y,
                        ref.z,
                        stamp_msg,
                    )
                )
        finally:
            if restore_trigger:
                self.auto_scene_figure_eight_trigger_time = old_trigger_time
        return path_msg

    def _publish_rviz_reference_path_if_needed(self) -> None:
        if (
            not bool(getattr(self, "rviz_trajectory_enabled", False))
            or getattr(self, "pub_reference_path", None) is None
            or bool(getattr(self, "rviz_reference_path_published", False))
        ):
            return
        path_msg = MMCUAVROS2Controller._build_rviz_reference_path_message(self)
        self.rviz_reference_path_msg = path_msg
        self.pub_reference_path.publish(path_msg)
        self.rviz_reference_path_published = True

    def _publish_rviz_actual_path(self, state_snapshot: State6) -> None:
        if not bool(getattr(self, "rviz_trajectory_enabled", False)):
            return
        publisher = getattr(self, "pub_actual_path", None)
        if publisher is None:
            return
        stamp_msg = MMCUAVROS2Controller._rviz_stamp_msg(self)
        if getattr(self, "rviz_actual_path_msg", None) is None:
            self.rviz_actual_path_msg = MMCUAVROS2Controller._new_rviz_path_message(self, stamp_msg)
        path_msg = self.rviz_actual_path_msg
        path_msg.header.frame_id = str(getattr(self, "rviz_trajectory_frame_id", "mmc_world"))
        if stamp_msg is not None:
            path_msg.header.stamp = stamp_msg
        path_msg.poses.append(
            MMCUAVROS2Controller._make_rviz_pose_stamped(
                self,
                state_snapshot.x,
                state_snapshot.y,
                state_snapshot.z,
                stamp_msg,
            )
        )
        max_points = max(int(getattr(self, "rviz_actual_path_max_points", 5000)), 1)
        overflow = len(path_msg.poses) - max_points
        if overflow > 0:
            del path_msg.poses[:overflow]
        publisher.publish(path_msg)

    def _publish_rviz_vehicle_marker(self, state_snapshot: State6) -> None:
        if not bool(getattr(self, "rviz_vehicle_marker_enabled", False)):
            return
        publisher = getattr(self, "pub_vehicle_marker", None)
        if publisher is None:
            return

        stamp_msg = MMCUAVROS2Controller._rviz_stamp_msg(self)
        frame_id = str(getattr(self, "rviz_trajectory_frame_id", "mmc_world"))
        sphere_diameter = max(float(getattr(self, "rviz_vehicle_sphere_diameter", 0.18)), 0.02)
        arrow_length = max(float(getattr(self, "rviz_vehicle_arrow_length", 0.32)), 0.05)
        arrow_shaft_diameter = max(float(getattr(self, "rviz_vehicle_arrow_shaft_diameter", 0.035)), 0.005)
        arrow_head_diameter = max(float(getattr(self, "rviz_vehicle_arrow_head_diameter", 0.08)), 0.01)
        arrow_z_offset = float(getattr(self, "rviz_vehicle_arrow_z_offset", 0.03))
        marker_yaw = getattr(self, "rviz_vehicle_marker_yaw", None)
        if marker_yaw is None:
            marker_yaw = float(state_snapshot.psi)
        qx, qy, qz, qw = MMCUAVROS2Controller._quaternion_from_yaw(self, marker_yaw)

        marker_array = MarkerArray()

        sphere_marker = Marker()
        sphere_marker.header.frame_id = frame_id
        if stamp_msg is not None:
            sphere_marker.header.stamp = stamp_msg
        sphere_marker.ns = "mmc_vehicle"
        sphere_marker.id = 0
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position.x = float(state_snapshot.x)
        sphere_marker.pose.position.y = float(state_snapshot.y)
        sphere_marker.pose.position.z = float(state_snapshot.z)
        sphere_marker.pose.orientation.x = 0.0
        sphere_marker.pose.orientation.y = 0.0
        sphere_marker.pose.orientation.z = 0.0
        sphere_marker.pose.orientation.w = 1.0
        sphere_marker.scale.x = sphere_diameter
        sphere_marker.scale.y = sphere_diameter
        sphere_marker.scale.z = sphere_diameter
        sphere_marker.color.r = 0.15
        sphere_marker.color.g = 0.67
        sphere_marker.color.b = 0.96
        sphere_marker.color.a = 0.95
        marker_array.markers.append(sphere_marker)

        yaw_marker = Marker()
        yaw_marker.header.frame_id = frame_id
        if stamp_msg is not None:
            yaw_marker.header.stamp = stamp_msg
        yaw_marker.ns = "mmc_vehicle"
        yaw_marker.id = 1
        yaw_marker.type = Marker.ARROW
        yaw_marker.action = Marker.ADD
        yaw_marker.pose.position.x = float(state_snapshot.x)
        yaw_marker.pose.position.y = float(state_snapshot.y)
        yaw_marker.pose.position.z = float(state_snapshot.z + arrow_z_offset)
        yaw_marker.pose.orientation.x = qx
        yaw_marker.pose.orientation.y = qy
        yaw_marker.pose.orientation.z = qz
        yaw_marker.pose.orientation.w = qw
        yaw_marker.scale.x = arrow_length
        yaw_marker.scale.y = arrow_shaft_diameter
        yaw_marker.scale.z = arrow_head_diameter
        yaw_marker.color.r = 0.93
        yaw_marker.color.g = 0.24
        yaw_marker.color.b = 0.17
        yaw_marker.color.a = 0.98
        marker_array.markers.append(yaw_marker)

        self.rviz_vehicle_marker_array = marker_array
        publisher.publish(marker_array)

    def _estimate_auto_scene_path_projection(
        self,
        *,
        state_snapshot: State6,
        activation_time: float,
        center_time: float,
    ) -> tuple[Optional[float], Optional[RefPos]]:
        if not hasattr(self, "planner"):
            return None, None

        search_back = 0.70
        search_ahead = 0.25
        sample_count = 41
        start_t = max(float(activation_time), float(center_time) - search_back)
        end_t = max(start_t, float(center_time) + search_ahead)
        if end_t <= start_t + 1e-9:
            ref = MMCUAVROS2Controller._planner_get_ref(self, start_t, apply_adaptive_phase=False)
            return float(start_t), ref

        state_pos = np.array(
            [float(state_snapshot.x), float(state_snapshot.y), float(state_snapshot.z)],
            dtype=float,
        )
        state_vel = np.array(
            [float(state_snapshot.vx), float(state_snapshot.vy), float(state_snapshot.vz)],
            dtype=float,
        )
        best_cost = float("inf")
        best_time: Optional[float] = None
        best_ref: Optional[RefPos] = None
        for candidate_t in np.linspace(start_t, end_t, sample_count):
            ref_candidate = MMCUAVROS2Controller._planner_get_ref(
                self,
                float(candidate_t),
                apply_adaptive_phase=False,
            )
            ref_pos = np.array(
                [float(ref_candidate.x), float(ref_candidate.y), float(ref_candidate.z)],
                dtype=float,
            )
            ref_vel = np.array(
                [float(ref_candidate.vx), float(ref_candidate.vy), float(ref_candidate.vz)],
                dtype=float,
            )
            pos_err = ref_pos - state_pos
            vel_err = ref_vel - state_vel
            cost = float(np.dot(pos_err, pos_err) + 0.25 * np.dot(vel_err, vel_err))
            if cost < best_cost:
                best_cost = cost
                best_time = float(candidate_t)
                best_ref = ref_candidate
        return best_time, best_ref

    def _update_auto_scene_adaptive_phase_state(
        self,
        t: float,
        state_snapshot: State6,
        base_ref: RefPos,
    ):
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if not MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
            # 历史字段名仍叫 Adaptive_phase_active，但在第5章新口径下也承担
            # “有效验证窗口”标记：排除起飞和进入场景前的悬停等待段，使
            # 单点保持、偏航阶跃和纯悬停风扰日志都能被同一统计脚本追溯。
            effective_start_time = max(
                float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
                1e-6,
            ) + max(float(getattr(self, "auto_scene_hover_hold_time", 4.0)), 0.0)
            if scene_mode in (
                "hover_only",
                "hover_to_point_hold",
                "hover_to_point_yaw_step_hold",
                "hover_to_yaw_step_hold",
                "hover_to_open_loop_rotor_diff",
            ) and float(t) >= effective_start_time:
                self.last_adaptive_phase_active = True
                self.last_adaptive_phase_time = float(t)
                self.last_adaptive_phase_rate = 1.0
                self.last_adaptive_phase_metric = 0.0
            else:
                MMCUAVROS2Controller._reset_auto_scene_adaptive_phase_state(self)
            return

        cfg = MMCUAVROS2Controller._adaptive_phase_config(self)
        activation_time = MMCUAVROS2Controller._resolve_auto_scene_adaptive_phase_start_time(
            self,
            t,
            mutate=False,
        )
        if not cfg.enabled or activation_time is None or float(t) <= float(activation_time):
            MMCUAVROS2Controller._reset_auto_scene_adaptive_phase_state(self)
            return

        phase_state = MMCUAVROS2Controller._ensure_auto_scene_adaptive_phase_state(self)
        curr_t = float(t)
        prev_update_time = phase_state.last_update_time
        if phase_state.last_update_time is None:
            phase_state.last_update_time = curr_t
            phase_state.effective_time = curr_t
            phase_state.phase_rate = 1.0
            phase_state.phase_accel = 0.0
            phase_state.error_metric = 0.0
        else:
            dt = max(curr_t - float(phase_state.last_update_time), 0.0)
            if dt > 0.0:
                phase_state.effective_time = float(phase_state.effective_time) + float(
                    phase_state.phase_rate
                ) * dt
                phase_state.last_update_time = curr_t
        dt_filter = (
            max(curr_t - float(prev_update_time), 0.0)
            if prev_update_time is not None
            else 0.0
        )

        ref_for_metric = base_ref
        projection_time, projection_ref = MMCUAVROS2Controller._estimate_auto_scene_path_projection(
            self,
            state_snapshot=state_snapshot,
            activation_time=float(activation_time),
            center_time=float(phase_state.effective_time),
        )
        projection_rate = 1.0
        if projection_time is not None and projection_ref is not None:
            projection_offset = float(projection_time) - float(phase_state.effective_time)
            if abs(projection_offset) > float(cfg.projection_deadband):
                correction_target = clamp(
                    projection_offset,
                    -float(cfg.projection_max_correction),
                    float(cfg.projection_max_correction),
                )
                if prev_update_time is None or cfg.projection_align_time_constant <= 1e-9 or dt_filter <= 1e-9:
                    correction = correction_target
                else:
                    beta = dt_filter / (float(cfg.projection_align_time_constant) + dt_filter)
                    correction = beta * correction_target
                phase_state.effective_time = max(
                    float(activation_time),
                    float(phase_state.effective_time) + float(correction),
                )
            ref_for_metric = projection_ref
            projection_offset = float(projection_time) - float(phase_state.effective_time)
            projection_rate = clamp(1.0 + projection_offset / 1.20, float(cfg.min_rate), 1.0)

        ref_vel = np.array(
            [float(ref_for_metric.vx), float(ref_for_metric.vy), float(ref_for_metric.vz)],
            dtype=float,
        )
        pos_err = np.array(
            [
                float(ref_for_metric.x) - float(state_snapshot.x),
                float(ref_for_metric.y) - float(state_snapshot.y),
                float(ref_for_metric.z) - float(state_snapshot.z),
            ],
            dtype=float,
        )
        vel_err = ref_vel - np.array(
            [float(state_snapshot.vx), float(state_snapshot.vy), float(state_snapshot.vz)],
            dtype=float,
        )
        path_speed = float(np.linalg.norm(ref_vel))
        nominal_speed = max(path_speed, float(cfg.speed_floor))
        if path_speed <= 1e-6:
            target_rate = 1.0
            error_metric = 0.0
        else:
            tangent = ref_vel / path_speed
            along_err_signed = float(np.dot(pos_err, tangent))
            along_err = max(along_err_signed, 0.0)
            cross_err_vec = pos_err - along_err_signed * tangent
            cross_err = float(np.linalg.norm(cross_err_vec))
            along_scale = max(
                float(cfg.position_floor),
                nominal_speed * float(cfg.along_track_window),
            )
            cross_scale = max(
                float(cfg.position_floor),
                nominal_speed * float(cfg.cross_track_window),
            )
            vel_scale = max(float(cfg.velocity_floor), nominal_speed)
            error_metric = (
                float(cfg.lag_weight) * (along_err / along_scale) ** 2
                + float(cfg.cross_track_weight) * (cross_err / cross_scale) ** 2
                + float(cfg.velocity_weight) * (float(np.linalg.norm(vel_err)) / vel_scale) ** 2
            )
            target_rate = min(
                projection_rate,
                1.0 / math.sqrt(1.0 + error_metric),
            )

        target_rate = clamp(float(target_rate), float(cfg.min_rate), 1.0)
        prev_rate = float(phase_state.phase_rate)
        if prev_update_time is None or cfg.filter_time_constant <= 1e-9 or dt_filter <= 1e-9:
            new_rate = target_rate
        else:
            alpha = dt_filter / (float(cfg.filter_time_constant) + dt_filter)
            new_rate = prev_rate + alpha * (target_rate - prev_rate)
        phase_state.phase_rate = clamp(float(new_rate), float(cfg.min_rate), 1.0)
        if dt_filter > 1e-9:
            phase_state.phase_accel = (float(phase_state.phase_rate) - prev_rate) / dt_filter
        else:
            phase_state.phase_accel = 0.0
        phase_state.error_metric = float(error_metric)

        self.last_adaptive_phase_active = True
        self.last_adaptive_phase_time = float(phase_state.effective_time)
        self.last_adaptive_phase_rate = float(phase_state.phase_rate)
        self.last_adaptive_phase_metric = float(phase_state.error_metric)

    def _default_trajectory_functions(self):
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        hover_z = float(getattr(self, "auto_scene_target_z", 1.5))
        target_x = float(getattr(self, "auto_scene_target_x", 0.0))
        target_y = float(getattr(self, "auto_scene_target_y", 3.0))
        target_z = float(getattr(self, "auto_scene_target_z", hover_z))
        takeoff_transition_time = max(
            float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
            1e-6,
        )
        hover_hold_time = max(
            float(getattr(self, "auto_scene_hover_hold_time", 4.0)),
            0.0,
        )
        move_duration = MMCUAVROS2Controller._resolved_auto_scene_move_duration(self)
        move_start = takeoff_transition_time + hover_hold_time

        def quintic_progress(alpha: float) -> float:
            alpha = clamp(float(alpha), 0.0, 1.0)
            return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))

        figure_eight_lateral_axis = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_axis(scene_mode)
        if figure_eight_lateral_axis is not None:
            lateral_amplitude = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_amplitude(
                self,
                scene_mode,
            )
            z_amplitude = float(getattr(self, "auto_scene_figure_eight_z_amplitude", 0.675))
            forward_tilt_rad = math.radians(
                MMCUAVROS2Controller._auto_scene_figure_eight_forward_tilt_deg(self, scene_mode)
            )
            figure_period = max(float(getattr(self, "auto_scene_figure_eight_period", 16.0)), 1e-6)
            ramp_duration = MMCUAVROS2Controller._auto_scene_figure_eight_entry_duration(self)
            omega = (2.0 * math.pi) / figure_period
            entry_duration = max(ramp_duration, 1e-3)
            entry_phase_target = MMCUAVROS2Controller._auto_scene_figure_eight_entry_phase_target(self)

            tilt_sin = math.sin(forward_tilt_rad)
            tilt_cos = math.cos(forward_tilt_rad)

            def orbit_state_at_phase(
                phase: float,
            ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
                phase = float(phase)
                lateral_pos = float(lateral_amplitude * math.sin(omega * phase))
                lateral_vel = float(lateral_amplitude * omega * math.cos(omega * phase))
                lateral_acc = float(-lateral_amplitude * (omega ** 2) * math.sin(omega * phase))

                vertical_offset = float(z_amplitude * math.sin(2.0 * omega * phase))
                vertical_vel = float(2.0 * omega * z_amplitude * math.cos(2.0 * omega * phase))
                vertical_acc = float(-4.0 * (omega ** 2) * z_amplitude * math.sin(2.0 * omega * phase))

                if figure_eight_lateral_axis == "x":
                    return (
                        (lateral_pos, 0.0, float(hover_z + vertical_offset)),
                        (lateral_vel, 0.0, vertical_vel),
                        (lateral_acc, 0.0, vertical_acc),
                    )

                return (
                    (
                        float(vertical_offset * tilt_sin),
                        lateral_pos,
                        float(hover_z + vertical_offset * tilt_cos),
                    ),
                    (
                        float(vertical_vel * tilt_sin),
                        lateral_vel,
                        float(vertical_vel * tilt_cos),
                    ),
                    (
                        float(vertical_acc * tilt_sin),
                        lateral_acc,
                        float(vertical_acc * tilt_cos),
                    ),
                )

            def figure_eight_trigger_time() -> Optional[float]:
                trigger_time = getattr(self, "auto_scene_figure_eight_trigger_time", None)
                if trigger_time is None and not hasattr(self, "state"):
                    return float(move_start)
                if trigger_time is None:
                    return None
                return float(trigger_time)

            def figure_eight_phase(t: float) -> float:
                trigger_time = figure_eight_trigger_time()
                if trigger_time is None:
                    return 0.0
                bridge_end_time = (
                    MMCUAVROS2Controller._resolve_auto_scene_figure_eight_entry_end_time(self)
                    if bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False))
                    else None
                )
                if bridge_end_time is not None:
                    if float(t) <= trigger_time:
                        return 0.0
                    if float(t) < float(bridge_end_time):
                        return max(float(t) - trigger_time, 0.0)
                    effective_t = MMCUAVROS2Controller._query_auto_scene_adaptive_phase_time(
                        self,
                        float(t),
                        activation_time=float(bridge_end_time),
                    )
                    return entry_phase_target + max(float(effective_t) - float(bridge_end_time), 0.0)
                effective_t = MMCUAVROS2Controller._query_auto_scene_adaptive_phase_time(
                    self,
                    float(t),
                    activation_time=trigger_time,
                )
                phase = max(float(effective_t) - trigger_time, 0.0)
                if (
                    bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False))
                    and bridge_end_time is not None
                    and float(t) >= float(bridge_end_time)
                    and bool(getattr(self, "auto_scene_figure_eight_phase_bias_locked", False))
                ):
                    phase += float(getattr(self, "auto_scene_figure_eight_phase_bias", 0.0))
                return phase

            def resolved_phase(phase: float) -> float:
                if bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False)):
                    return max(float(phase), 0.0)
                return max(float(phase) - entry_duration + entry_phase_target, 0.0)

            def traj_x(t):
                trigger_time = figure_eight_trigger_time()
                if trigger_time is None or t <= trigger_time:
                    return 0.0
                phase = figure_eight_phase(t)
                return orbit_state_at_phase(resolved_phase(phase))[0][0]

            def traj_y(t):
                trigger_time = figure_eight_trigger_time()
                if trigger_time is None or t <= trigger_time:
                    return 0.0
                phase = figure_eight_phase(t)
                return orbit_state_at_phase(resolved_phase(phase))[0][1]

            def traj_z(t):
                trigger_time = figure_eight_trigger_time()
                if trigger_time is None or t <= trigger_time:
                    return hover_z
                phase = figure_eight_phase(t)
                return orbit_state_at_phase(resolved_phase(phase))[0][2]

            return traj_x, traj_y, traj_z

        def quintic_blend(t: float, start_value: float, end_value: float) -> float:
            if scene_mode in ("hover_only", "hover_to_yaw_step_hold", "hover_to_open_loop_rotor_diff"):
                return float(start_value)
            if t <= move_start:
                return float(start_value)
            if t >= (move_start + move_duration):
                return float(end_value)
            alpha = (float(t) - move_start) / move_duration
            smooth_alpha = quintic_progress(alpha)
            return float(start_value + (end_value - start_value) * smooth_alpha)

        def traj_x(t):
            return quintic_blend(t, 0.0, target_x)

        def traj_y(t):
            return quintic_blend(t, 0.0, target_y)

        def traj_z(t):
            return quintic_blend(t, hover_z, target_z)

        return traj_x, traj_y, traj_z

    def _build_outer_ref(self, base_ref: RefPos, manual_status: ManualXYStatus) -> RefPos:
        ref_x = base_ref.x if manual_status.ref_x is None else manual_status.ref_x
        ref_y = base_ref.y if manual_status.ref_y is None else manual_status.ref_y
        ref_vx = manual_status.ref_vx if manual_status.ref_x is not None else base_ref.vx
        ref_vy = manual_status.ref_vy if manual_status.ref_y is not None else base_ref.vy
        ref_ax = manual_status.ref_ax if manual_status.ref_x is not None else base_ref.ax
        ref_ay = manual_status.ref_ay if manual_status.ref_y is not None else base_ref.ay

        return RefPos(
            x=ref_x,
            y=ref_y,
            z=base_ref.z,
            vx=ref_vx,
            vy=ref_vy,
            vz=base_ref.vz,
            ax=ref_ax,
            ay=ref_ay,
            az=base_ref.az,
            psi=manual_status.psi_ref_override if manual_status.psi_ref_override is not None else base_ref.psi,
        )

    def _attitude_step_targets_at_time(
        self,
        t: float,
        fallback_phi: float,
        fallback_theta: float,
    ) -> Tuple[float, float]:
        if not getattr(self, "attitude_step_test_enabled", False):
            return float(fallback_phi), float(fallback_theta)
        if not getattr(self, "attitude_step_test_active", False):
            return float(fallback_phi), float(fallback_theta)

        axis = str(getattr(self, "attitude_step_test_axis", "pitch")).strip().lower()
        if axis not in ("pitch", "roll"):
            axis = "pitch"

        start_time = getattr(self, "attitude_step_test_start_time", None)
        if start_time is None:
            return float(fallback_phi), float(fallback_theta)

        hold_time = max(float(getattr(self, "attitude_step_hold_time", 3.0)), 1e-6)
        recovery_time = max(float(getattr(self, "attitude_step_recovery_time", 2.0)), 1e-6)
        cycle_time = 2.0 * (hold_time + recovery_time)
        cycle_phase = max(0.0, float(t) - float(start_time)) % cycle_time
        target_angle = float(getattr(self, "attitude_step_test_angle_ref", 0.0))

        if cycle_phase < hold_time:
            signed_target = target_angle
        elif cycle_phase < (hold_time + recovery_time):
            signed_target = None
        elif cycle_phase < (2.0 * hold_time + recovery_time):
            signed_target = -target_angle
        else:
            signed_target = None

        phi_target = float(fallback_phi)
        theta_target = float(fallback_theta)
        if signed_target is None:
            return phi_target, theta_target

        if axis == "roll":
            phi_target = signed_target
        else:
            theta_target = signed_target
        return phi_target, theta_target

    def _attitude_step_psi_target_at_time(self, t: float, fallback_psi: float) -> float:
        if not getattr(self, "attitude_step_test_enabled", False):
            return float(fallback_psi)
        if not getattr(self, "attitude_step_test_active", False):
            return float(fallback_psi)

        axis = str(getattr(self, "attitude_step_test_axis", "pitch")).strip().lower()
        if axis != "yaw":
            return float(fallback_psi)

        start_time = getattr(self, "attitude_step_test_start_time", None)
        if start_time is None:
            return float(fallback_psi)

        hold_time = max(float(getattr(self, "attitude_step_hold_time", 3.0)), 1e-6)
        recovery_time = max(float(getattr(self, "attitude_step_recovery_time", 2.0)), 1e-6)
        cycle_time = 2.0 * (hold_time + recovery_time)
        cycle_phase = max(0.0, float(t) - float(start_time)) % cycle_time
        target_angle = float(getattr(self, "attitude_step_test_angle_ref", 0.0))

        if cycle_phase < hold_time:
            return target_angle
        if cycle_phase < (hold_time + recovery_time):
            return float(fallback_psi)
        if cycle_phase < (2.0 * hold_time + recovery_time):
            return -target_angle
        return float(fallback_psi)

    def _resolve_attitude_mapping_psi(self, ref_psi: float, current_psi: float) -> float:
        """
        选择将世界系水平加速度映射到机体系姿态时应采用的偏航角.

        在“平移 + 同步偏航”的耦合场景里，如果直接使用目标偏航角，
        那么当 yaw_ref 明显领先实际 yaw 时，姿态分配会提前旋转，
        导致世界 X/Y 平移任务被错误投影到侧向通道。
        因此该场景优先使用当前实际偏航角来做当前控制步与短预测域内的
        水平加速度几何映射；偏航参考本身仍然保持 ref_psi 交给 yaw 通道。
        """
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if scene_mode == "hover_to_point_yaw_step_hold":
            return float(current_psi)
        if (
            MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode)
            and MMCUAVROS2Controller._auto_scene_yaw_ref_mode(self) != "fixed"
        ):
            yaw_err_abs = abs(wrap_angle_pi(float(ref_psi) - float(current_psi)))
            blend_start = math.radians(
                max(
                    0.0,
                    float(getattr(self, "auto_scene_attitude_mapping_yaw_blend_start_deg", 3.0)),
                )
            )
            blend_stop = math.radians(
                max(
                    0.0,
                    float(getattr(self, "auto_scene_attitude_mapping_yaw_blend_stop_deg", 15.0)),
                )
            )
            if blend_stop <= blend_start + 1e-9:
                blend_alpha = 1.0 if yaw_err_abs <= blend_start else 0.0
            elif yaw_err_abs <= blend_start:
                blend_alpha = 1.0
            elif yaw_err_abs >= blend_stop:
                blend_alpha = 0.0
            else:
                normalized = (yaw_err_abs - blend_start) / max(blend_stop - blend_start, 1e-9)
                blend_alpha = 1.0 - quintic_smoothstep(normalized)
            return blend_angle_near(float(current_psi), float(ref_psi), blend_alpha)
        return float(ref_psi)

    def _warn_ndo_once_per_second(self, message: str):
        now = self._elapsed_sec()
        if (now - float(getattr(self, "_last_ndo_warning_time", -math.inf))) >= 1.0:
            self.get_logger().warning(message)
            self._last_ndo_warning_time = now

    def _set_ndo_log_row(
        self,
        *,
        comp_phi_total: float = 0.0,
        comp_theta_total: float = 0.0,
        comp_phi_force: float = 0.0,
        comp_theta_force: float = 0.0,
        comp_phi_torque: float = 0.0,
        comp_theta_torque: float = 0.0,
    ):
        if not getattr(self, "ndo_enabled", False):
            self._last_ndo_log_row = self._zero_ndo_log_row
            return

        d_force_hat, d_torque_hat = self.ndo.get_disturbance_estimates()
        row = (
            float(d_force_hat[0]),
            float(d_force_hat[1]),
            float(d_force_hat[2]),
            float(d_torque_hat[0]),
            float(d_torque_hat[1]),
            float(d_torque_hat[2]),
            math.degrees(float(comp_phi_force)),
            math.degrees(float(comp_theta_force)),
            math.degrees(float(comp_phi_torque)),
            math.degrees(float(comp_theta_torque)),
            math.degrees(float(comp_phi_total)),
            math.degrees(float(comp_theta_total)),
        )
        if not all(math.isfinite(value) for value in row):
            self._warn_ndo_once_per_second("NDO produced non-finite diagnostics; zeroing NDO log row for this cycle.")
            self._last_ndo_log_row = self._zero_ndo_log_row
            return
        self._last_ndo_log_row = row

    def _reference_attitude_limits_for_ndo(self) -> Tuple[float, float]:
        compensated_limit = MMCUAVROS2Controller._effective_ndo_compensated_attitude_limit(self)
        roll_limit = max(
            float(getattr(self.outer, "roll_limit", math.radians(12.0))),
            compensated_limit,
        )
        pitch_limit = max(
            float(getattr(self.outer, "pitch_limit", math.radians(12.0))),
            compensated_limit,
        )
        return roll_limit, pitch_limit

    def _effective_ndo_compensated_attitude_limit(self) -> float:
        base_limit = max(0.0, float(getattr(self, "ndo_compensated_attitude_limit", 0.0)))
        if not bool(getattr(self, "ndo_transient_attitude_boost_enabled", False)):
            return base_limit
        if not bool(getattr(self, "wind_runtime_active", False)):
            return base_limit

        activation_time = float(getattr(self, "wind_activation_time", math.nan))
        if not math.isfinite(activation_time):
            return base_limit
        elapsed = max(0.0, float(self._elapsed_sec()) - activation_time)
        boost_duration = max(0.0, float(getattr(self, "ndo_transient_attitude_boost_duration", 0.0)))
        fade_duration = max(0.0, float(getattr(self, "ndo_transient_attitude_boost_fade", 0.0)))
        boosted_limit = max(base_limit, max(0.0, float(getattr(self, "ndo_transient_attitude_limit", base_limit))))

        if elapsed <= boost_duration:
            return boosted_limit
        if fade_duration <= 1e-9 or elapsed >= boost_duration + fade_duration:
            return base_limit
        alpha = clamp((elapsed - boost_duration) / fade_duration, 0.0, 1.0)
        return boosted_limit + alpha * (base_limit - boosted_limit)

    def _effective_ndo_compensation_limit(self) -> float:
        base_limit = max(0.0, float(getattr(self, "ndo_compensation_limit", 0.0)))
        if not bool(getattr(self, "ndo_compensation_limit_schedule_enabled", False)):
            return base_limit

        wind_summary = getattr(self, "wind_config_summary", None)
        wind_speed = float(getattr(wind_summary, "wind_speed_world", 0.0) or 0.0)
        if not math.isfinite(wind_speed):
            return base_limit

        low_speed = max(0.0, float(getattr(self, "ndo_compensation_limit_low_speed", 3.0)))
        high_speed = max(0.0, float(getattr(self, "ndo_compensation_limit_high_speed", 5.0)))
        low_limit = max(0.0, float(getattr(self, "ndo_compensation_limit_low", base_limit)))
        high_limit = max(0.0, float(getattr(self, "ndo_compensation_limit_high", base_limit)))
        if high_speed <= low_speed + 1e-9:
            scheduled = high_limit if wind_speed >= high_speed else low_limit
        else:
            alpha = clamp((wind_speed - low_speed) / (high_speed - low_speed), 0.0, 1.0)
            scheduled = low_limit + alpha * (high_limit - low_limit)
        return max(0.0, float(scheduled))

    @staticmethod
    def _relieve_ndo_compensation_against_feedback_angle(
        compensation: float,
        feedback_angle: float,
        *,
        gain: float,
        deadband: float,
        max_fraction: float,
    ) -> float:
        """
        在不改变 NDO→姿态前馈→内环 MPC 结构的前提下，给前馈姿态参考加
        一个“反馈反向抗振”整形：当外环反馈角与 NDO 前馈角方向相反且已经
        超过小死区时，说明位置环正在持续抵消前馈量；此时只削减一部分
        前馈幅值，避免补偿角在目标点附近把内环长期顶在同一方向。
        """
        comp = float(compensation)
        fb = float(feedback_angle)
        if not (math.isfinite(comp) and math.isfinite(fb)):
            return comp
        comp_abs = abs(comp)
        if comp_abs <= 1e-12 or comp * fb >= 0.0:
            return comp
        opposing = max(0.0, abs(fb) - max(0.0, float(deadband)))
        if opposing <= 0.0:
            return comp
        denom = max(comp_abs, max(0.0, float(deadband)), 1e-6)
        fraction = clamp(float(gain) * opposing / denom, 0.0, clamp(float(max_fraction), 0.0, 0.95))
        return comp * (1.0 - fraction)

    def _shape_ndo_compensation_against_outer_feedback(
        self,
        comp_phi_total: float,
        comp_theta_total: float,
        raw_phi: Sequence[float],
        raw_theta: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_phi_arr = np.asarray(raw_phi, dtype=float)
        raw_theta_arr = np.asarray(raw_theta, dtype=float)
        comp_phi_arr = np.full(raw_phi_arr.shape, float(comp_phi_total), dtype=float)
        comp_theta_arr = np.full(raw_theta_arr.shape, float(comp_theta_total), dtype=float)

        if not bool(getattr(self, "ndo_feedback_relief_enabled", False)):
            return comp_phi_arr, comp_theta_arr

        gain = max(0.0, float(getattr(self, "ndo_feedback_relief_gain", 1.0)))
        deadband = max(0.0, float(getattr(self, "ndo_feedback_relief_deadband", 0.0)))
        max_fraction = clamp(float(getattr(self, "ndo_feedback_relief_max_fraction", 0.65)), 0.0, 0.95)
        for idx in range(comp_phi_arr.size):
            comp_phi_arr[idx] = MMCUAVROS2Controller._relieve_ndo_compensation_against_feedback_angle(
                comp_phi_arr[idx],
                raw_phi_arr[idx],
                gain=gain,
                deadband=deadband,
                max_fraction=max_fraction,
            )
            comp_theta_arr[idx] = MMCUAVROS2Controller._relieve_ndo_compensation_against_feedback_angle(
                comp_theta_arr[idx],
                raw_theta_arr[idx],
                gain=gain,
                deadband=deadband,
                max_fraction=max_fraction,
            )
        return comp_phi_arr, comp_theta_arr

    def _overwrite_ndo_log_total(self, comp_phi_total: float, comp_theta_total: float):
        row = tuple(getattr(self, "_last_ndo_log_row", getattr(self, "_zero_ndo_log_row", (0.0,) * 12)))
        if len(row) != 12:
            return
        updated = row[:10] + (
            math.degrees(float(comp_phi_total)),
            math.degrees(float(comp_theta_total)),
        )
        if all(math.isfinite(value) for value in updated):
            self._last_ndo_log_row = updated

    def _resolve_ndo_attitude_compensation(
        self,
        current_thrust: float,
    ) -> Tuple[float, float]:
        """
        计算当前内环周期可用的 NDO 姿态补偿。

        该函数只负责“是否允许补偿 + 补偿量 + 黑匣子日志”三件事。
        真正把补偿注入参考序列的位置由调用方决定。这样正常
        自动场景可以在 AxisManeuverCoordinator 之前注入补偿，
        避免出现“角度参考已补偿，但滑块参考仍按未补偿外环姿态
        生成”的内部矛盾；手动/姿态阶跃等直通路径仍可在末端注入。
        """
        zero_ndo_log_row = getattr(self, "_zero_ndo_log_row", (0.0,) * 12)
        if not getattr(self, "ndo_enabled", False) or not getattr(self, "_ndo_current_cycle_valid", False):
            self._last_ndo_log_row = zero_ndo_log_row
            return 0.0, 0.0

        wind_summary = getattr(self, "wind_config_summary", None)
        wind_enabled = bool(getattr(wind_summary, "enable_wind", False))
        wind_mode = str(getattr(wind_summary, "activation_mode", "")).strip().lower()
        wind_speed = float(getattr(wind_summary, "wind_speed_world", 0.0) or 0.0)
        wind_active = bool(getattr(self, "wind_runtime_active", False))
        apply_ndo = wind_active or (
            wind_enabled
            and wind_mode == "immediate"
            and wind_speed > 1e-6
        )
        if not apply_ndo:
            MMCUAVROS2Controller._set_ndo_log_row(self)
            return 0.0, 0.0

        try:
            (comp_phi_total, comp_theta_total), (comp_phi_f, comp_theta_f), (
                comp_phi_r,
                comp_theta_r,
            ) = self.ndo.get_combined_compensation(self.state, current_thrust)
        except Exception as exc:
            self._warn_ndo_once_per_second(f"NDO compensation failed; bypassing compensation for this cycle: {exc}")
            self._last_ndo_log_row = zero_ndo_log_row
            return 0.0, 0.0

        limit = MMCUAVROS2Controller._effective_ndo_compensation_limit(self)
        comp_phi_total = clamp(float(comp_phi_total), -limit, limit)
        comp_theta_total = clamp(float(comp_theta_total), -limit, limit)
        if not (math.isfinite(comp_phi_total) and math.isfinite(comp_theta_total)):
            self._warn_ndo_once_per_second("NDO compensation became non-finite; bypassing compensation for this cycle.")
            self._last_ndo_log_row = zero_ndo_log_row
            return 0.0, 0.0

        MMCUAVROS2Controller._set_ndo_log_row(
            self,
            comp_phi_total=comp_phi_total,
            comp_theta_total=comp_theta_total,
            comp_phi_force=comp_phi_f,
            comp_theta_force=comp_theta_f,
            comp_phi_torque=comp_phi_r,
            comp_theta_torque=comp_theta_r,
        )
        return float(comp_phi_total), float(comp_theta_total)

    def _apply_attitude_compensation_to_reference_sequence(
        self,
        seq: np.ndarray,
        comp_phi_total: float,
        comp_theta_total: float,
    ) -> np.ndarray:
        compensated = np.array(seq, dtype=float, copy=True)
        if compensated.ndim != 2 or compensated.shape[1] < 2:
            return compensated
        roll_limit, pitch_limit = MMCUAVROS2Controller._reference_attitude_limits_for_ndo(self)
        compensated[:, 0] = np.clip(compensated[:, 0] + comp_phi_total, -roll_limit, roll_limit)
        compensated[:, 1] = np.clip(compensated[:, 1] + comp_theta_total, -pitch_limit, pitch_limit)
        return compensated

    def _apply_ndo_compensation_to_reference_sequence(
        self,
        seq: np.ndarray,
        current_thrust: float,
    ) -> np.ndarray:
        comp_phi_total, comp_theta_total = MMCUAVROS2Controller._resolve_ndo_attitude_compensation(
            self,
            current_thrust,
        )
        if abs(comp_phi_total) <= 1e-12 and abs(comp_theta_total) <= 1e-12:
            return seq
        return MMCUAVROS2Controller._apply_attitude_compensation_to_reference_sequence(
            self,
            seq,
            comp_phi_total,
            comp_theta_total,
        )

    def _finalize_attitude_reference_sequence(
        self,
        seq: np.ndarray,
        current_thrust: float,
        *,
        apply_ndo: bool = True,
    ) -> np.ndarray:
        if apply_ndo:
            finalized = MMCUAVROS2Controller._apply_ndo_compensation_to_reference_sequence(
                self,
                seq,
                current_thrust,
            )
        else:
            finalized = seq
        finalized_array = np.asarray(finalized, dtype=float)
        if finalized_array.ndim == 2 and finalized_array.shape[0] > 0 and finalized_array.shape[1] >= 3:
            self.inner_phi_ref = float(finalized_array[0, 0])
            self.inner_theta_ref = float(finalized_array[0, 1])
            self.inner_psi_ref = float(finalized_array[0, 2])
        else:
            self.inner_phi_ref = 0.0
            self.inner_theta_ref = 0.0
            self.inner_psi_ref = 0.0
        return finalized

    def _build_attitude_reference_sequence(self, t: float) -> np.ndarray:
        with optional_lock(self):
            if getattr(self, "manual_xy_enabled", True) and self.manual_xy.ready:
                horizon = int(self.att_mpc.Np)
                seq = np.zeros((horizon, 10), dtype=float)
                seq[:, 0] = float(self.phi_ref)
                seq[:, 1] = float(self.theta_ref)
                state_for_yaw = getattr(self, "state", None)
                current_psi = float(getattr(state_for_yaw, "psi", self.psi_ref))
                seq[:, 2] = unwrap_angle_near(float(self.psi_ref), current_psi)

                self.raw_phi_ref = float(self.phi_ref)
                self.raw_theta_ref = float(self.theta_ref)
                self.shaped_phi_ref = float(self.phi_ref)
                self.shaped_theta_ref = float(self.theta_ref)
                self.shaped_p_ref = 0.0
                self.shaped_q_ref = 0.0
                self.coordinator_chi_ref = 0.0
                self.coordinator_ups_ref = 0.0
                return MMCUAVROS2Controller._finalize_attitude_reference_sequence(
                    self,
                    seq,
                    getattr(self, "thrust_cmd", P.M * P.g),
                )

            state_snapshot = State6(**self.state.__dict__)
            attitude_mapping_psi_now = MMCUAVROS2Controller._resolve_attitude_mapping_psi(
                self,
                ref_psi=self.ref_pos_now.psi,
                current_psi=state_snapshot.psi,
            )
            _, phi_ff_now, theta_ff_now, psi_ff_now = self.outer._accel_to_attitude(
                self.ref_pos_now.ax,
                self.ref_pos_now.ay,
                self.ref_pos_now.az,
                self.ref_pos_now.psi,
                attitude_mapping_psi=attitude_mapping_psi_now,
            )

            delta_phi = self.phi_ref - phi_ff_now
            delta_theta = self.theta_ref - theta_ff_now
            delta_psi = self.psi_ref - psi_ff_now
            future_refs: list[RefPos] = []
            if MMCUAVROS2Controller._planner_reference_is_effectively_static(self, t):
                raw_phi = np.full(self.att_mpc.Np, float(phi_ff_now + delta_phi), dtype=float)
                raw_theta = np.full(self.att_mpc.Np, float(theta_ff_now + delta_theta), dtype=float)
                raw_psi = np.full(self.att_mpc.Np, float(psi_ff_now + delta_psi), dtype=float)
            else:
                raw_psi = []
                for k in range(self.att_mpc.Np):
                    t_pred = t + k * self.att_mpc.dt
                    ref_future = self.planner.get_ref(t_pred)
                    ref_future.psi = MMCUAVROS2Controller._resolve_auto_scene_yaw_target(
                        self,
                        t=t_pred,
                        base_psi=ref_future.psi,
                        current_psi=state_snapshot.psi,
                        base_ref=ref_future,
                        mutate=False,
                    )
                    ref_future = MMCUAVROS2Controller._resolve_auto_scene_hover_point_lock(
                        self,
                        t=t_pred,
                        base_ref=ref_future,
                        state_snapshot=state_snapshot,
                        mutate=False,
                    )
                    future_refs.append(ref_future)
                    raw_psi.append(float(ref_future.psi + delta_psi))

            raw_psi = np.asarray(raw_psi, dtype=float)
            yaw_ref_rate_limit = max(
                0.0,
                float(getattr(self, "auto_scene_yaw_ref_rate_limit_rad_s", 0.0)),
            )
            if (
                yaw_ref_rate_limit > 0.0
                and MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(
                    getattr(self, "auto_scene_mode", "hover_only")
                )
                and MMCUAVROS2Controller._auto_scene_yaw_ref_mode(self) == "path_tangent_xy"
            ):
                yaw_anchor = getattr(self, "auto_scene_figure_eight_last_yaw_ref", None)
                if yaw_anchor is None:
                    yaw_anchor = getattr(getattr(self, "ref_pos_now", None), "psi", None)
                if yaw_anchor is None:
                    yaw_anchor = state_snapshot.psi
                raw_psi = slew_limit_angle_sequence(
                    raw_psi,
                    anchor=float(yaw_anchor),
                    max_delta=yaw_ref_rate_limit * max(float(self.att_mpc.dt), 1e-6),
                )
            raw_psi = unwrap_angle_sequence(raw_psi, float(state_snapshot.psi))
            yaw_rate_limit_for_sequence = yaw_ref_rate_limit if yaw_ref_rate_limit > 0.0 else None

            if future_refs:
                predicted_mapping_psi = predict_angle_tracking_sequence(
                    raw_psi,
                    initial_angle=float(state_snapshot.psi),
                    dt=self.att_mpc.dt,
                    rate_limit=yaw_rate_limit_for_sequence,
                )
                raw_phi = []
                raw_theta = []
                scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
                for k, ref_future in enumerate(future_refs):
                    mapping_current_psi = (
                        state_snapshot.psi
                        if scene_mode == "hover_to_point_yaw_step_hold"
                        else predicted_mapping_psi[k]
                    )
                    attitude_mapping_psi_future = MMCUAVROS2Controller._resolve_attitude_mapping_psi(
                        self,
                        ref_psi=raw_psi[k],
                        current_psi=mapping_current_psi,
                    )
                    _, phi_ff, theta_ff, _ = self.outer._accel_to_attitude(
                        ref_future.ax,
                        ref_future.ay,
                        ref_future.az,
                        raw_psi[k],
                        attitude_mapping_psi=attitude_mapping_psi_future,
                    )
                    raw_phi.append(float(phi_ff + delta_phi))
                    raw_theta.append(float(theta_ff + delta_theta))
                raw_phi = np.asarray(raw_phi, dtype=float)
                raw_theta = np.asarray(raw_theta, dtype=float)

            if getattr(self, "attitude_step_test_enabled", False) and getattr(self, "attitude_step_test_active", False):
                for k in range(self.att_mpc.Np):
                    phi_target, theta_target = self._attitude_step_targets_at_time(
                        t + k * self.att_mpc.dt,
                        raw_phi[k],
                        raw_theta[k],
                    )
                    raw_phi[k] = float(phi_target)
                    raw_theta[k] = float(theta_target)
                    raw_psi[k] = float(
                        MMCUAVROS2Controller._attitude_step_psi_target_at_time(
                            self,
                            t + k * self.att_mpc.dt,
                            raw_psi[k],
                        )
                    )

                raw_psi = unwrap_angle_sequence(raw_psi, float(state_snapshot.psi))
                raw_r = angle_sequence_to_rate_sequence(
                    raw_psi,
                    self.att_mpc.dt,
                    rate_limit=yaw_rate_limit_for_sequence,
                )

                seq = np.zeros((self.att_mpc.Np, 10), dtype=float)
                seq[:, 0] = np.asarray(raw_phi, dtype=float)
                seq[:, 1] = np.asarray(raw_theta, dtype=float)
                seq[:, 2] = raw_psi
                seq[:, 5] = raw_r

                self.raw_phi_ref = float(raw_phi[0])
                self.raw_theta_ref = float(raw_theta[0])
                self.shaped_phi_ref = float(raw_phi[0])
                self.shaped_theta_ref = float(raw_theta[0])
                self.shaped_p_ref = 0.0
                self.shaped_q_ref = 0.0
                self.coordinator_chi_ref = 0.0
                self.coordinator_ups_ref = 0.0
                return MMCUAVROS2Controller._finalize_attitude_reference_sequence(
                    self,
                    seq,
                    self.thrust_cmd,
                )

            # NDO 补偿必须在 AxisManeuverCoordinator 之前进入 raw 姿态参考。
            # 旧实现是在最终序列末端只给 phi/theta 加补偿，这会让 MPC 同时看到：
            #   1) 已补偿后的强扰动姿态目标；
            #   2) 仍按未补偿外环姿态生成的滑块参考/速度参考。
            # 3 m/s 横风日志中已出现 inner pitch 约 -15°、但 chi_ref 仍为正的
            # 矛盾信号，导致 NDO 角度补偿没有被滑块通道充分执行。
            # 因此正常自动场景先求 NDO 姿态偏置，再由 coordinator 统一生成
            # “姿态角 + 姿态角速度 + 滑块参考”的一致预测序列。
            ndo_comp_phi, ndo_comp_theta = MMCUAVROS2Controller._resolve_ndo_attitude_compensation(
                self,
                self.thrust_cmd,
            )
            if abs(ndo_comp_phi) > 1e-12 or abs(ndo_comp_theta) > 1e-12:
                roll_limit, pitch_limit = MMCUAVROS2Controller._reference_attitude_limits_for_ndo(self)
                comp_phi_seq, comp_theta_seq = MMCUAVROS2Controller._shape_ndo_compensation_against_outer_feedback(
                    self,
                    ndo_comp_phi,
                    ndo_comp_theta,
                    raw_phi,
                    raw_theta,
                )
                if comp_phi_seq.size > 0 and comp_theta_seq.size > 0:
                    MMCUAVROS2Controller._overwrite_ndo_log_total(
                        self,
                        float(comp_phi_seq[0]),
                        float(comp_theta_seq[0]),
                    )
                raw_phi = np.clip(np.asarray(raw_phi, dtype=float) + comp_phi_seq, -roll_limit, roll_limit)
                raw_theta = np.clip(np.asarray(raw_theta, dtype=float) + comp_theta_seq, -pitch_limit, pitch_limit)

            roll_seq = self.roll_axis_coordinator.build_reference_sequence(
                state=state_snapshot,
                thrust_cmd=self.thrust_cmd,
                raw_angle_now=raw_phi[0],
                raw_angle_future=raw_phi[1:],
            )
            pitch_seq = self.pitch_axis_coordinator.build_reference_sequence(
                state=state_snapshot,
                thrust_cmd=self.thrust_cmd,
                raw_angle_now=raw_theta[0],
                raw_angle_future=raw_theta[1:],
            )

            raw_r = angle_sequence_to_rate_sequence(
                raw_psi,
                self.att_mpc.dt,
                rate_limit=yaw_rate_limit_for_sequence,
            )
            seq = np.zeros((self.att_mpc.Np, 10), dtype=float)
            seq[:, 0] = roll_seq[:, 0]
            seq[:, 1] = pitch_seq[:, 1]
            seq[:, 2] = raw_psi
            seq[:, 3] = roll_seq[:, 3]
            seq[:, 4] = pitch_seq[:, 4]
            seq[:, 5] = raw_r
            seq[:, 6] = pitch_seq[:, 6]
            seq[:, 7] = pitch_seq[:, 7]
            seq[:, 8] = roll_seq[:, 8]
            seq[:, 9] = roll_seq[:, 9]

            self.raw_phi_ref = float(raw_phi[0])
            self.raw_theta_ref = float(raw_theta[0])
            self.shaped_phi_ref = float(seq[0, 0])
            self.shaped_theta_ref = float(seq[0, 1])
            self.shaped_p_ref = float(seq[0, 3])
            self.shaped_q_ref = float(seq[0, 4])
            self.coordinator_chi_ref = float(seq[0, 6])
            self.coordinator_ups_ref = float(seq[0, 8])

            return MMCUAVROS2Controller._finalize_attitude_reference_sequence(
                self,
                seq,
                self.thrust_cmd,
                apply_ndo=False,
            )

    def _planner_reference_is_effectively_static(self, t: float) -> bool:
        ref_future = self.planner.get_ref(t + self.att_mpc.dt)
        ref_future.psi = MMCUAVROS2Controller._resolve_auto_scene_yaw_target(
            self,
            t=t + self.att_mpc.dt,
            base_psi=ref_future.psi,
            current_psi=self.state.psi,
            base_ref=ref_future,
            mutate=False,
        )
        ref_future = MMCUAVROS2Controller._resolve_auto_scene_hover_point_lock(
            self,
            t=t + self.att_mpc.dt,
            base_ref=ref_future,
            state_snapshot=self.state,
            mutate=False,
        )
        return (
            abs(ref_future.x - self.ref_pos_now.x) <= 1e-6
            and abs(ref_future.y - self.ref_pos_now.y) <= 1e-6
            and abs(ref_future.z - self.ref_pos_now.z) <= 1e-6
            and abs(ref_future.vx - self.ref_pos_now.vx) <= 1e-5
            and abs(ref_future.vy - self.ref_pos_now.vy) <= 1e-5
            and abs(ref_future.vz - self.ref_pos_now.vz) <= 1e-5
            and abs(ref_future.ax - self.ref_pos_now.ax) <= 1e-4
            and abs(ref_future.ay - self.ref_pos_now.ay) <= 1e-4
            and abs(ref_future.az - self.ref_pos_now.az) <= 1e-4
            and abs(ref_future.psi - self.ref_pos_now.psi) <= 1e-6
        )

    def _legacy_pitch_step_snapshot(self) -> dict:
        return {
            "pitch_step_test_enabled": bool(
                getattr(self, "pitch_step_test_enabled", getattr(self, "attitude_step_test_enabled", False))
            ),
            "pitch_step_hover_hold_time": float(
                getattr(self, "pitch_step_hover_hold_time", getattr(self, "attitude_step_hover_hold_time", 4.0))
            ),
            "pitch_step_test_theta_ref": float(
                getattr(self, "pitch_step_test_theta_ref", getattr(self, "attitude_step_test_angle_ref", 0.0))
            ),
            "pitch_step_hover_z_tol": float(
                getattr(self, "pitch_step_hover_z_tol", getattr(self, "attitude_step_hover_z_tol", 0.15))
            ),
            "pitch_step_hover_speed_tol": float(
                getattr(self, "pitch_step_hover_speed_tol", getattr(self, "attitude_step_hover_speed_tol", 0.15))
            ),
            "pitch_step_hover_start_time": getattr(
                self, "pitch_step_hover_start_time", getattr(self, "attitude_step_hover_start_time", None)
            ),
            "pitch_step_test_start_time": getattr(
                self, "pitch_step_test_start_time", getattr(self, "attitude_step_test_start_time", None)
            ),
            "pitch_step_test_active": bool(
                getattr(self, "pitch_step_test_active", getattr(self, "attitude_step_test_active", False))
            ),
            "pitch_step_test_logged": bool(
                getattr(self, "pitch_step_test_logged", getattr(self, "attitude_step_test_logged", False))
            ),
            "pitch_step_test_theta_start": float(
                getattr(self, "pitch_step_test_theta_start", getattr(self, "attitude_step_test_pitch_start", 0.0))
            ),
        }

    def _sync_attitude_step_aliases_from_legacy(self, force: bool = False):
        current = MMCUAVROS2Controller._legacy_pitch_step_snapshot(self)
        shadow = getattr(self, "_pitch_step_legacy_shadow", None)
        has_changes = bool(force or shadow is None or current != shadow)
        if not has_changes:
            return

        self.attitude_step_test_enabled = bool(current["pitch_step_test_enabled"])
        self.attitude_step_hover_hold_time = float(current["pitch_step_hover_hold_time"])
        self.attitude_step_test_angle_ref = float(current["pitch_step_test_theta_ref"])
        self.attitude_step_hover_z_tol = float(current["pitch_step_hover_z_tol"])
        self.attitude_step_hover_speed_tol = float(current["pitch_step_hover_speed_tol"])
        self.attitude_step_hover_start_time = current["pitch_step_hover_start_time"]
        self.attitude_step_test_start_time = current["pitch_step_test_start_time"]
        self.attitude_step_test_active = bool(current["pitch_step_test_active"])
        self.attitude_step_test_logged = bool(current["pitch_step_test_logged"])
        self.attitude_step_test_pitch_start = float(current["pitch_step_test_theta_start"])
        self._pitch_step_legacy_shadow = dict(current)

    def _sync_legacy_pitch_step_aliases(self):
        self.pitch_step_test_enabled = bool(
            getattr(self, "attitude_step_test_enabled", getattr(self, "pitch_step_test_enabled", False))
        )
        self.pitch_step_hover_hold_time = float(
            getattr(self, "attitude_step_hover_hold_time", getattr(self, "pitch_step_hover_hold_time", 4.0))
        )
        self.pitch_step_test_theta_ref = float(
            getattr(self, "attitude_step_test_angle_ref", getattr(self, "pitch_step_test_theta_ref", 0.0))
        )
        self.pitch_step_hover_z_tol = float(
            getattr(self, "attitude_step_hover_z_tol", getattr(self, "pitch_step_hover_z_tol", 0.15))
        )
        self.pitch_step_hover_speed_tol = float(
            getattr(self, "attitude_step_hover_speed_tol", getattr(self, "pitch_step_hover_speed_tol", 0.15))
        )
        self.pitch_step_hover_start_time = getattr(
            self, "attitude_step_hover_start_time", getattr(self, "pitch_step_hover_start_time", None)
        )
        self.pitch_step_test_start_time = getattr(
            self, "attitude_step_test_start_time", getattr(self, "pitch_step_test_start_time", None)
        )
        self.pitch_step_test_active = bool(
            getattr(self, "attitude_step_test_active", getattr(self, "pitch_step_test_active", False))
        )
        self.pitch_step_test_logged = bool(
            getattr(self, "attitude_step_test_logged", getattr(self, "pitch_step_test_logged", False))
        )
        self.pitch_step_test_theta_start = float(
            getattr(self, "attitude_step_test_pitch_start", getattr(self, "pitch_step_test_theta_start", 0.0))
        )
        self._pitch_step_legacy_shadow = MMCUAVROS2Controller._legacy_pitch_step_snapshot(self)

    def _attitude_step_hover_target_z(self) -> float:
        return float(
            getattr(
                getattr(self, "wind_config_summary", None),
                "hover_target_z",
                getattr(getattr(self, "ref_pos_now", None), "z", self.state.z),
            )
        )

    def _is_attitude_step_hover_ready(self, target_z: Optional[float] = None) -> bool:
        if target_z is None:
            target_z = MMCUAVROS2Controller._attitude_step_hover_target_z(self)
        speed_tol = float(getattr(self, "attitude_step_hover_speed_tol", 0.15))
        z_tol = float(getattr(self, "attitude_step_hover_z_tol", 0.15))
        return (
            abs(self.state.z - float(target_z)) <= z_tol
            and abs(self.state.vx) <= speed_tol
            and abs(self.state.vy) <= speed_tol
            and abs(self.state.vz) <= speed_tol
        )

    def _is_auto_scene_yaw_hover_ready(self, target_z: Optional[float] = None) -> bool:
        if target_z is None:
            target_z = float(getattr(self, "auto_scene_target_z", self._attitude_step_hover_target_z()))
        horizontal_speed_tol = min(float(getattr(self, "attitude_step_hover_speed_tol", 0.15)), 0.12)
        vertical_speed_tol = min(float(getattr(self, "attitude_step_hover_speed_tol", 0.15)), 0.05)
        z_tol = min(float(getattr(self, "attitude_step_hover_z_tol", 0.15)), 0.08)
        return (
            abs(self.state.z - float(target_z)) <= z_tol
            and abs(self.state.vx) <= horizontal_speed_tol
            and abs(self.state.vy) <= horizontal_speed_tol
            and abs(self.state.vz) <= vertical_speed_tol
        )

    def _resolve_auto_scene_figure_eight_start_time(
        self,
        t: float,
        *,
        mutate: bool = True,
    ) -> Optional[float]:
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if not MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
            return None

        takeoff_transition_time = max(
            float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
            1e-6,
        )
        hold_time = max(float(getattr(self, "auto_scene_hover_hold_time", 4.0)), 0.0)

        if float(t) < takeoff_transition_time:
            if mutate:
                self.auto_scene_figure_eight_hover_start_time = None
            return None

        trigger_time = getattr(self, "auto_scene_figure_eight_trigger_time", None)
        if trigger_time is not None:
            return float(trigger_time)

        hover_ready = True
        if hasattr(self, "state"):
            hover_ready = bool(
                MMCUAVROS2Controller._is_auto_scene_yaw_hover_ready(
                    self,
                    getattr(self, "auto_scene_target_z", None),
                )
            )

        if not hover_ready:
            if mutate:
                self.auto_scene_figure_eight_hover_start_time = None
            return None

        hover_start_time = getattr(self, "auto_scene_figure_eight_hover_start_time", None)
        if hover_start_time is None:
            hover_start_time = float(t)
            if mutate:
                self.auto_scene_figure_eight_hover_start_time = hover_start_time

        if (float(t) - float(hover_start_time)) < hold_time:
            return None

        trigger_time = float(t)
        if mutate:
            self.auto_scene_figure_eight_trigger_time = trigger_time
            if (
                bool(getattr(self, "mission_initialized", False))
                and not bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False))
                and hasattr(self, "planner")
                and hasattr(self, "state")
            ):
                traj_x, traj_y, traj_z = MMCUAVROS2Controller._default_trajectory_functions(self)
                entry_transition_time = max(
                    float(getattr(self, "auto_scene_figure_eight_ramp_duration", 5.0)),
                    1e-3,
                )
                self.planner.set_mission(
                    traj_x,
                    traj_y,
                    traj_z,
                    current_t=trigger_time,
                    current_state=State6(**self.state.__dict__),
                    trans_time=entry_transition_time,
                )
                self.auto_scene_figure_eight_entry_replanned = True
                self.auto_scene_figure_eight_entry_end_time = float(trigger_time + entry_transition_time)
                self.auto_scene_figure_eight_phase_bias = 0.0
                self.auto_scene_figure_eight_phase_bias_locked = False
                if bool(getattr(self, "rviz_trajectory_enabled", False)):
                    self.rviz_reference_path_published = False
                    self.rviz_reference_path_msg = None
            if not getattr(self, "auto_scene_figure_eight_logged", False):
                plane_label = MMCUAVROS2Controller._auto_scene_figure_eight_plane_label(scene_mode)
                lateral_axis = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_axis(scene_mode) or "y"
                lateral_amplitude = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_amplitude(
                    self,
                    scene_mode,
                )
                self.get_logger().info(
                    f"{plane_label} figure-eight scene triggered after continuous hover-ready hold: "
                    f"hover_hold_time={hold_time:.2f}s, "
                    f"trigger_time={trigger_time:.2f}s, "
                    f"hover_z={float(getattr(self, 'auto_scene_target_z', 1.5)):.2f}, "
                    f"{lateral_axis}_amplitude={lateral_amplitude:.2f}, "
                    f"z_amplitude={float(getattr(self, 'auto_scene_figure_eight_z_amplitude', 0.675)):.3f}, "
                    f"forward_tilt_deg={MMCUAVROS2Controller._auto_scene_figure_eight_forward_tilt_deg(self, scene_mode):.1f}, "
                    f"yaw_ref_mode={MMCUAVROS2Controller._auto_scene_yaw_ref_mode(self)}"
                )
                self.auto_scene_figure_eight_logged = True
        return trigger_time

    def _maybe_finalize_auto_scene_figure_eight_phase_bias(self, t: float) -> None:
        if not bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False)):
            return
        if bool(getattr(self, "auto_scene_figure_eight_phase_bias_locked", False)):
            return
        trigger_time = getattr(self, "auto_scene_figure_eight_trigger_time", None)
        bridge_end_time = getattr(self, "auto_scene_figure_eight_entry_end_time", None)
        if trigger_time is None or bridge_end_time is None:
            return
        if float(t) < float(bridge_end_time):
            return

        raw_phase = max(
            float(
                MMCUAVROS2Controller._query_auto_scene_adaptive_phase_time(
                    self,
                    float(t),
                    activation_time=float(trigger_time),
                )
            ) - float(trigger_time),
            0.0,
        )
        desired_phase = max(float(bridge_end_time) - float(trigger_time), 0.0)
        self.auto_scene_figure_eight_phase_bias = float(desired_phase - raw_phase)
        self.auto_scene_figure_eight_phase_bias_locked = True
        if bool(getattr(self, "rviz_trajectory_enabled", False)):
            self.rviz_reference_path_published = False
            self.rviz_reference_path_msg = None

    def _filter_rotor_tau_z_command(self, thrust_cmd: float, tau_z_cmd: float, dt: float) -> float:
        tau_limited = float(self.mixer.clamp_tau_z(thrust_cmd, tau_z_cmd))
        tau_tc = float(getattr(self, "rotor_tau_z_filter_time_constant", 0.0))
        if tau_tc <= 1e-9 or dt <= 1e-9:
            filtered = tau_limited
        else:
            alpha = clamp(float(dt) / (tau_tc + float(dt)), 0.0, 1.0)
            filtered = float(self.filtered_tau_z_cmd + alpha * (tau_limited - self.filtered_tau_z_cmd))
        filtered = float(self.mixer.clamp_tau_z(thrust_cmd, filtered))
        self.filtered_tau_z_cmd = filtered
        return filtered

    def _shape_mpc_slider_commands(
        self,
        raw_chi_cmd: float,
        raw_ups_cmd: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        对 MPC 的滑块位置设定值做发布前整形。

        MPC 仍然是唯一的滑块命令来源；这里不引入第二套控制律，只把
        raw setpoint 变成 Gazebo 关节控制器和实际滑块速度能跟上的命令。
        这样可以避免 CSV 中看到的 ±0.1 m 级设定值高频换向把实体滑块
        长时间打到 ±0.25 m/s 速度上限，却没有形成有效位移。
        """
        limit = max(0.0, float(getattr(getattr(self, "att_mpc", None), "u2_lim", P.u2_lim)))
        raw_chi_cmd = clamp(float(raw_chi_cmd), -limit, limit)
        raw_ups_cmd = clamp(float(raw_ups_cmd), -limit, limit)
        if not bool(getattr(self, "slider_command_shaping_enabled", True)):
            return raw_chi_cmd, raw_ups_cmd

        tau = max(0.0, float(getattr(self, "slider_command_shape_tau", 0.06)))
        rate_limit = max(0.0, float(getattr(self, "slider_command_rate_limit", 1.8 * P.slider_vel_max)))
        chi_cmd = shape_slider_command(
            current=float(getattr(self, "chi_cmd", 0.0)),
            target=raw_chi_cmd,
            dt=float(dt),
            tau=tau,
            rate_limit=rate_limit,
            limit=limit,
        )
        ups_cmd = shape_slider_command(
            current=float(getattr(self, "ups_cmd", 0.0)),
            target=raw_ups_cmd,
            dt=float(dt),
            tau=tau,
            rate_limit=rate_limit,
            limit=limit,
        )
        return chi_cmd, ups_cmd

    def _retarget_thrust_for_inner_reference(
        self,
        outer_thrust_cmd: float,
        outer_phi_ref: float,
        outer_theta_ref: float,
    ) -> Tuple[float, float]:
        """
        根据最终送入内环的有效姿态参考补足垂向推力余量。

        外环 PID 仍负责给出名义推力与姿态；NDO 或参考整形只改变内环
        实际要跟踪的姿态时，若有效倾角大于外环名义倾角，就需要按
        cos(phi)cos(theta) 的几何关系补一点总推力，否则“姿态补偿变强”
        会伴随垂向推力不足。为了不破坏名义场景，本函数只增加推力，
        不因为内环参考暂时更小而削减外环推力。
        """
        outer_thrust_cmd = float(outer_thrust_cmd)
        self.thrust_retarget_ratio = 1.0
        if (
            not bool(getattr(self, "inner_thrust_retarget_enabled", True))
            or outer_thrust_cmd <= 1e-9
        ):
            return outer_thrust_cmd, 1.0

        inner_phi_ref = float(getattr(self, "inner_phi_ref", outer_phi_ref))
        inner_theta_ref = float(getattr(self, "inner_theta_ref", outer_theta_ref))
        values = (outer_phi_ref, outer_theta_ref, inner_phi_ref, inner_theta_ref)
        if not all(math.isfinite(float(value)) for value in values):
            return outer_thrust_cmd, 1.0

        cos_outer = max(math.cos(float(outer_phi_ref)) * math.cos(float(outer_theta_ref)), 0.1)
        cos_inner = max(math.cos(inner_phi_ref) * math.cos(inner_theta_ref), 0.1)
        ratio = max(1.0, cos_outer / cos_inner)
        thrust = clamp(
            outer_thrust_cmd * ratio,
            self.P.thrust_min,
            self.P.thrust_max,
        )
        applied_ratio = thrust / max(outer_thrust_cmd, 1e-9)
        self.thrust_retarget_ratio = float(applied_ratio)
        return float(thrust), float(applied_ratio)

    def _resolve_auto_scene_yaw_target(
        self,
        t: float,
        base_psi: float,
        current_psi: float,
        *,
        base_ref: Optional[RefPos] = None,
        mutate: bool = True,
    ) -> float:
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
            trigger_time = MMCUAVROS2Controller._resolve_auto_scene_figure_eight_start_time(
                self,
                t,
                mutate=mutate,
            )
            if trigger_time is None:
                return float(base_psi)
            base_hover_psi = getattr(self, "auto_scene_figure_eight_base_psi", None)
            if base_hover_psi is None:
                base_hover_psi = current_psi
                if mutate:
                    self.auto_scene_figure_eight_base_psi = float(base_hover_psi)
            base_hover_psi = float(base_hover_psi)
            yaw_mode = MMCUAVROS2Controller._auto_scene_yaw_ref_mode(self)
            if yaw_mode == "fixed":
                if mutate:
                    self.auto_scene_figure_eight_last_yaw_ref = base_hover_psi
                    self.auto_scene_figure_eight_last_yaw_ref_time = float(t)
                return base_hover_psi

            if base_ref is None and hasattr(self, "planner"):
                try:
                    base_ref = self.planner.get_ref(float(t))
                except Exception:
                    base_ref = None
            speed_floor = max(float(getattr(self, "auto_scene_yaw_ref_speed_floor", 0.05)), 1e-6)
            last_yaw_ref = getattr(self, "auto_scene_figure_eight_last_yaw_ref", None)
            if last_yaw_ref is None:
                last_yaw_ref = base_hover_psi
            last_yaw_ref = float(last_yaw_ref)
            if base_ref is None:
                return last_yaw_ref
            entry_end_time = (
                MMCUAVROS2Controller._resolve_auto_scene_figure_eight_entry_end_time(self)
                if bool(getattr(self, "auto_scene_figure_eight_entry_replanned", False))
                else None
            )
            bridge_active = (
                entry_end_time is not None
                and float(trigger_time) <= float(t) < float(entry_end_time)
            )
            if bridge_active:
                merge_ref = None
                if hasattr(self, "planner"):
                    try:
                        merge_ref = self.planner.get_ref(float(entry_end_time))
                    except Exception:
                        merge_ref = None
                if merge_ref is None:
                    target_psi = last_yaw_ref
                else:
                    merge_speed = math.hypot(float(merge_ref.vx), float(merge_ref.vy))
                    if merge_speed <= speed_floor:
                        target_merge_psi = base_hover_psi
                    else:
                        target_merge_psi = wrap_angle_pi(
                            math.atan2(float(merge_ref.vy), float(merge_ref.vx))
                        )
                    bridge_alpha = quintic_smoothstep(
                        (float(t) - float(trigger_time))
                        / max(float(entry_end_time) - float(trigger_time), 1e-6)
                    )
                    target_psi = blend_angle_near(
                        float(base_hover_psi),
                        float(target_merge_psi),
                        bridge_alpha,
                    )
            else:
                horizontal_speed = math.hypot(float(base_ref.vx), float(base_ref.vy))
                if horizontal_speed <= speed_floor:
                    target_psi = last_yaw_ref
                else:
                    target_psi = wrap_angle_pi(math.atan2(float(base_ref.vy), float(base_ref.vx)))
            yaw_ref_rate_limit = max(
                0.0,
                float(getattr(self, "auto_scene_yaw_ref_rate_limit_rad_s", 0.0)),
            )
            if yaw_ref_rate_limit > 0.0 and mutate and not bridge_active:
                last_yaw_ref_time = getattr(self, "auto_scene_figure_eight_last_yaw_ref_time", None)
                if last_yaw_ref_time is None:
                    dt_limit = max(float(getattr(self, "outer_dt", 1.0 / 25.0)), 1e-6)
                else:
                    dt_limit = max(float(t) - float(last_yaw_ref_time), 1e-6)
                target_psi = slew_limit_angle(
                    target_psi,
                    anchor=last_yaw_ref,
                    max_delta=yaw_ref_rate_limit * dt_limit,
                )
            if mutate:
                self.auto_scene_figure_eight_last_yaw_ref = float(target_psi)
                self.auto_scene_figure_eight_last_yaw_ref_time = float(t)
            return float(target_psi)

        if scene_mode == "hover_to_point_yaw_step_hold":
            takeoff_transition_time = max(
                float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
                1e-6,
            )
            hold_time = max(float(getattr(self, "auto_scene_hover_hold_time", 4.0)), 0.0)
            ramp_start_time = takeoff_transition_time + hold_time

            if float(t) < ramp_start_time:
                return float(base_psi)

            base_hover_psi = getattr(self, "auto_scene_yaw_step_base_psi", None)
            if base_hover_psi is None:
                base_hover_psi = current_psi
            base_hover_psi = float(base_hover_psi)
            target_psi = wrap_angle_pi(
                base_hover_psi + math.radians(float(getattr(self, "auto_scene_yaw_step_deg", 90.0)))
            )
            ramp_duration = MMCUAVROS2Controller._resolved_auto_scene_move_duration(self)
            if mutate:
                if getattr(self, "auto_scene_yaw_step_trigger_time", None) is None:
                    self.auto_scene_yaw_step_trigger_time = float(ramp_start_time)
                self.auto_scene_yaw_step_base_psi = base_hover_psi
                self.auto_scene_yaw_step_target_psi = target_psi
                if not getattr(self, "auto_scene_yaw_step_logged", False):
                    self.get_logger().info(
                        "Coupled point+yaw scene triggered after scheduled hover hold: "
                        f"target_x={float(getattr(self, 'auto_scene_target_x', 0.0)):.2f}, "
                        f"target_y={float(getattr(self, 'auto_scene_target_y', 0.0)):.2f}, "
                        f"yaw_step_deg={float(getattr(self, 'auto_scene_yaw_step_deg', 90.0)):.1f}, "
                        f"effective_coupled_duration={ramp_duration:.2f}s, "
                        f"base_psi_deg={math.degrees(base_hover_psi):.1f}, "
                        f"target_psi_deg={math.degrees(target_psi):.1f}"
                    )
                    self.auto_scene_yaw_step_logged = True

            if float(t) >= (float(ramp_start_time) + ramp_duration):
                return float(target_psi)

            blend = quintic_smoothstep((float(t) - float(ramp_start_time)) / ramp_duration)
            yaw_delta = wrap_angle_pi(target_psi - base_hover_psi)
            return float(wrap_angle_pi(base_hover_psi + blend * yaw_delta))

        if scene_mode != "hover_to_yaw_step_hold":
            return float(base_psi)

        takeoff_transition_time = max(
            float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
            1e-6,
        )
        hold_time = max(float(getattr(self, "auto_scene_hover_hold_time", 4.0)), 0.0)

        if float(t) < takeoff_transition_time:
            return float(base_psi)

        ramp_start_time = getattr(self, "auto_scene_yaw_step_trigger_time", None)
        if ramp_start_time is None:
            hover_ready = True
            if hasattr(self, "state"):
                hover_ready = bool(
                    MMCUAVROS2Controller._is_auto_scene_yaw_hover_ready(
                        self,
                        getattr(self, "auto_scene_target_z", None),
                    )
                )

            if not hover_ready:
                if mutate:
                    self.auto_scene_yaw_step_hover_start_time = None
                return float(base_psi)

            hover_start_time = getattr(self, "auto_scene_yaw_step_hover_start_time", None)
            if hover_start_time is None:
                hover_start_time = float(t)
                if mutate:
                    self.auto_scene_yaw_step_hover_start_time = hover_start_time

            if (float(t) - float(hover_start_time)) < hold_time:
                return float(base_psi)

            ramp_start_time = float(t)
            if mutate:
                self.auto_scene_yaw_step_trigger_time = ramp_start_time

        base_hover_psi = getattr(self, "auto_scene_yaw_step_base_psi", None)
        if base_hover_psi is None:
            base_hover_psi = current_psi
        base_hover_psi = float(base_hover_psi)
        target_psi = wrap_angle_pi(
            base_hover_psi + math.radians(float(getattr(self, "auto_scene_yaw_step_deg", 90.0)))
        )
        if mutate:
            self.auto_scene_yaw_step_base_psi = base_hover_psi
            self.auto_scene_yaw_step_target_psi = target_psi
            if not getattr(self, "auto_scene_yaw_step_logged", False):
                self.get_logger().info(
                    "Yaw step scene triggered after hover hold: "
                    f"yaw_step_deg={float(getattr(self, 'auto_scene_yaw_step_deg', 90.0)):.1f}, "
                    f"yaw_ramp_duration={float(getattr(self, 'auto_scene_yaw_ramp_duration', 4.0)):.2f}s, "
                    f"base_psi_deg={math.degrees(base_hover_psi):.1f}, "
                    f"target_psi_deg={math.degrees(target_psi):.1f}"
                )
                self.auto_scene_yaw_step_logged = True
        ramp_duration = max(float(getattr(self, "auto_scene_yaw_ramp_duration", 4.0)), 1e-6)
        if float(t) >= (float(ramp_start_time) + ramp_duration):
            return float(target_psi)

        blend = quintic_smoothstep((float(t) - float(ramp_start_time)) / ramp_duration)
        yaw_delta = wrap_angle_pi(target_psi - base_hover_psi)
        return float(wrap_angle_pi(base_hover_psi + blend * yaw_delta))

    def _resolve_auto_scene_yaw_ref(
        self,
        t: float,
        base_psi: float,
        current_psi: float,
        *,
        base_ref: Optional[RefPos] = None,
    ) -> float:
        return float(
            MMCUAVROS2Controller._resolve_auto_scene_yaw_target(
                self,
                t=t,
                base_psi=base_psi,
                current_psi=current_psi,
                base_ref=base_ref,
                mutate=True,
            )
        )

    def _resolve_auto_scene_hover_point_lock(
        self,
        t: float,
        base_ref: RefPos,
        state_snapshot: State6,
        *,
        mutate: bool = True,
    ) -> RefPos:
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if scene_mode != "hover_to_yaw_step_hold":
            return RefPos(**base_ref.__dict__)

        ramp_start_time = getattr(self, "auto_scene_yaw_step_trigger_time", None)
        if ramp_start_time is None or float(t) < float(ramp_start_time):
            return RefPos(**base_ref.__dict__)

        hold_x = getattr(self, "auto_scene_yaw_step_hold_x", None)
        hold_y = getattr(self, "auto_scene_yaw_step_hold_y", None)
        hold_z = getattr(self, "auto_scene_yaw_step_hold_z", None)
        if hold_x is None or hold_y is None or hold_z is None:
            hold_x = float(state_snapshot.x)
            hold_y = float(state_snapshot.y)
            hold_z = float(base_ref.z)
            if mutate:
                self.auto_scene_yaw_step_hold_x = hold_x
                self.auto_scene_yaw_step_hold_y = hold_y
                self.auto_scene_yaw_step_hold_z = hold_z

        return RefPos(
            x=float(hold_x),
            y=float(hold_y),
            z=float(hold_z),
            vx=0.0,
            vy=0.0,
            vz=0.0,
            ax=0.0,
            ay=0.0,
            az=0.0,
            psi=float(base_ref.psi),
        )

    def _resolved_rotor_lower_command_scale(self) -> float:
        base_scale = float(getattr(self, "rotor_lower_command_scale", 1.0))
        delayed_scale = float(getattr(self, "rotor_lower_command_scale_after_hover", base_scale))
        hold_time = max(float(getattr(self, "rotor_lower_command_scale_after_hover_hold_time", 0.0)), 0.0)

        if hold_time <= 1e-9 or abs(delayed_scale - base_scale) <= 1e-9:
            return base_scale

        if getattr(self, "rotor_lower_command_scale_event_active", False):
            return delayed_scale

        target_z = float(getattr(self, "auto_scene_target_z", 1.5))
        hover_ready = MMCUAVROS2Controller._is_attitude_step_hover_ready(self, target_z=target_z)
        if not hover_ready:
            self.rotor_lower_command_scale_hover_start_time = None
            return base_scale

        t = self._elapsed_sec()
        if self.rotor_lower_command_scale_hover_start_time is None:
            self.rotor_lower_command_scale_hover_start_time = t
            return base_scale

        if (t - self.rotor_lower_command_scale_hover_start_time) < hold_time:
            return base_scale

        self.rotor_lower_command_scale_event_active = True
        if not getattr(self, "rotor_lower_command_scale_event_logged", False):
            self.get_logger().warning(
                "RotorPhysics validation override triggered after stable hover hold: "
                f"lower-rotor command scale switched from {base_scale:.3f} "
                f"to {delayed_scale:.3f}."
            )
            self.rotor_lower_command_scale_event_logged = True

        return delayed_scale

    def _resolve_open_loop_rotor_override(
        self,
        thrust_cmd_current: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
        if scene_mode != "hover_to_open_loop_rotor_diff":
            self.auto_scene_open_loop_rotor_hover_start_time = None
            return None

        if getattr(self, "auto_scene_open_loop_rotor_active", False):
            return (
                float(self.auto_scene_open_loop_thrust_real),
                float(self.auto_scene_open_loop_tau_z_real),
                float(self.auto_scene_open_loop_w1_sq),
                float(self.auto_scene_open_loop_w2_sq),
            )

        target_z = float(getattr(self, "auto_scene_target_z", 1.5))
        hover_ready = MMCUAVROS2Controller._is_attitude_step_hover_ready(self, target_z=target_z)
        if not hover_ready:
            self.auto_scene_open_loop_rotor_hover_start_time = None
            return None

        t = self._elapsed_sec()
        hold_time = max(float(getattr(self, "auto_scene_hover_hold_time", 4.0)), 0.0)
        if self.auto_scene_open_loop_rotor_hover_start_time is None:
            self.auto_scene_open_loop_rotor_hover_start_time = t
            return None

        if (t - self.auto_scene_open_loop_rotor_hover_start_time) < hold_time:
            return None

        thrust_real, tau_z_real, w1_sq, w2_sq = self.mixer.mix(
            float(thrust_cmd_current),
            float(getattr(self, "auto_scene_open_loop_tau_z_step", -0.03)),
        )
        self.auto_scene_open_loop_thrust_real = float(thrust_real)
        self.auto_scene_open_loop_tau_z_real = float(tau_z_real)
        self.auto_scene_open_loop_w1_sq = float(w1_sq)
        self.auto_scene_open_loop_w2_sq = float(w2_sq)
        self.auto_scene_open_loop_rotor_active = True

        if not getattr(self, "auto_scene_open_loop_rotor_logged", False):
            self.get_logger().warning(
                "Open-loop rotor-differential scene triggered after hover hold: "
                f"tau_z_step={float(getattr(self, 'auto_scene_open_loop_tau_z_step', -0.03)):.4f} N*m, "
                f"upper_speed_cmd={math.sqrt(max(float(w2_sq), 0.0)):.3f}, "
                f"lower_speed_cmd={math.sqrt(max(float(w1_sq), 0.0)):.3f}"
            )
            self.auto_scene_open_loop_rotor_logged = True

        return (
            float(self.auto_scene_open_loop_thrust_real),
            float(self.auto_scene_open_loop_tau_z_real),
            float(self.auto_scene_open_loop_w1_sq),
            float(self.auto_scene_open_loop_w2_sq),
        )

    def _apply_attitude_step_test_override(
        self,
        t: float,
        thrust_cmd: float,
        phi_auto: float,
        theta_auto: float,
        phi_ref: float,
        theta_ref: float,
    ) -> Tuple[float, float, float]:
        MMCUAVROS2Controller._sync_attitude_step_aliases_from_legacy(self, force=False)
        if not getattr(self, "attitude_step_test_enabled", False):
            MMCUAVROS2Controller._sync_legacy_pitch_step_aliases(self)
            return thrust_cmd, phi_ref, theta_ref

        axis = str(getattr(self, "attitude_step_test_axis", "pitch")).strip().lower()
        if axis not in ("pitch", "roll", "yaw"):
            axis = "pitch"

        if getattr(self, "attitude_step_test_active", False):
            start_time = getattr(self, "attitude_step_test_start_time", None)
            if start_time is None:
                start_time = t
                self.attitude_step_test_start_time = t
            phi_target, theta_target = MMCUAVROS2Controller._attitude_step_targets_at_time(
                self,
                t=t,
                fallback_phi=phi_ref,
                fallback_theta=theta_ref,
            )
            if axis != "yaw":
                thrust_cmd = self.outer.retarget_thrust_for_attitude(
                    thrust_cmd=thrust_cmd,
                    phi_old=phi_auto,
                    theta_old=theta_auto,
                    phi_new=phi_target,
                    theta_new=theta_target,
                )
            MMCUAVROS2Controller._sync_legacy_pitch_step_aliases(self)
            return thrust_cmd, phi_target, theta_target

        target_z = MMCUAVROS2Controller._attitude_step_hover_target_z(self)
        hover_ready = MMCUAVROS2Controller._is_attitude_step_hover_ready(self, target_z=target_z)
        if not hover_ready:
            self.attitude_step_hover_start_time = None
            MMCUAVROS2Controller._sync_legacy_pitch_step_aliases(self)
            return thrust_cmd, phi_ref, theta_ref

        if self.attitude_step_hover_start_time is None:
            self.attitude_step_hover_start_time = t
            MMCUAVROS2Controller._sync_legacy_pitch_step_aliases(self)
            return thrust_cmd, phi_ref, theta_ref

        if (t - self.attitude_step_hover_start_time) < float(getattr(self, "attitude_step_hover_hold_time", 4.0)):
            MMCUAVROS2Controller._sync_legacy_pitch_step_aliases(self)
            return thrust_cmd, phi_ref, theta_ref

        self.attitude_step_test_active = True
        self.attitude_step_test_start_time = t
        self.attitude_step_test_roll_start = float(phi_ref)
        self.attitude_step_test_pitch_start = float(theta_ref)
        phi_target = float(phi_ref)
        theta_target = float(theta_ref)
        if axis != "yaw":
            thrust_cmd = self.outer.retarget_thrust_for_attitude(
                thrust_cmd=thrust_cmd,
                phi_old=phi_auto,
                theta_old=theta_auto,
                phi_new=phi_target,
                theta_new=theta_target,
            )
        if not getattr(self, "attitude_step_test_logged", False):
            self.get_logger().info(
                "Attitude-step test triggered after hover hold: "
                f"axis={axis}, "
                f"+/-{math.degrees(float(getattr(self, 'attitude_step_test_angle_ref', 0.0))):.1f} deg, "
                f"hold {float(getattr(self, 'attitude_step_hold_time', 3.0)):.1f}s, "
                f"recover {float(getattr(self, 'attitude_step_recovery_time', 2.0)):.1f}s."
            )
            self.attitude_step_test_logged = True
        MMCUAVROS2Controller._sync_legacy_pitch_step_aliases(self)
        return thrust_cmd, phi_target, theta_target

    def _apply_pitch_step_test_override(
        self,
        t: float,
        thrust_cmd: float,
        phi_auto: float,
        theta_auto: float,
        phi_ref: float,
        theta_ref: float,
    ) -> Tuple[float, float, float]:
        MMCUAVROS2Controller._sync_attitude_step_aliases_from_legacy(self, force=True)
        return MMCUAVROS2Controller._apply_attitude_step_test_override(
            self,
            t=t,
            thrust_cmd=thrust_cmd,
            phi_auto=phi_auto,
            theta_auto=theta_auto,
            phi_ref=phi_ref,
            theta_ref=theta_ref,
        )

    def _run_outer_controller_step(self, t: float):
        with optional_lock(self):
            manual_xy_enabled = getattr(self, "manual_xy_enabled", True)
            now_sec = self._now_sec()
            MMCUAVROS2Controller._resolve_auto_scene_figure_eight_start_time(self, t)
            state_snapshot = State6(**self.state.__dict__)
            provisional_ref = self.planner.get_ref(t)
            MMCUAVROS2Controller._update_auto_scene_adaptive_phase_state(
                self,
                t,
                state_snapshot,
                provisional_ref,
            )
            MMCUAVROS2Controller._maybe_finalize_auto_scene_figure_eight_phase_bias(self, t)
            base_ref = self.planner.get_ref(t)
            outer_ax_limit = float(getattr(self.outer, "ax_cmd_limit", 0.55))
            outer_ay_limit = float(getattr(self.outer, "ay_cmd_limit", outer_ax_limit))
            requested_scene_accel_limit = max(
                float(getattr(self, "auto_scene_horizontal_accel_limit", outer_ax_limit)),
                1e-6,
            )
            use_scene_accel_limit = (
                not manual_xy_enabled
                and not getattr(self, "attitude_step_test_enabled", False)
                and (
                    str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
                    in ("hover_to_point_hold", "hover_to_point_yaw_step_hold")
                    or MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(
                        getattr(self, "auto_scene_mode", "hover_only")
                    )
                )
            )
            if use_scene_accel_limit:
                # auto_scene_horizontal_accel_limit 是轨迹规划/名义机动的加速度预算，
                # 不能反向削弱外环 PID 的抗风扰控制权限。否则无 NDO 对照组在
                # 3 m/s 横风下会被限制在约 5° 俯仰并稳定平移跑飞，失去对照意义。
                self.outer.ax_cmd_limit = max(outer_ax_limit, requested_scene_accel_limit)
                self.outer.ay_cmd_limit = max(outer_ay_limit, requested_scene_accel_limit)
            if manual_xy_enabled:
                manual_status = self.manual_xy.advance(
                    state_snapshot,
                    base_ref=base_ref,
                    now_sec=now_sec,
                    dt=self.outer_dt,
                )

                mode_changed = self.manual_xy.mode != self.last_manual_mode
                if mode_changed:
                    self.outer.reset_horizontal_hold_state()
            else:
                manual_status = ManualXYStatus(
                    hold_x=base_ref.x,
                    hold_y=base_ref.y,
                    mode_label="DISABLED",
                )
                mode_changed = False

            outer_ref = self._build_outer_ref(base_ref, manual_status)
            outer_ref.psi = MMCUAVROS2Controller._resolve_auto_scene_yaw_ref(
                self,
                t=t,
                base_psi=outer_ref.psi,
                current_psi=state_snapshot.psi,
                base_ref=outer_ref,
            )
            outer_ref = MMCUAVROS2Controller._resolve_auto_scene_hover_point_lock(
                self,
                t=t,
                base_ref=outer_ref,
                state_snapshot=state_snapshot,
                mutate=True,
            )
            attitude_mapping_psi = MMCUAVROS2Controller._resolve_attitude_mapping_psi(
                self,
                ref_psi=outer_ref.psi,
                current_psi=state_snapshot.psi,
            )
            try:
                thrust_auto, phi_auto, theta_auto, psi_ref = self.outer.compute(
                    state_snapshot,
                    outer_ref,
                    attitude_mapping_psi=attitude_mapping_psi,
                )
            finally:
                self.outer.ax_cmd_limit = outer_ax_limit
                self.outer.ay_cmd_limit = outer_ay_limit
            thrust_cmd = thrust_auto
            phi_ref = phi_auto
            theta_ref = theta_auto
            if (
                manual_xy_enabled
                and manual_status.phi_ref_override is not None
                and manual_status.theta_ref_override is not None
            ):
                phi_ref = manual_status.phi_ref_override
                theta_ref = manual_status.theta_ref_override
                thrust_cmd_ref = self.outer.retarget_thrust_for_attitude(
                    thrust_cmd=thrust_auto,
                    phi_old=phi_auto,
                    theta_old=theta_auto,
                    phi_new=phi_ref,
                    theta_new=theta_ref,
                )
                thrust_cmd_actual = self.outer.retarget_thrust_for_attitude(
                    thrust_cmd=thrust_auto,
                    phi_old=phi_auto,
                    theta_old=theta_auto,
                    phi_new=state_snapshot.phi,
                    theta_new=state_snapshot.theta,
                )
                tilt_ref_mag = math.hypot(phi_ref, theta_ref)
                tilt_actual_mag = math.hypot(state_snapshot.phi, state_snapshot.theta)
                tilt_mag = max(tilt_ref_mag, tilt_actual_mag)
                tilt_ratio = clamp(tilt_mag / max(self.manual_xy.max_tilt, 1e-6), 0.0, 1.5)
                thrust_margin = 1.0 + self.manual_tilt_thrust_margin * tilt_ratio
                thrust_cmd = clamp(
                    max(thrust_cmd_ref, thrust_cmd_actual) * thrust_margin,
                    self.P.thrust_min,
                    self.P.thrust_max,
                )
            thrust_cmd, phi_ref, theta_ref = MMCUAVROS2Controller._apply_attitude_step_test_override(
                self,
                t=t,
                thrust_cmd=thrust_cmd,
                phi_auto=phi_auto,
                theta_auto=theta_auto,
                phi_ref=phi_ref,
                theta_ref=theta_ref,
            )
            psi_ref = MMCUAVROS2Controller._attitude_step_psi_target_at_time(self, t, psi_ref)

            if manual_xy_enabled and self.manual_xy.ready != self.last_manual_ready:
                if self.manual_xy.ready:
                    self.get_logger().info(
                        "Manual XY unlocked. Hovered near 1.5 m long enough to accept keyboard input."
                    )
                self.last_manual_ready = self.manual_xy.ready

            if manual_xy_enabled and mode_changed:
                if self.manual_xy.mode == ManualXYMode.COMMAND:
                    self.get_logger().info("Manual XY engaged. Keyboard input now drives bounded body-speed targets.")
                elif self.manual_xy.mode == ManualXYMode.BRAKE:
                    self.get_logger().info(
                        "Manual XY released. Zero-speed braking with higher brake tilt authority is active."
                    )
                elif self.manual_xy.mode == ManualXYMode.HOLD:
                    self.get_logger().info("Manual XY hold relocked at the new hover point.")
                self.last_manual_mode = self.manual_xy.mode

            self.manual_status = manual_status
            self.ref_pos_now = outer_ref
            self.thrust_cmd_outer = thrust_cmd
            self.thrust_cmd = thrust_cmd
            self.thrust_retarget_ratio = 1.0
            self.phi_ref = phi_ref
            self.theta_ref = theta_ref
            self.psi_ref = psi_ref
            MMCUAVROS2Controller._publish_rviz_actual_path(self, state_snapshot)
            MMCUAVROS2Controller._publish_rviz_vehicle_marker(self, state_snapshot)
            self.outer_ran_once = True

    def _try_init_mission(self):
        with optional_lock(self):
            if self.mission_initialized:
                return
            if not (self.imu_ready and self.odom_ready):
                return
            if not self.preflight_centering_started:
                self.preflight_centering_started = True
                self.center_hold_start_ns = None
                self.last_inner_tick = self.get_clock().now()
                self.get_logger().info("Sensors ready. Holding sliders at center before takeoff...")
                return
            if not self.preflight_centering_confirmed:
                return

            # 当 IMU 和里程计都就绪后，启动任务时间基准
            self.start_time = self.get_clock().now()
            self.last_inner_tick = self.start_time

            traj_x, traj_y, traj_z = self._default_trajectory_functions()
            transition_time = max(
                float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
                1e-6,
            )
            self.planner.set_mission(
                traj_x,
                traj_y,
                traj_z,
                current_t=0.0,
                current_state=State6(**self.state.__dict__),
                trans_time=transition_time,
            )
            self.mission_initialized = True
            self.outer_ran_once = False
            self.auto_scene_yaw_step_hover_start_time = None
            self.auto_scene_yaw_step_base_psi = None
            self.auto_scene_yaw_step_target_psi = None
            self.auto_scene_yaw_step_trigger_time = None
            self.auto_scene_yaw_step_hold_x = None
            self.auto_scene_yaw_step_hold_y = None
            self.auto_scene_yaw_step_hold_z = None
            self.auto_scene_yaw_step_logged = False
            self.auto_scene_figure_eight_hover_start_time = None
            self.auto_scene_figure_eight_trigger_time = None
            self.auto_scene_figure_eight_logged = False
            self.auto_scene_figure_eight_base_psi = None
            self.auto_scene_figure_eight_last_yaw_ref = None
            self.auto_scene_figure_eight_last_yaw_ref_time = None
            self.auto_scene_figure_eight_entry_replanned = False
            self.auto_scene_figure_eight_entry_end_time = None
            self.auto_scene_figure_eight_phase_bias = 0.0
            self.auto_scene_figure_eight_phase_bias_locked = False
            self.rviz_actual_path_msg = MMCUAVROS2Controller._new_rviz_path_message(self)
            self.rviz_reference_path_msg = None
            self.rviz_reference_path_published = False
            self.rviz_vehicle_marker_array = None
            self.rviz_vehicle_marker_yaw = None
            MMCUAVROS2Controller._reset_auto_scene_adaptive_phase_state(self)
            self.filtered_tau_z_cmd = 0.0
            self.auto_scene_open_loop_rotor_hover_start_time = None
            self.auto_scene_open_loop_rotor_active = False
            self.auto_scene_open_loop_rotor_logged = False
            self.auto_scene_open_loop_w1_sq = 0.0
            self.auto_scene_open_loop_w2_sq = 0.0
            self.auto_scene_open_loop_thrust_real = 0.0
            self.auto_scene_open_loop_tau_z_real = 0.0
            self.wind_move_window_start_requested = bool(getattr(self, "wind_runtime_active", False))
            self.wind_move_window_stop_requested = False
            self.rotor_lower_command_scale_hover_start_time = None
            self.rotor_lower_command_scale_event_active = False
            self.rotor_lower_command_scale_event_logged = False
            MMCUAVROS2Controller._publish_rviz_reference_path_if_needed(self)
            scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
            requested_move_duration = max(
                float(getattr(self, "auto_scene_move_duration", 4.0)),
                1e-6,
            )
            resolved_move_duration = MMCUAVROS2Controller._resolved_auto_scene_move_duration(self)
            if (
                scene_mode in ("hover_to_point_hold", "hover_to_point_yaw_step_hold")
                and resolved_move_duration > (requested_move_duration + 1e-9)
            ):
                self.get_logger().warning(
                    "Auto scene move duration is shorter than the current horizontal acceleration budget; "
                    f"using {resolved_move_duration:.2f}s instead of requested {requested_move_duration:.2f}s."
                )
            if scene_mode == "hover_only":
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"hover_z={float(getattr(self, 'auto_scene_target_z', 1.5)):.2f}"
                )
            elif scene_mode == "hover_to_point_hold":
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"hover_hold_time={float(getattr(self, 'auto_scene_hover_hold_time', 4.0)):.2f}s, "
                    f"auto_scene_move_duration={resolved_move_duration:.2f}s, "
                    f"auto_scene_horizontal_accel_limit={float(getattr(self, 'auto_scene_horizontal_accel_limit', 0.8)):.2f}m/s^2, "
                    f"target_x={float(getattr(self, 'auto_scene_target_x', 0.0)):.2f}, "
                    f"target_y={float(getattr(self, 'auto_scene_target_y', 0.0)):.2f}, "
                    f"target_z={float(getattr(self, 'auto_scene_target_z', 1.5)):.2f}"
                )
            elif scene_mode == "hover_to_yaw_step_hold":
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"hover_hold_time={float(getattr(self, 'auto_scene_hover_hold_time', 4.0)):.2f}s, "
                    f"yaw_step_deg={float(getattr(self, 'auto_scene_yaw_step_deg', 90.0)):.1f}"
                )
            elif scene_mode == "hover_to_point_yaw_step_hold":
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"hover_hold_time={float(getattr(self, 'auto_scene_hover_hold_time', 4.0)):.2f}s, "
                    f"auto_scene_move_duration={resolved_move_duration:.2f}s, "
                    f"auto_scene_horizontal_accel_limit={float(getattr(self, 'auto_scene_horizontal_accel_limit', 0.8)):.2f}m/s^2, "
                    f"target_x={float(getattr(self, 'auto_scene_target_x', 0.0)):.2f}, "
                    f"target_y={float(getattr(self, 'auto_scene_target_y', 0.0)):.2f}, "
                    f"yaw_step_deg={float(getattr(self, 'auto_scene_yaw_step_deg', 90.0)):.1f}, "
                    f"yaw_ramp_duration={float(getattr(self, 'auto_scene_yaw_ramp_duration', 4.0)):.2f}s"
                )
            elif scene_mode == "hover_to_open_loop_rotor_diff":
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"hover_hold_time={float(getattr(self, 'auto_scene_hover_hold_time', 4.0)):.2f}s, "
                    f"open_loop_tau_z_step={float(getattr(self, 'auto_scene_open_loop_tau_z_step', -0.03)):.4f}N*m"
                )
            elif MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
                plane_label = MMCUAVROS2Controller._auto_scene_figure_eight_plane_label(scene_mode)
                lateral_axis = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_axis(scene_mode) or "y"
                lateral_amplitude = MMCUAVROS2Controller._auto_scene_figure_eight_lateral_amplitude(
                    self,
                    scene_mode,
                )
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"figure_eight_plane={plane_label}, "
                    f"hover_hold_time={float(getattr(self, 'auto_scene_hover_hold_time', 4.0)):.2f}s, "
                    f"hover_z={float(getattr(self, 'auto_scene_target_z', 1.5)):.2f}, "
                    f"outer_z_thrust_scale={float(getattr(getattr(self, 'outer', None), 'z_thrust_scale', 1.0)):.3f}, "
                    f"{lateral_axis}_amplitude={lateral_amplitude:.2f}, "
                    f"z_amplitude={float(getattr(self, 'auto_scene_figure_eight_z_amplitude', 0.675)):.3f}, "
                    f"forward_tilt_deg={MMCUAVROS2Controller._auto_scene_figure_eight_forward_tilt_deg(self, scene_mode):.1f}, "
                    f"figure_period={float(getattr(self, 'auto_scene_figure_eight_period', 16.0)):.2f}s, "
                    f"ramp_duration={float(getattr(self, 'auto_scene_figure_eight_ramp_duration', 5.0)):.2f}s, "
                    f"adaptive_phase_enabled={bool(getattr(self, 'auto_scene_adaptive_phase_enabled', False))}, "
                    f"adaptive_phase_min_rate={float(getattr(self, 'auto_scene_adaptive_phase_min_rate', 0.45)):.2f}, "
                    f"yaw_ref_mode={MMCUAVROS2Controller._auto_scene_yaw_ref_mode(self)}"
                )
            else:
                self.get_logger().info(
                    "Mission initialized. Control loops are active. "
                    f"actuation_backend={getattr(self, 'actuation_backend', 'rotor_physics')}, "
                    f"auto_scene_mode={scene_mode}, "
                    f"auto_scene_move_duration={resolved_move_duration:.2f}s, "
                    f"auto_scene_horizontal_accel_limit={float(getattr(self, 'auto_scene_horizontal_accel_limit', 0.8)):.2f}m/s^2"
                )

    def _sliders_within_center_tolerance(self) -> bool:
        return (
            abs(self.state.chi) <= self.center_pos_tol
            and abs(self.state.ups) <= self.center_pos_tol
            and abs(self.state.chi_d) <= self.center_vel_tol
            and abs(self.state.ups_d) <= self.center_vel_tol
        )

    def _sliders_centered_for_takeoff(self, now_ns: int) -> bool:
        if not self._joint_feedback_fresh():
            self.preflight_centering_degraded_mode = True
            if not self.preflight_centering_degraded_logged:
                self.get_logger().warning(
                    "JointState feedback is missing or stale during preflight centering; "
                    "using estimated slider state for degraded takeoff gating."
                )
                self.preflight_centering_degraded_logged = True
        else:
            self.preflight_centering_degraded_mode = False

        if not self._sliders_within_center_tolerance():
            self.center_hold_start_ns = None
            return False

        if self.center_hold_start_ns is None:
            self.center_hold_start_ns = now_ns
            return False

        hold_elapsed = (now_ns - self.center_hold_start_ns) * 1e-9
        return hold_elapsed >= self.center_hold_time

    def _run_preflight_centering(self, dt: float):
        with optional_lock(self):
            self.thrust_cmd = 0.0
            self.phi_ref = 0.0
            self.theta_ref = 0.0
            self.psi_ref = 0.0
            self.tau_z_cmd = 0.0
            self.chi_cmd = 0.0
            self.ups_cmd = 0.0
            self.last_u_mpc = np.zeros(3)

            now_ns = self.get_clock().now().nanoseconds
            if self._sliders_centered_for_takeoff(now_ns) and not self.preflight_centering_confirmed:
                self.preflight_centering_confirmed = True
                mode_label = "estimated state / degraded mode" if self.preflight_centering_degraded_mode else "joint feedback"
                self.get_logger().info(
                    "Preflight slider centering confirmed "
                    f"({mode_label}). chi={self.state.chi:.4f} m, ups={self.state.ups:.4f} m"
                )

            zero_force = np.zeros(3, dtype=float)
            zero_torque = np.zeros(3, dtype=float)
            self._publish_actuation(0.0, 0.0, 0.0, 0.0, dt, zero_force, zero_torque)
            self._publish_slider_commands(0.0, 0.0)

    def _joint_feedback_fresh(self) -> bool:
        with optional_lock(self):
            if self.last_joint_update_ns is None:
                return False
            age_sec = (self.get_clock().now().nanoseconds - self.last_joint_update_ns) * 1e-9
            return age_sec < 0.2

    def _publish_runtime_wind_command(
        self,
        enable_wind: bool,
        velocity_world: Tuple[float, float, float],
        source: str,
    ) -> WindCommand:
        self.wind_command_seq += 1
        self.wind_command_source = source
        self.wind_command_pending = True
        self.wind_status_publish_ok = False
        self.wind_status_detail = ""

        command = WindCommand()
        command.command_seq = int(self.wind_command_seq)
        command.stamp = self.get_clock().now().to_msg()
        command.world_name = self.wind_world_name
        command.enable_wind = bool(enable_wind)
        command.linear_velocity_world.x = float(velocity_world[0])
        command.linear_velocity_world.y = float(velocity_world[1])
        command.linear_velocity_world.z = float(velocity_world[2])
        command.source = source
        self.wind_command_pub.publish(command)
        return command

    def _auto_scene_move_window_bounds(self) -> Tuple[float, float]:
        start = max(
            float(getattr(self, "auto_scene_takeoff_transition_time", 5.0)),
            1e-6,
        ) + max(float(getattr(self, "auto_scene_hover_hold_time", 4.0)), 0.0)
        end = start + max(MMCUAVROS2Controller._resolved_auto_scene_move_duration(self), 1e-6)
        return float(start), float(end)

    def _update_move_window_wind_state(self, t: float):
        summary = self.wind_config_summary
        if not summary.config_valid or not summary.enable_wind:
            return
        if getattr(self, "wind_command_pending", False):
            return

        start, end = MMCUAVROS2Controller._auto_scene_move_window_bounds(self)
        velocity_world = (
            summary.wind_vx_world,
            summary.wind_vy_world,
            summary.wind_vz_world,
        )
        if (
            float(t) >= start
            and float(t) < end
            and not bool(getattr(self, "wind_move_window_start_requested", False))
        ):
            self.wind_move_window_start_requested = True
            command = self._publish_runtime_wind_command(
                True,
                velocity_world,
                source="move_window_start",
            )
            self.get_logger().info(
                "Published move-window wind start command and waiting for bridge ack: "
                f"seq={command.command_seq}, window=({start:.2f}, {end:.2f})s, "
                f"v=({summary.wind_vx_world:.2f}, {summary.wind_vy_world:.2f}, "
                f"{summary.wind_vz_world:.2f}) m/s"
            )
            return

        if (
            float(t) >= end
            and bool(getattr(self, "wind_move_window_start_requested", False))
            and not bool(getattr(self, "wind_move_window_stop_requested", False))
        ):
            self.wind_move_window_stop_requested = True
            command = self._publish_runtime_wind_command(
                False,
                (0.0, 0.0, 0.0),
                source="move_window_stop",
            )
            self.get_logger().info(
                "Published move-window wind stop command and waiting for bridge ack: "
                f"seq={command.command_seq}, window=({start:.2f}, {end:.2f})s"
            )

    def _update_runtime_wind_state(self, t: float):
        summary = self.wind_config_summary
        if summary.activation_mode == "move_window":
            MMCUAVROS2Controller._update_move_window_wind_state(self, t)
            return
        if (
            not summary.config_valid
            or not summary.enable_wind
            or self.wind_runtime_active
            or getattr(self, "wind_command_pending", False)
            or int(getattr(self, "wind_command_seq", 0)) > 0
        ):
            return

        if summary.activation_mode == "hover_hold":
            if not hover_ready_for_wind_activation(self.state, summary):
                self.wind_hover_hold_start_time = None
                return
            if self.wind_hover_hold_start_time is None:
                self.wind_hover_hold_start_time = t
                return
            if (t - self.wind_hover_hold_start_time) < summary.hover_hold_time:
                return
        elif summary.activation_mode != "immediate":
            return

        velocity_world = (
            summary.wind_vx_world,
            summary.wind_vy_world,
            summary.wind_vz_world,
        )
        command = self._publish_runtime_wind_command(
            True,
            velocity_world,
            source="hover_hold",
        )
        self.get_logger().info(
            "Published hover-hold wind command and waiting for bridge ack: "
            f"seq={command.command_seq}, v=({summary.wind_vx_world:.2f}, "
            f"{summary.wind_vy_world:.2f}, {summary.wind_vz_world:.2f}) m/s"
        )

    def _peek_ndo_log_row(self, current_thrust: float):
        d_force_hat, d_torque_hat = self.ndo.get_disturbance_estimates()

        saved_force = (self.ndo.last_comp_phi_f, self.ndo.last_comp_theta_f)
        saved_torque = (self.ndo.last_comp_phi_r, self.ndo.last_comp_theta_r)
        try:
            (comp_phi_total, comp_theta_total), (comp_phi_f, comp_theta_f), (
                comp_phi_r,
                comp_theta_r,
            ) = self.ndo.get_combined_compensation(self.state, current_thrust)
        finally:
            self.ndo.last_comp_phi_f, self.ndo.last_comp_theta_f = saved_force
            self.ndo.last_comp_phi_r, self.ndo.last_comp_theta_r = saved_torque

        return [
            float(d_force_hat[0]),
            float(d_force_hat[1]),
            float(d_force_hat[2]),
            float(d_torque_hat[0]),
            float(d_torque_hat[1]),
            float(d_torque_hat[2]),
            math.degrees(comp_phi_f),
            math.degrees(comp_theta_f),
            math.degrees(comp_phi_r),
            math.degrees(comp_theta_r),
            math.degrees(comp_phi_total),
            math.degrees(comp_theta_total),
        ]

    def _estimate_slider_states_if_needed(self, dt: float):
        """
        如果无法获得联合反馈，则保留原有的二阶滑块执行器模型
        作为MPC/NDO内部一致性的状态估计器。
        """
        with optional_lock(self):
            if self._joint_feedback_fresh():
                return

            wn = self.P.wn_mass
            zt = self.P.zeta_mass

            chi_dd = wn * wn * (self.chi_cmd - self.state.chi) - 2.0 * zt * wn * self.state.chi_d
            ups_dd = wn * wn * (self.ups_cmd - self.state.ups) - 2.0 * zt * wn * self.state.ups_d

            self.state.chi_d += chi_dd * dt
            self.state.ups_d += ups_dd * dt
            self.state.chi += self.state.chi_d * dt
            self.state.ups += self.state.ups_d * dt

            self.state.chi, self.state.chi_d = clamp_slider_state(
                self.state.chi,
                self.state.chi_d,
                self.P.u2_lim,
                self.P.slider_vel_max,
            )
            self.state.ups, self.state.ups_d = clamp_slider_state(
                self.state.ups,
                self.state.ups_d,
                self.P.u2_lim,
                self.P.slider_vel_max,
            )

    # ----------------------------
    # 外环（25 Hz）
    # ----------------------------
    def outer_loop_cb(self):
        with optional_lock(self):
            mission_initialized = self.mission_initialized
        if not mission_initialized:
            return

        t = self._elapsed_sec()
        self._run_outer_controller_step(t)
        with optional_lock(self):
            self.outer_ran_once = True

    # ----------------------------
    # 内环（100 Hz）
    # ----------------------------
    def inner_loop_cb(self):
        if not self._sensors_ready():
            return

        inner_exec_start = time.perf_counter()
        now = self.get_clock().now()
        dt_raw = (now - self.last_inner_tick).nanoseconds * 1e-9
        self.last_inner_tick = now
        dt = dt_raw
        if dt <= 1e-6:
            dt = self.inner_dt
        elif dt > 0.2:
            elapsed_sec = self._elapsed_sec()
            if (elapsed_sec - self.last_large_inner_dt_warn_time) >= 1.0:
                self.get_logger().warning(
                    f"Inner loop raw dt spike detected ({dt_raw:.3f}s); "
                    f"falling back to nominal control dt={self.inner_dt:.3f}s for this cycle."
                )
                self.last_large_inner_dt_warn_time = elapsed_sec
            dt = self.inner_dt
        self.last_inner_dt_raw = max(dt_raw, 0.0)
        self.last_inner_dt = self.last_inner_dt_raw if dt_raw > 1e-6 else dt
        self.last_inner_exec_dt = 0.0
        self.last_inner_mpc_dt = 0.0
        self.last_inner_observer_dt = 0.0
        self.last_inner_drag_dt = 0.0
        self.last_inner_publish_dt = 0.0
        self.last_inner_ref_build_dt = 0.0
        self.last_inner_log_dt = 0.0

        self._estimate_slider_states_if_needed(dt)

        with optional_lock(self):
            mission_initialized = self.mission_initialized
        if not mission_initialized:
            self._run_preflight_centering(dt)
            with optional_lock(self):
                preflight_centering_confirmed = self.preflight_centering_confirmed
            if preflight_centering_confirmed:
                self._try_init_mission()
            return

        with optional_lock(self):
            t = self._elapsed_sec()
            self._update_runtime_wind_state(t)
            if not self.outer_ran_once:
                self._run_outer_controller_step(t)
            state_snapshot = State6(**self.state.__dict__)
            thrust_cmd = float(self.thrust_cmd)
            thrust_cmd_outer = float(getattr(self, "thrust_cmd_outer", thrust_cmd))
            observer_input = Input6(
                thrust=thrust_cmd,
                tau_z=float(self.tau_z_cmd),
                chi_cmd=float(self.chi_cmd),
                ups_cmd=float(self.ups_cmd),
            )
            last_u_mpc = self.last_u_mpc.copy()

        observer_start = time.perf_counter()
        self._ndo_current_cycle_valid = False
        if getattr(self, "ndo_enabled", False):
            try:
                self.ndo.update(state_snapshot, observer_input, dt)
                self._ndo_current_cycle_valid = True
                self.last_inner_observer_dt = time.perf_counter() - observer_start
            except Exception as exc:
                self._warn_ndo_once_per_second(f"NDO update failed; disabling compensation for this cycle: {exc}")
                self._last_ndo_log_row = self._zero_ndo_log_row
                self.last_inner_observer_dt = 0.0
        else:
            self._last_ndo_log_row = self._zero_ndo_log_row
            self.last_inner_observer_dt = 0.0

        ref_build_start = time.perf_counter()
        ref_att_seq = self._build_attitude_reference_sequence(t)
        self.last_inner_ref_build_dt = time.perf_counter() - ref_build_start

        with optional_lock(self):
            thrust_cmd_current, thrust_retarget_ratio = (
                MMCUAVROS2Controller._retarget_thrust_for_inner_reference(
                    self,
                    outer_thrust_cmd=thrust_cmd_outer,
                    outer_phi_ref=float(self.phi_ref),
                    outer_theta_ref=float(self.theta_ref),
                )
            )
            self.thrust_cmd = float(thrust_cmd_current)
            self.thrust_retarget_ratio = float(thrust_retarget_ratio)

        mpc_start = time.perf_counter()
        tau_z_cmd, raw_chi_cmd, raw_ups_cmd = self.att_mpc.compute(
            t,
            state_snapshot,
            thrust_cmd_current,
            ref_att_seq,
            last_u_opt=last_u_mpc,
        )
        self.last_inner_mpc_dt = time.perf_counter() - mpc_start

        raw_chi_cmd = clamp(raw_chi_cmd, -self.att_mpc.u2_lim, self.att_mpc.u2_lim)
        raw_ups_cmd = clamp(raw_ups_cmd, -self.att_mpc.u2_lim, self.att_mpc.u2_lim)
        chi_cmd, ups_cmd = MMCUAVROS2Controller._shape_mpc_slider_commands(
            self,
            raw_chi_cmd,
            raw_ups_cmd,
            dt,
        )
        with optional_lock(self):
            self.tau_z_cmd = float(
                MMCUAVROS2Controller._filter_rotor_tau_z_command(
                    self,
                    thrust_cmd_current,
                    tau_z_cmd,
                    dt,
                )
            )
            self.raw_chi_cmd = float(raw_chi_cmd)
            self.raw_ups_cmd = float(raw_ups_cmd)
            self.chi_cmd = float(chi_cmd)
            self.ups_cmd = float(ups_cmd)
            self.last_u_mpc[:] = (self.tau_z_cmd, self.chi_cmd, self.ups_cmd)
            state_after_mpc = State6(**self.state.__dict__)
            phi_ref = float(self.phi_ref)
            theta_ref = float(self.theta_ref)
            psi_ref = float(self.psi_ref)
            manual_status = ManualXYStatus(**self.manual_status.__dict__)

        thrust_real, tau_z_real, w1_sq, w2_sq = self.mixer.mix(thrust_cmd_current, self.tau_z_cmd)
        open_loop_rotor_override = MMCUAVROS2Controller._resolve_open_loop_rotor_override(
            self,
            thrust_cmd_current=thrust_cmd_current,
        )
        if open_loop_rotor_override is not None:
            thrust_real, tau_z_real, w1_sq, w2_sq = open_loop_rotor_override

        ndo_log_row = self._last_ndo_log_row

        drag_start = time.perf_counter()
        drag_force, damping_torque, _ = calculate_aerodynamic_forces(state_after_mpc, self.P)
        self.last_inner_drag_dt = time.perf_counter() - drag_start

        # 恢复原有的脉冲补偿发布路径，并把缩放相关诊断写入黑匣子。
        publish_start = time.perf_counter()
        with optional_lock(self):
            self._publish_actuation(thrust_real, tau_z_real, w1_sq, w2_sq, dt, drag_force, damping_torque)
            self._publish_slider_commands(self.chi_cmd, self.ups_cmd)
            self._publish_euler_degrees()
        self.last_inner_publish_dt = time.perf_counter() - publish_start

        # 记录当前时间步的数据
        log_start = time.perf_counter()
        with optional_lock(self):
            state_for_log = State6(**self.state.__dict__)
            ref_pos_now = RefPos(**self.ref_pos_now.__dict__)
            scene_mode = str(getattr(self, "auto_scene_mode", "hover_only")).strip().lower()
            if MMCUAVROS2Controller._is_auto_scene_figure_eight_mode(scene_mode):
                self.last_scene_hover_ready = MMCUAVROS2Controller._is_auto_scene_yaw_hover_ready(
                    self,
                    getattr(self, "auto_scene_target_z", None),
                )
            else:
                self.last_scene_hover_ready = MMCUAVROS2Controller._is_attitude_step_hover_ready(self)
            horiz_speed = math.hypot(state_for_log.vx, state_for_log.vy)
            _, Tv_b = T_transformation(state_for_log)
            self._body_velocity_buffer[:] = (state_for_log.vx, state_for_log.vy, state_for_log.vz)
            v_body = Tv_b @ self._body_velocity_buffer
            rotor_max_speed = max(float(getattr(self, "rotor_max_speed_rad_s", 450.0)), 1e-9)
            upper_speed_for_util = abs(float(self.upper_rotor_actual))
            lower_speed_for_util = abs(float(self.lower_rotor_actual))
            if upper_speed_for_util <= 1e-9:
                upper_speed_for_util = abs(float(self.upper_rotor_cmd))
            if lower_speed_for_util <= 1e-9:
                lower_speed_for_util = abs(float(self.lower_rotor_cmd))
            rotor_speed_util_actual = max(upper_speed_for_util, lower_speed_for_util) / rotor_max_speed
            rotor_thrust_util_est = float(self.rotor_total_thrust_est) / max(float(self.P.thrust_max), 1e-9)
            rotor_thrust_margin_est = 1.0 - rotor_thrust_util_est
            row = [
                t,
                state_for_log.x, state_for_log.y, state_for_log.z,
                ref_pos_now.x, ref_pos_now.y, ref_pos_now.z,
                state_for_log.vx, state_for_log.vy, state_for_log.vz,
                math.degrees(state_for_log.phi), math.degrees(state_for_log.theta), math.degrees(state_for_log.psi),
                math.degrees(state_for_log.p), math.degrees(state_for_log.q), math.degrees(state_for_log.r),
                math.degrees(phi_ref), math.degrees(theta_ref), math.degrees(psi_ref),
                math.degrees(phi_ref - state_for_log.phi),
                math.degrees(theta_ref - state_for_log.theta),
                float(getattr(self.outer, "sum_ex", 0.0)),
                float(getattr(self.outer, "sum_ey", 0.0)),
                float(getattr(self.outer, "sum_ez", 0.0)),
                float(self.last_scene_hover_ready),
                float(getattr(self, "last_adaptive_phase_active", False)),
                float(getattr(self, "last_adaptive_phase_time", 0.0)),
                float(getattr(self, "last_adaptive_phase_rate", 1.0)),
                float(getattr(self, "last_adaptive_phase_metric", 0.0)),
                manual_status.mode_label,
                manual_status.manual_input_forward,
                manual_status.manual_input_lateral,
                manual_status.hold_x,
                manual_status.hold_y,
                horiz_speed,
                math.degrees(manual_status.brake_phi_ref),
                math.degrees(manual_status.brake_theta_ref),
                float(v_body[0]),
                float(v_body[1]),
                self.actuation_backend,
                self.yaw_control_mode,
                str(getattr(self, "auto_scene_mode", "hover_only")),
                float(getattr(self, "ndo_enabled", False)),
                float(self.P.m_b),
                float(self.P.m),
                float(self.P.M),
                float(self.P.mu),
                float(self.P.wn_mass),
                float(self.P.zeta_mass),
                str(getattr(self, "world_sdf_path", "")),
                MMCUAVROS2Controller._auto_scene_yaw_ref_mode(self),
                MMCUAVROS2Controller._auto_scene_figure_eight_forward_tilt_deg(
                    self,
                    getattr(self, "auto_scene_mode", "hover_only"),
                ),
                self.upper_rotor_cmd,
                self.lower_rotor_cmd,
                self.upper_rotor_actual,
                self.lower_rotor_actual,
                abs(self.upper_rotor_cmd),
                abs(self.lower_rotor_cmd),
                abs(self.upper_rotor_actual),
                abs(self.lower_rotor_actual),
                rotor_speed_rad_s_to_rpm(abs(self.upper_rotor_cmd)),
                rotor_speed_rad_s_to_rpm(abs(self.lower_rotor_cmd)),
                rotor_speed_rad_s_to_rpm(abs(self.upper_rotor_actual)),
                rotor_speed_rad_s_to_rpm(abs(self.lower_rotor_actual)),
                self.rotor_total_thrust_est,
                self.rotor_tau_z_est,
                rotor_speed_util_actual,
                rotor_thrust_util_est,
                rotor_thrust_margin_est,
                rotor_max_speed,
                float(getattr(self, "thrust_cmd_outer", thrust_cmd_current)),
                thrust_cmd_current,
                float(getattr(self, "thrust_retarget_ratio", 1.0)),
                self.tau_z_cmd,
                self.last_inner_dt,
                0.0,
                self.last_inner_mpc_dt,
                self.att_mpc.last_model_update_dt,
                self.att_mpc.last_qp_build_dt,
                self.att_mpc.last_solver_setup_dt,
                self.att_mpc.last_solver_solve_dt,
                float(self.att_mpc.last_model_reused),
                self.att_mpc.last_lpv_reuse_count,
                self.att_mpc._lpv_reuse_max_skips,
                self.att_mpc.Np,
                self.att_mpc.Np,
                self.att_mpc.last_refresh_reason_mask,
                self.last_inner_observer_dt,
                self.last_inner_drag_dt,
                self.last_inner_publish_dt,
                self.last_inner_ref_build_dt,
                0.0,
                self.last_wrench_dt_for_scale,
                self.last_wrench_scale,
                self.last_force_z_world_total,
                self.last_force_z_world_published,
                math.degrees(self.raw_phi_ref),
                math.degrees(self.raw_theta_ref),
                math.degrees(self.shaped_phi_ref),
                math.degrees(self.shaped_theta_ref),
                math.degrees(self.shaped_p_ref),
                math.degrees(self.shaped_q_ref),
                math.degrees(float(getattr(self, "inner_phi_ref", 0.0))),
                math.degrees(float(getattr(self, "inner_theta_ref", 0.0))),
                math.degrees(float(getattr(self, "inner_psi_ref", 0.0))),
                self.coordinator_chi_ref,
                self.raw_chi_cmd,
                self.chi_cmd, state_for_log.chi,
                state_for_log.chi_d,
                self.coordinator_ups_ref,
                self.raw_ups_cmd,
                self.ups_cmd, state_for_log.ups,
                state_for_log.ups_d,
                *self.wind_log_row,
                self.wind_command_seq,
                self.wind_command_source,
                float(self.wind_command_pending),
                float(self.wind_status_publish_ok),
                self.wind_status_detail,
                float(self.wind_runtime_active),
                self.wind_activation_time,
                *ndo_log_row,
            ]
            self.flight_data_log.append(row)
        self.last_inner_log_dt = time.perf_counter() - log_start
        self.last_inner_exec_dt = time.perf_counter() - inner_exec_start
        row[FLIGHT_LOG_COLUMN_INDEX['Inner_exec_dt']] = self.last_inner_exec_dt
        row[FLIGHT_LOG_COLUMN_INDEX['Inner_log_dt']] = self.last_inner_log_dt

    # ----------------------------
    # 发布器
    # ----------------------------
    def _cached_float64_message(self, attr_name: str) -> Float64:
        msg = getattr(self, attr_name, None)
        if msg is None:
            msg = Float64()
            setattr(self, attr_name, msg)
        return msg

    def _cached_vector3_message(self, attr_name: str) -> Vector3:
        msg = getattr(self, attr_name, None)
        if msg is None:
            msg = Vector3()
            setattr(self, attr_name, msg)
        return msg

    def _cached_entity_wrench_message(self) -> EntityWrench:
        msg = getattr(self, "_cached_wrench_msg", None)
        if msg is None:
            msg = EntityWrench()
            if not hasattr(msg, "entity"):
                msg.entity = SimpleNamespace()
            if not hasattr(msg.entity, "name"):
                msg.entity.name = ""
            if not hasattr(msg.entity, "type"):
                msg.entity.type = getattr(Entity, "LINK", 0)
            if not hasattr(msg, "wrench"):
                msg.wrench = SimpleNamespace()
            if not hasattr(msg.wrench, "force"):
                msg.wrench.force = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            if not hasattr(msg.wrench, "torque"):
                msg.wrench.torque = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self._cached_wrench_msg = msg
        return msg

    def _cached_actuators_message(self) -> Actuators:
        msg = getattr(self, "_cached_motor_speed_msg", None)
        if msg is None:
            msg = Actuators()
            if not hasattr(msg, "header"):
                msg.header = SimpleNamespace(stamp=None, frame_id="")
            if not hasattr(msg, "velocity"):
                msg.velocity = []
            if not hasattr(msg, "position"):
                msg.position = []
            if not hasattr(msg, "normalized"):
                msg.normalized = []
            self._cached_motor_speed_msg = msg
        return msg

    def _publish_euler_degrees(self):
        msg = MMCUAVROS2Controller._cached_vector3_message(self, "_cached_euler_msg")
        msg.x = math.degrees(self.state.phi)
        msg.y = math.degrees(self.state.theta)
        msg.z = math.degrees(self.state.psi)
        self.pub_euler_deg.publish(msg)

    def _publish_control_wrench(self, thrust_real: float, tau_z_real: float, dt_actual: float,
                                drag_force: np.ndarray, damping_torque: np.ndarray):
        msg = MMCUAVROS2Controller._cached_entity_wrench_message(self)
        msg.entity.name = "mmc_uav::base_link"
        msg.entity.type = getattr(Entity, "LINK", 0)

        force_world, torque_world = resolve_body_wrench_to_world(
            self.state,
            thrust_real,
            tau_z_real,
            drag_force,
            damping_torque,
        )

        # Gazebo Harmonic 每 1 ms 会清空一次 ApplyLinkWrench，
        # 因此当前先恢复“按实际发布周期补偿脉冲面积”的旧基线，
        # 同时把缩放量记录到黑匣子里用于下一轮定位。
        self.wrench_dt_for_scale, scale = update_wrench_scale(
            dt_actual=dt_actual,
            prev_dt_for_scale=self.wrench_dt_for_scale,
        )

        self.last_wrench_dt_for_scale = self.wrench_dt_for_scale
        self.last_wrench_scale = scale
        self.last_force_z_world_total = float(force_world[2])
        self.last_force_z_world_published = float(force_world[2] * scale)

        msg.wrench.force.x = float(force_world[0] * scale)
        msg.wrench.force.y = float(force_world[1] * scale)
        msg.wrench.force.z = float(force_world[2] * scale)

        msg.wrench.torque.x = float(torque_world[0] * scale)
        msg.wrench.torque.y = float(torque_world[1] * scale)
        msg.wrench.torque.z = float(torque_world[2] * scale)

        self.pub_wrench.publish(msg)

    def _publish_aux_wrench(self, dt_actual: float, drag_force: np.ndarray, damping_torque: np.ndarray):
        msg = MMCUAVROS2Controller._cached_entity_wrench_message(self)
        msg.entity.name = "mmc_uav::base_link"
        msg.entity.type = getattr(Entity, "LINK", 0)

        force_world, torque_world = resolve_aux_wrench_to_world(drag_force, damping_torque)

        self.wrench_dt_for_scale, scale = update_wrench_scale(
            dt_actual=dt_actual,
            prev_dt_for_scale=self.wrench_dt_for_scale,
        )

        self.last_wrench_dt_for_scale = self.wrench_dt_for_scale
        self.last_wrench_scale = scale
        self.last_force_z_world_total = float(force_world[2])
        self.last_force_z_world_published = float(force_world[2] * scale)

        msg.wrench.force.x = float(force_world[0] * scale)
        msg.wrench.force.y = float(force_world[1] * scale)
        msg.wrench.force.z = float(force_world[2] * scale)
        msg.wrench.torque.x = float(torque_world[0] * scale)
        msg.wrench.torque.y = float(torque_world[1] * scale)
        msg.wrench.torque.z = float(torque_world[2] * scale)
        self.pub_wrench.publish(msg)

    def _shape_rotor_speed_command(
        self,
        current: float,
        target: float,
        dt_actual: float,
    ) -> float:
        max_speed = max(float(getattr(self, "rotor_max_speed_rad_s", 450.0)), 1.0)
        target = clamp(float(target), 0.0, max_speed)
        current = clamp(float(current), 0.0, max_speed)
        dt = max(float(dt_actual), 0.0)
        if dt <= 1e-9:
            return target

        if target >= current:
            tau = float(getattr(self, "rotor_motor_time_constant_up", 0.0))
        else:
            tau = float(getattr(self, "rotor_motor_time_constant_down", 0.0))
        if tau > 1e-9:
            alpha = clamp(dt / (tau + dt), 0.0, 1.0)
            shaped = current + alpha * (target - current)
        else:
            shaped = target

        rate_limit = float(getattr(self, "rotor_motor_rate_limit_rad_s2", 0.0))
        if rate_limit > 0.0:
            max_delta = rate_limit * dt
            shaped = current + clamp(shaped - current, -max_delta, max_delta)
        return clamp(float(shaped), 0.0, max_speed)

    def _publish_motor_speed_command(self, w1_sq: float, w2_sq: float, dt_actual: float):
        upper_target = float(math.sqrt(max(w2_sq, 0.0)))
        lower_mag_raw = float(math.sqrt(max(w1_sq, 0.0)))
        lower_scale = MMCUAVROS2Controller._resolved_rotor_lower_command_scale(self)
        lower_target = float(lower_mag_raw * lower_scale)

        self.upper_rotor_cmd_target = clamp(
            upper_target,
            0.0,
            float(getattr(self, "rotor_max_speed_rad_s", 450.0)),
        )
        self.lower_rotor_cmd_target = clamp(
            lower_target,
            0.0,
            float(getattr(self, "rotor_max_speed_rad_s", 450.0)),
        )

        upper_mag = MMCUAVROS2Controller._shape_rotor_speed_command(
            self,
            abs(float(getattr(self, "upper_rotor_cmd", 0.0))),
            self.upper_rotor_cmd_target,
            dt_actual,
        )
        lower_mag = MMCUAVROS2Controller._shape_rotor_speed_command(
            self,
            abs(float(getattr(self, "lower_rotor_cmd", 0.0))),
            self.lower_rotor_cmd_target,
            dt_actual,
        )

        self.upper_rotor_cmd = upper_mag
        self.lower_rotor_cmd = -lower_mag

        msg = MMCUAVROS2Controller._cached_actuators_message(self)
        try:
            msg.header.stamp = self.get_clock().now().to_msg()
        except AttributeError:
            pass
        msg.velocity = [upper_mag, lower_mag]
        msg.position = []
        msg.normalized = []
        self.pub_motor_speed.publish(msg)

    def _update_rotor_force_estimates(self):
        upper_speed = abs(float(getattr(self, "upper_rotor_actual", 0.0)))
        lower_speed = abs(float(getattr(self, "lower_rotor_actual", 0.0)))
        if upper_speed <= 1e-9 and abs(float(getattr(self, "upper_rotor_cmd", 0.0))) > 1e-9:
            upper_speed = abs(float(self.upper_rotor_cmd))
        if lower_speed <= 1e-9 and abs(float(getattr(self, "lower_rotor_cmd", 0.0))) > 1e-9:
            lower_speed = abs(float(self.lower_rotor_cmd))

        upper_thrust = self.rotor_upper_force_constant * upper_speed * upper_speed
        lower_thrust = self.rotor_lower_force_constant * lower_speed * lower_speed
        self.rotor_total_thrust_est = float(upper_thrust + lower_thrust)
        self.rotor_tau_z_est = float(self.rotor_moment_constant * (lower_thrust - upper_thrust))

    def _publish_actuation(
        self,
        thrust_real: float,
        tau_z_real: float,
        w1_sq: float,
        w2_sq: float,
        dt_actual: float,
        drag_force: np.ndarray,
        damping_torque: np.ndarray,
    ):
        if self.actuation_backend == "rotor_physics":
            self._publish_motor_speed_command(w1_sq, w2_sq, dt_actual)
            self._publish_aux_wrench(dt_actual, drag_force, damping_torque)
        else:
            self._publish_control_wrench(thrust_real, tau_z_real, dt_actual, drag_force, damping_torque)
            self._publish_rotor_visual(w1_sq, w2_sq)
            self.upper_rotor_cmd = float(math.sqrt(max(w2_sq, 0.0)))
            self.lower_rotor_cmd = float(-math.sqrt(max(w1_sq, 0.0)))
            self.rotor_total_thrust_est = float(thrust_real)
            self.rotor_tau_z_est = float(tau_z_real)

        if self.actuation_backend == "rotor_physics":
            self._update_rotor_force_estimates()

    def _publish_rotor_visual(self, w1_sq: float, w2_sq: float):
        # 上旋翼逆时针，对应正的转速平方根；下旋翼顺时针，对应负的转速平方根
        upper = MMCUAVROS2Controller._cached_float64_message(self, "_cached_upper_rotor_msg")
        lower = MMCUAVROS2Controller._cached_float64_message(self, "_cached_lower_rotor_msg")
        upper.data = float(math.sqrt(max(w2_sq, 0.0)))
        lower.data = float(-math.sqrt(max(w1_sq, 0.0)))
        self.pub_upper_rotor.publish(upper)
        self.pub_lower_rotor.publish(lower)

    def _publish_slider_commands(self, chi_cmd: float, ups_cmd: float):
        # 关键映射：
        # X 轴（轴向 = -X）：向两个滑块都发布 -chi_cmd。
        # Y 轴（轴向 = +Y）：向两个滑块都发布 +ups_cmd。
        green = MMCUAVROS2Controller._cached_float64_message(self, "_cached_slider_green_msg")
        purple = MMCUAVROS2Controller._cached_float64_message(self, "_cached_slider_purple_msg")
        blue = MMCUAVROS2Controller._cached_float64_message(self, "_cached_slider_blue_msg")
        red = MMCUAVROS2Controller._cached_float64_message(self, "_cached_slider_red_msg")

        green.data = float(-chi_cmd)
        purple.data = float(-chi_cmd)
        blue.data = float(ups_cmd)
        red.data = float(ups_cmd)

        self.pub_slider_green.publish(green)
        self.pub_slider_purple.publish(purple)
        self.pub_slider_blue.publish(blue)
        self.pub_slider_red.publish(red)

    def save_flight_data(self):
        if not self.flight_data_log:
            return
        log_dir = get_default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        output_path = log_dir / f"mmc_flight_log_{self.start_time_str}.csv"
        try:
            with output_path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(FLIGHT_LOG_HEADERS)
                writer.writerows(self.flight_data_log)
            print(f"\n==================================================")
            print(f"[Blackbox] Flight data successfully saved to: {output_path}")
            print(f"Total records: {len(self.flight_data_log)}")
            print(f"==================================================\n")
        except Exception as e:
            print(f"\n[Error] Failed to save flight data: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MMCUAVROS2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Ctrl+C detected] Interrupted by user, initiating safe shutdown...")
    finally:
        # 确保在节点销毁前将内存数据写入硬盘
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            node.save_flight_data()
            node.destroy_node()
            rclpy.try_shutdown()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
