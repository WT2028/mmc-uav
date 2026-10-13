"""Six-DOF equations shared with the ROS controller; pure numerical API."""
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
def clamp(value, minlim, maxlim):
    if value <= minlim:
        return minlim
    elif value >= maxlim:
        return maxlim
    else:
        return value

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

    # 正文 C_D,j：沿用现有名义有效系数，非实测辨识值。
    Cd_side: float = 1.2  # 侧面阻力系数（X/Y 轴）
    Cd_top: float = 1.3  # 垂直阻力系数（Z 轴，平板流）

    # 迎风面积（参考面积）[m^2]
    # 正文 A_j：沿用名义参考面积，不以未经核实的几何尺寸重新标定。
    S_side: float = 0.072  # A_x = A_y
    S_top: float = 0.16  # A_z

    # 正文 D_omega = diag(k_a,x, k_a,y, k_a,z)，单位 N*m*s/rad。
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

P = Params()

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

    # 由四质量的平行轴求和得到；来源文献式 (24)/(26) 的 s7/s8 对调。
    # Ixx 仅随 ups 改变，Iyy 仅随 chi 改变。
    add_xx = S.s10 + S.s6 + S.s5 + S.s3 - S.s7
    add_yy = S.s9 + S.s6 + S.s4 + S.s2 - S.s8
    add_zz = m * l * l + S.s10 + S.s9 + S.s5 + S.s4 + S.s3 + S.s2 - S.s8 - S.s7

    Ixx = Ib_x + 4.0 * Im_x + add_xx
    Iyy = Ib_y + 4.0 * Im_y + add_yy
    Izz = Ib_z + 4.0 * Im_z + add_zz
    Ixy = S.s1

    dIxx = S.ds10 + S.ds6 + S.ds5 + S.ds3 - S.ds7
    dIyy = S.ds9 + S.ds6 + S.ds4 + S.ds2 - S.ds8
    dIzz = S.ds10 + S.ds9 + S.ds5 + S.ds4 + S.ds3 + S.ds2 - S.ds8 - S.ds7
    dIxy = S.ds1

    return Inertia(Ixx, Iyy, Izz, Ixy, dIxx, dIyy, dIzz, dIxy)

def attitude_kinematics(p, q, r, phi, theta):
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    ctheta = math.cos(theta)
    ttheta = math.tan(theta)

    phid = p + r * (cphi * ttheta) + q * (sphi * ttheta)
    thetad = q * cphi - r * sphi
    psid = r * (cphi / ctheta) + q * (sphi / ctheta)

    return phid, thetad, psid

@dataclass(frozen=True)
class AerodynamicLoads:
    force_world: np.ndarray
    force_body: np.ndarray
    moment_cog_body: np.ndarray
    moment_origin_body: np.ndarray

def calculate_aerodynamic_loads(
    x: State6, P: Params, wind_world: Optional[Sequence[float]] = None,
) -> AerodynamicLoads:
    """正文式 (7)：机体原点的相对气流阻力，以及关于瞬时系统质心的力矩。

    wind_world 是惯性系风速 [m/s]；缺省零风用于未测风条件下的名义预测。
    位置/速度属于机体原点，故此处不再加质心速度或 omega x r 修正。
    """
    Tb_v, Tv_b = T_transformation(x)
    wind = np.zeros(3) if wind_world is None else np.asarray(wind_world, dtype=float).reshape(3)
    v_rel_body = Tv_b @ (np.array([x.vx, x.vy, x.vz]) - wind)
    drag_factors = 0.5 * P.rho * np.array([
        P.Cd_side * P.S_side, P.Cd_side * P.S_side, P.Cd_top * P.S_top,
    ])
    force_body = -drag_factors * v_rel_body * np.abs(v_rel_body)
    moment_origin = -np.array([P.kp * x.p, P.kq * x.q, P.kr * x.r])
    cog = 2.0 * P.mu * np.array([x.chi, x.ups, 0.0])
    moment_cog = moment_origin - np.cross(cog, force_body)
    return AerodynamicLoads(Tb_v @ force_body, force_body, moment_cog, moment_origin)

def calculate_aerodynamic_forces(
    x: State6, P: Params, wind_world: Optional[Sequence[float]] = None,
):
    """返回 (惯性系气动力, 惯性系质心气动力矩, 机体系质心气动力矩)。"""
    loads = calculate_aerodynamic_loads(x, P, wind_world)
    Tb_v, _ = T_transformation(x)
    return loads.force_world, Tb_v @ loads.moment_cog_body, loads.moment_cog_body

def update_6dof_model(
    P: Params, x: State6, u: Input6, external_loads: Optional[Sequence[float]] = None,
) -> State6:
    """完整模型；external_loads = [v_w^i, F_res^i, M_res^0]，共 9 个分量。

    缺省为零，供 MPC/NDO 名义模型使用；不会把仿真真值风速当作控制器测量。
    """
    w = np.zeros(9) if external_loads is None else np.asarray(external_loads, dtype=float).reshape(9)
    F_drag_inertial, _, M_aero_body = calculate_aerodynamic_forces(x, P, w[:3])
    F_external_inertial = F_drag_inertial + w[3:6]
    M_external_body = M_aero_body + w[6:9]
    Tb_v, _ = T_transformation(x)

    # 滑块执行器二阶动态
    wn, zt = P.wn_mass, P.zeta_mass
    chi_dd = wn * wn * (u.chi_cmd - x.chi) - 2.0 * zt * wn * x.chi_d
    ups_dd = wn * wn * (u.ups_cmd - x.ups) - 2.0 * zt * wn * x.ups_d

    S = cal_sigma(P, x.chi, x.ups, x.chi_d, x.ups_d)
    I = cal_inertia(P, S)
    M, mu, m = P.M, P.mu, P.m

    # 当前英文稿式 (6)：所有向量在机体系表达，I_dot/h_dot 为分量导数。
    inertia = np.array([
        [I.Ixx, I.Ixy, 0.0],
        [I.Ixy, I.Iyy, 0.0],
        [0.0, 0.0, I.Izz]
    ])
    inertia_dot = np.array([
        [I.dIxx, I.dIxy, 0.0],
        [I.dIxy, I.dIyy, 0.0],
        [0.0, 0.0, I.dIzz]
    ])
    omega = np.array([x.p, x.q, x.r])
    h_m = np.array([0.0, 0.0, 4.0 * m * mu * (x.ups * x.chi_d - x.chi * x.ups_d)])
    h_m_dot = np.array([0.0, 0.0, 4.0 * m * mu * (x.ups * chi_dd - x.chi * ups_dd)])
    cog = 2.0 * mu * np.array([x.chi, x.ups, 0.0])
    thrust_body = np.array([0.0, 0.0, u.thrust])
    torque = (-np.cross(cog, thrust_body) + np.array([0.0, 0.0, u.tau_z])
              + M_external_body)
    angular_rhs = (torque - inertia_dot @ omega
                   - np.cross(omega, inertia @ omega + h_m) - h_m_dot)

    try:
        omega_dot = np.linalg.solve(inertia, angular_rhs)
    except np.linalg.LinAlgError:
        omega_dot = np.zeros(3)
    pd, qd, rd = omega_dot

    # 当前英文稿式 (5)：位置/速度属于机体原点，质心修正需整体转入惯性系。
    cog_dot = 2.0 * mu * np.array([x.chi_d, x.ups_d, 0.0])
    cog_ddot = 2.0 * mu * np.array([chi_dd, ups_dd, 0.0])
    cog_acc_body = (cog_ddot + 2.0 * np.cross(omega, cog_dot)
                    + np.cross(omega_dot, cog) + np.cross(omega, np.cross(omega, cog)))
    x_dd, y_dd, z_dd = (Tb_v @ (thrust_body / M - cog_acc_body)
                       + F_external_inertial / M + np.array([0.0, 0.0, -P.g]))

    phid, thetad, psid = attitude_kinematics(x.p, x.q, x.r, x.phi, x.theta)
    dx = State6()
    dx.x, dx.y, dx.z = x.vx, x.vy, x.vz
    dx.vx, dx.vy, dx.vz = x_dd, y_dd, z_dd
    dx.phi, dx.theta, dx.psi = phid, thetad, psid
    dx.p, dx.q, dx.r = pd, qd, rd
    dx.chi, dx.ups = x.chi_d, x.ups_d
    dx.chi_d, dx.ups_d = chi_dd, ups_dd
    return dx

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
