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
from . import simulation as base
@dataclasses.dataclass(frozen=True)
class Case:
    key: str
    family: str = "preposition_move"
    mu: float = 0.05 / 1.45
    wn: float = 20.0
    sx0: float = 0.0
    sy0: float = 0.0
    sx1: float = 0.0
    sy1: float = 0.0
    delay_y: float = 0.025
    transition: float = 0.10
    thrust_scale0: float = 1.0
    thrust_scale1: float = 1.0
    phi0: float = 0.0
    theta0: float = 0.0
    psi0: float = 0.0
    p0: float = 0.0
    q0: float = 0.0
    r0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0
    tau_z: float = 0.0
    sxv0: float = 0.0
    syv0: float = 0.0
    orbit_radius: float = 0.0
    orbit_rate: float = 0.0
    orbit_angle: float = 0.0
    orbit_radial_velocity: float = 0.0

    def initial(self):
        return base.State6(phi=self.phi0, theta=self.theta0, psi=self.psi0,
                           p=self.p0, q=self.q0, r=self.r0,
                           vx=self.vx0, vy=self.vy0,
                           chi=self.sx0, ups=self.sy0, chi_d=self.sxv0, ups_d=self.syv0)

    def input(self, t, P):
        a = base.quintic(t / self.transition)
        b = base.quintic((t - self.delay_y) / self.transition)
        sx = self.sx0 + (self.sx1 - self.sx0) * a
        sy = self.sy0 + (self.sy1 - self.sy0) * b
        if self.family == "counter_yaw_orbit":
            angle = self.orbit_angle - self.orbit_rate * t
            radius = self.orbit_radius + self.orbit_radial_velocity*t
            sx = radius * math.cos(angle)
            sy = radius * math.sin(angle)
            vx = radius * self.orbit_rate * math.sin(angle) + self.orbit_radial_velocity*math.cos(angle)
            vy = -radius * self.orbit_rate * math.cos(angle) + self.orbit_radial_velocity*math.sin(angle)
            ax = -self.orbit_rate**2 * sx + 2*self.orbit_radial_velocity*self.orbit_rate*math.sin(angle)
            ay = -self.orbit_rate**2 * sy - 2*self.orbit_radial_velocity*self.orbit_rate*math.cos(angle)
            sx += (2*P.zeta_mass*P.wn_mass*vx + ax) / P.wn_mass**2
            sy += (2*P.zeta_mass*P.wn_mass*vy + ay) / P.wn_mass**2
        if max(abs(sx),abs(sy)) > .135+1e-12:
            raise ValueError(f"command exceeds allowance: {self.key}")
        thrust = P.M * P.g * (self.thrust_scale0 + (self.thrust_scale1 - self.thrust_scale0) * a)
        u = base.Input6(thrust=thrust, tau_z=self.tau_z, chi_cmd=sx, ups_cmd=sy)
        return u, {"sx_cmd": sx, "sy_cmd": sy, "thrust": thrust, "tau_z": self.tau_z}
