"""Mass, center-of-mass and full-tensor inertia from the URDF assembly."""
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


DEFAULT_URDF = ROOT / "src/mmc_uav_description/urdf/mmc_uav.urdf"

PAIR_MAP = {"joint_slider_green": (0, -1.0),
            "joint_slider_purple": (0, -1.0),
            "joint_slider_blue": (1, 1.0),
            "joint_slider_red": (1, 1.0)}

INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def vector(text):
    result = np.fromstring(text or "0 0 0", sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Expected three finite coordinates: " + str(text))
    return result

def skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])

def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return (math.cos(angle)*np.eye(3) + (1-math.cos(angle))*np.outer(axis, axis)
            + math.sin(angle)*skew(axis))

def origin(node):
    return (vector(node.get("xyz")), rotation(vector(node.get("rpy")))) if node is not None else (np.zeros(3), np.eye(3))

def inertia(node):
    vals = {key: float(node.get(key, "0")) for key in INERTIA_KEYS}
    result = np.array([[vals["ixx"], vals["ixy"], vals["ixz"]],
                       [vals["ixy"], vals["iyy"], vals["iyz"]],
                       [vals["ixz"], vals["iyz"], vals["izz"]]])
    if not np.isfinite(result).all() or np.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("Invalid positive-definite link CoM inertia")
    return result

def joint_coordinates(sx, sy):
    values = (float(sx), float(sy))
    return {name: sign*values[axis] for name, (axis, sign) in PAIR_MAP.items()}

@dataclasses.dataclass
class Link:
    name: str
    mass: float
    com_local: np.ndarray
    inertia_rotation: np.ndarray
    own_inertia: np.ndarray
    joint_name: str | None
    joint_type: str
    joint_position: np.ndarray
    joint_rotation: np.ndarray
    joint_axis: np.ndarray

class CADAssembly:
    """Exact mass-property sums for the current seven-link star topology."""

    def __init__(self, urdf=DEFAULT_URDF):
        self.source = Path(urdf) if not isinstance(urdf, ET.Element) else None
        self.root = ET.parse(urdf).getroot() if self.source else copy.deepcopy(urdf)
        child_joints = {j.find("child").get("link"): j for j in self.root.findall("joint")}
        self.links = []
        for link in self.root.findall("link"):
            node = link.find("inertial")
            if node is None:
                continue
            name = link.get("name")
            joint = child_joints.get(name)
            if joint is not None and joint.find("parent").get("link") != "base_link":
                raise ValueError("Nested CAD topology requires explicit recursive kinematics")
            local, local_R = origin(node.find("origin"))
            jp, jr = origin(joint.find("origin") if joint is not None else None)
            joint_type = joint.get("type") if joint is not None else "fixed"
            if joint_type not in {"fixed", "prismatic", "continuous", "revolute"}:
                raise ValueError("Unsupported joint type: " + joint_type)
            ja = vector(joint.find("axis").get("xyz")) if joint is not None and joint.find("axis") is not None else np.array([0., 0., 1.])
            ja /= np.linalg.norm(ja)
            mass = float(node.find("mass").get("value"))
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError("Nonpositive CAD link mass: " + name)
            self.links.append(Link(name, mass, local, local_R, inertia(node.find("inertia")),
                                   joint.get("name") if joint is not None else None,
                                   joint_type, jp, jr, ja))
        if len(self.links) != 7 or {link.joint_name for link in self.links if link.joint_type == "prismatic"} != set(PAIR_MAP):
            raise ValueError("Expected the current base, two rotors, and four slider links")
        self.M = sum(link.mass for link in self.links)
        self.m = next(link.mass for link in self.links if link.joint_type == "prismatic")
        if any(abs(link.mass-self.m) > 1e-12 for link in self.links if link.joint_type == "prismatic"):
            raise ValueError("Paired Params interface requires equal moving masses")
        self.mu = self.m/self.M
        self.rotor_config = {}
        for plugin in self.root.findall("gazebo/plugin"):
            if "MulticopterMotorModel" in plugin.get("name", ""):
                name = plugin.findtext("jointName")
                self.rotor_config[name] = dict(
                    link=plugin.findtext("linkName"),
                    direction={"ccw": 1., "cw": -1.}[plugin.findtext("turningDirection")],
                    motor_constant=float(plugin.findtext("motorConstant")),
                    moment_constant=float(plugin.findtext("momentConstant")),
                    slowdown=float(plugin.findtext("rotorVelocitySlowdownSim")),
                    maximum_speed=float(plugin.findtext("maxRotVelocity")))
        self._build_coefficients()

    def link_states(self, positions=None, velocities=None, accelerations=None):
        positions, velocities, accelerations = positions or {}, velocities or {}, accelerations or {}
        records = []
        for link in self.links:
            q, qd, qdd = (float(data.get(link.joint_name, 0.)) for data in (positions, velocities, accelerations))
            Rj = link.joint_rotation
            axis = Rj @ link.joint_axis
            rel_omega, rel_alpha = np.zeros(3), np.zeros(3)
            if link.joint_type == "prismatic":
                R = Rj
                p = link.joint_position + R @ link.com_local + axis*q
                velocity, acceleration = axis*qd, axis*qdd
            elif link.joint_type in {"continuous", "revolute"}:
                R = Rj @ axis_rotation(link.joint_axis, q)
                offset = R @ link.com_local
                p = link.joint_position + offset
                rel_omega, rel_alpha = axis*qd, axis*qdd
                velocity = np.cross(rel_omega, offset)
                acceleration = np.cross(rel_alpha, offset) + np.cross(rel_omega, np.cross(rel_omega, offset))
            else:
                R = Rj
                p = link.joint_position + R @ link.com_local
                velocity, acceleration = np.zeros(3), np.zeros(3)
            own = R @ link.inertia_rotation @ link.own_inertia @ link.inertia_rotation.T @ R.T
            own_dot = skew(rel_omega) @ own - own @ skew(rel_omega)
            records.append(dict(name=link.name, joint_name=link.joint_name, mass=link.mass,
                                p=p, v=velocity, a=acceleration, own_I=own,
                                own_I_dot=own_dot, relative_omega=rel_omega,
                                relative_alpha=rel_alpha, joint_axis_body=axis))
        return records

    def terms_from_joints(self, positions=None, velocities=None, accelerations=None):
        """Independent direct sums; use every measured joint, including rotors."""
        records = self.link_states(positions, velocities, accelerations)
        c = sum(r["mass"]*r["p"] for r in records)/self.M
        cd = sum(r["mass"]*r["v"] for r in records)/self.M
        cdd = sum(r["mass"]*r["a"] for r in records)/self.M
        I, Id, h, hd = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3)
        for r in records:
            rel, rel_v, mass = r["p"]-c, r["v"]-cd, r["mass"]
            I += r["own_I"] + mass*((rel@rel)*np.eye(3)-np.outer(rel, rel))
            Id += r["own_I_dot"] + mass*(2*(rel@rel_v)*np.eye(3)-np.outer(rel, rel_v)-np.outer(rel_v, rel))
            h += mass*np.cross(rel, r["v"]) + r["own_I"] @ r["relative_omega"]
            hd += mass*(np.cross(rel_v, r["v"]) + np.cross(rel, r["a"]))
            hd += r["own_I_dot"] @ r["relative_omega"] + r["own_I"] @ r["relative_alpha"]
        return dict(c=c, c_dot=cd, c_ddot=cdd, I=I, I_dot=Id, h_m=h, h_m_dot=hd)

    def _build_coefficients(self):
        records = self.link_states()
        self.c0 = sum(r["mass"]*r["p"] for r in records)/self.M
        self.link_B = {}
        for link, record in zip(self.links, records):
            B = np.zeros((3, 2))
            if link.joint_name in PAIR_MAP:
                axis, sign = PAIR_MAP[link.joint_name]
                B[:, axis] = record["joint_axis_body"]*sign
            self.link_B[link.name] = B
        self.C = sum(r["mass"]*self.link_B[r["name"]] for r in records)/self.M
        self.I0 = self.terms_from_joints()["I"]
        self.I_linear = np.zeros((2, 3, 3))
        self.I_quadratic = np.zeros((3, 3, 3))  # sx², sx*sy, sy²
        self.h_linear = np.zeros((3, 2))
        self.h_bilinear = np.zeros((2, 2, 3))
        for r in records:
            mass, p, B = r["mass"], r["p"]-self.c0, self.link_B[r["name"]]
            D = B-self.C
            for k in range(2):
                d = D[:, k]
                self.I_linear[k] += mass*(2*(p@d)*np.eye(3)-np.outer(p, d)-np.outer(d, p))
                self.h_linear[:, k] += mass*np.cross(p, B[:, k])
                for l in range(2):
                    self.h_bilinear[k, l] += mass*np.cross(D[:, k], B[:, l])
            dx, dy = D[:, 0], D[:, 1]
            self.I_quadratic[0] += mass*((dx@dx)*np.eye(3)-np.outer(dx, dx))
            self.I_quadratic[1] += mass*(2*(dx@dy)*np.eye(3)-np.outer(dx, dy)-np.outer(dy, dx))
            self.I_quadratic[2] += mass*((dy@dy)*np.eye(3)-np.outer(dy, dy))
        self.records_zero = records

    def terms(self, sx, sy, vx=0., vy=0., ax=0., ay=0., *,
              rotor_positions=None, rotor_velocities=None, rotor_accelerations=None):
        """Exact paired CAD polynomials; optional independent rotor kinematics."""
        s, v, a = np.array([sx, sy], float), np.array([vx, vy], float), np.array([ax, ay], float)
        if not np.isfinite(np.r_[s, v, a]).all():
            raise ValueError("Nonfinite slider history")
        if rotor_positions or rotor_velocities or rotor_accelerations:
            q, qd, qdd = joint_coordinates(*s), joint_coordinates(*v), joint_coordinates(*a)
            q.update(rotor_positions or {}); qd.update(rotor_velocities or {}); qdd.update(rotor_accelerations or {})
            return self.terms_from_joints(q, qd, qdd)
        I = self.I0 + sx*self.I_linear[0] + sy*self.I_linear[1]
        I = I + sx*sx*self.I_quadratic[0] + sx*sy*self.I_quadratic[1] + sy*sy*self.I_quadratic[2]
        Id = vx*self.I_linear[0] + vy*self.I_linear[1]
        Id = Id + 2*sx*vx*self.I_quadratic[0] + (vx*sy+sx*vy)*self.I_quadratic[1] + 2*sy*vy*self.I_quadratic[2]
        h = self.h_linear @ v + np.einsum("k,l,klj->j", s, v, self.h_bilinear)
        hd = self.h_linear @ a + np.einsum("k,l,klj->j", v, v, self.h_bilinear) + np.einsum("k,l,klj->j", s, a, self.h_bilinear)
        return dict(c=self.c0+self.C@s, c_dot=self.C@v, c_ddot=self.C@a,
                    I=I, I_dot=Id, h_m=h, h_m_dot=hd)

    def slider_required_forces(self, positions, velocities, accelerations,
                               body_acceleration_world, body_to_world,
                               omega_body, omega_dot_body,
                               gravity_world=(0., 0., -9.81)):
        """Exact prismatic inverse force from absolute CoM acceleration.

        The input is the actual body-origin acceleration, not the whole-CoM
        acceleration. Each force is in that joint's signed URDF coordinate.
        Gravity, body translational/rotational reaction, and Coriolis terms
        are included. There is no link-local aerodynamic force or friction in
        the current physical slider object. This is a force demand, not a
        force-clipped forward actuator model.
        """
        R = np.asarray(body_to_world, float)
        omega, od = np.asarray(omega_body, float), np.asarray(omega_dot_body, float)
        specific_acc = R.T @ (np.asarray(body_acceleration_world)-np.asarray(gravity_world))
        result = {}
        for r in self.link_states(positions, velocities, accelerations):
            if r["joint_name"] not in PAIR_MAP:
                continue
            absolute_specific = (specific_acc + r["a"] + 2*np.cross(omega, r["v"])
                                 + np.cross(od, r["p"]) + np.cross(omega, np.cross(omega, r["p"])))
            result[r["joint_name"]] = float(r["mass"]*(r["joint_axis_body"] @ absolute_specific))
        return result

    def rotor_wrench_from_speeds(self, velocities, positions=None, cog=None):
        """Signed Gazebo motor wrench, including the actual rotor force point.

        Gazebo's MulticopterMotorModel uses Link::AddWorldForce without an
        offset, so rotor thrust acts at that rotor link's CoM. Its aerodynamic
        drag moment acts on the parent. Independent rigid-rotor spin remains
        in terms_from_joints, not in this external wrench.
        """
        positions = positions or {}
        c = self.terms_from_joints(positions)["c"] if cog is None else np.asarray(cog)
        records = {r["name"]: r for r in self.link_states(positions)}
        force, torque = np.zeros(3), np.zeros(3)
        rotors = {}
        for name, config in self.rotor_config.items():
            speed = float(velocities.get(name, 0.))*config["slowdown"]
            thrust = config["direction"]*float(np.sign(speed))*speed**2*config["motor_constant"]
            record = records[config["link"]]
            link = next(l for l in self.links if l.joint_name == name)
            R = link.joint_rotation @ axis_rotation(link.joint_axis, float(positions.get(name, 0.)))
            f = R[:, 2]*thrust
            drag = R[:, 2]*(-config["direction"]*thrust*config["moment_constant"])
            force += f
            torque += np.cross(record["p"], f)+drag
            rotors[name] = dict(thrust_N=thrust, real_speed_rad_s=speed,
                                force_point_body_m=record["p"], force_body_N=f,
                                aerodynamic_drag_moment_body_Nm=drag)
        return dict(force_body=force, moment_body_origin=torque,
                    moment_cog=torque-np.cross(c, force), rotors=rotors,
                    force_application="Each rotor link CoM (Gazebo AddWorldForce)")

    def rotor_wrench_from_inputs(self, thrust, tau_z, positions=None, cog=None):
        """Allocate total positive thrust and aerodynamic yaw moment exactly.

        Returned motor commands obey each physical 400 rad/s ceiling only if
        feasible=True. Requests are never silently clipped. The resulting
        moment includes the small CAD horizontal thrust-point offset.
        """
        upper = self.rotor_config["joint_rotor_upper"]
        lower = self.rotor_config["joint_rotor_lower"]
        du, dl = upper["moment_constant"], lower["moment_constant"]
        fu = (dl*thrust-tau_z)/(du+dl)
        fl = (du*thrust+tau_z)/(du+dl)
        if min(fu, fl) < -1e-12:
            raise ValueError("Requested yaw moment requires negative rotor thrust")
        speeds = {"joint_rotor_upper": math.sqrt(max(0., fu)/upper["motor_constant"]),
                  "joint_rotor_lower": math.sqrt(max(0., fl)/lower["motor_constant"])}
        actual = {name: self.rotor_config[name]["direction"]*value/self.rotor_config[name]["slowdown"]
                  for name, value in speeds.items()}
        result = self.rotor_wrench_from_speeds(actual, positions, cog)
        result["motor_commands_rad_s"] = speeds
        result["feasible"] = all(speed <= self.rotor_config[name]["maximum_speed"]+1e-10 for name, speed in speeds.items())
        return result

    def coefficients(self):
        return dict(total_mass_kg=self.M, moving_mass_each_kg=self.m, mu=self.mu,
                    c0=self.c0, C=self.C, I0=self.I0, I_linear=self.I_linear,
                    I_quadratic=self.I_quadratic, h_linear=self.h_linear,
                    h_bilinear=self.h_bilinear,
                    coordinate_order=["sx", "sy"],
                    inertia_quadratic_order=["sx_squared", "sx_sy", "sy_squared"],
                    independent_rotor_spin="Excluded by fast path; use measured-joint direct path when available.")

def _translate_origin(node, shift):
    if node is None:
        return
    node.set("xyz", " ".join(f"{x:.17g}" for x in vector(node.get("xyz"))-shift))

def build_variant(mu, *, urdf=DEFAULT_URDF, wn=20., zeta=1.2):
    """Return a new in-memory URDF root and mass/rebasing provenance."""
    original = ET.parse(urdf).getroot()
    root = copy.deepcopy(original)
    base_assembly = CADAssembly(original)
    mu = float(mu)
    fixed_rotor_mass = sum(link.mass for link in base_assembly.links if link.name.startswith("rotor_"))
    moving = mu*base_assembly.M
    base_mass = base_assembly.M-fixed_rotor_mass-4*moving
    if not (math.isfinite(mu) and mu > 0 and base_mass > 0 and wn > 0 and zeta > 0):
        raise ValueError("Mass ratio must leave positive base_link mass and actuator parameters")
    changes = []
    for link in root.findall("link"):
        name = link.get("name")
        node = link.find("inertial")
        old_mass = float(node.find("mass").get("value"))
        new_mass = base_mass if name == "base_link" else moving if name.startswith("slider_") else old_mass
        factor = new_mass/old_mass
        node.find("mass").set("value", f"{new_mass:.17g}")
        for key in INERTIA_KEYS:
            item = node.find("inertia")
            item.set(key, f"{float(item.get(key))*factor:.17g}")
        changes.append(dict(link=name, nominal_mass_kg=old_mass, variant_mass_kg=new_mass,
                            own_inertia_mass_scale=factor))
    before = CADAssembly(root)
    shift = before.c0
    base = root.find("link[@name='base_link']")
    for node in (base.find("inertial/origin"), *base.findall("visual/origin"), *base.findall("collision/origin")):
        _translate_origin(node, shift)
    for joint in root.findall("joint"):
        if joint.find("parent").get("link") == "base_link":
            _translate_origin(joint.find("origin"), shift)
    for gazebo in root.findall("gazebo"):
        if gazebo.get("reference") == "base_link":
            for pose in gazebo.findall("sensor/pose"):
                values = np.fromstring(pose.text, sep=" ")
                values[:3] -= shift
                pose.text = " ".join(f"{x:.17g}" for x in values)
    for plugin in root.findall("gazebo/plugin"):
        if plugin.get("name") == "mmc::SliderServo":
            plugin.find("moving_mass").text = f"{moving:.17g}"
            plugin.find("wn").text = f"{wn:.17g}"
            plugin.find("zeta").text = f"{zeta:.17g}"
    after = CADAssembly(root)
    geometry_error = max(np.linalg.norm(a["p"]-(b["p"]+shift))
                         for a, b in zip(before.records_zero, after.records_zero))
    metadata = dict(mu=mu, total_mass_kg=after.M, moving_mass_each_kg=moving,
                    base_link_mass_kg=base_mass, fixed_rotor_mass_kg=fixed_rotor_mass,
                    mass_family="CAD_fixed_total_mass_fixed_rotors",
                    family_rule="Only base_link absorbs the four-slider mass change; each changed link own CoM inertia scales with its mass; rotor masses/inertias and relative geometry remain fixed.",
                    rebase_shift_in_default_body_coordinates_m=shift,
                    zero_cog_after_rebase_m=after.c0,
                    zero_inertia_before_rebase=before.I0,
                    zero_inertia_after_rebase=after.I0,
                    physical_geometry_rebase_error_m=geometry_error,
                    per_link_mass_inertia_changes=changes,
                    servo_wn_rad_s=wn, servo_zeta=zeta,
                    default_urdf_sha256=sha256(urdf),
                    defaults_modified=False)
    return root, metadata

@functools.lru_cache(maxsize=256)
def make_cad_assembly(mu, urdf=DEFAULT_URDF, wn=20., zeta=1.2):
    root, metadata = build_variant(mu, urdf=urdf, wn=wn, zeta=zeta)
    model = CADAssembly(root)
    model.variant_metadata = metadata
    return model

def legacy_parameter_values(model, wn=20., zeta=1.2):
    """Zero-diagonal calibration only; full CAD geometry must use terms()."""
    moving_own = next(r["own_I"] for r in model.records_zero if r["name"].startswith("slider_"))
    Imx, Imy, Imz = np.diag(moving_own)
    # The effective CAD rail half-span is 0.1817202 m, not the ideal 0.2 m.
    xgroup = [r["p"][1] for r in model.records_zero if r["joint_name"] in ("joint_slider_green", "joint_slider_purple")]
    ygroup = [r["p"][0] for r in model.records_zero if r["joint_name"] in ("joint_slider_blue", "joint_slider_red")]
    span = .5*(abs(xgroup[0]-xgroup[1])+abs(ygroup[0]-ygroup[1]))
    return dict(m=model.m, m_b=model.M-4*model.m, l=span,
                Ib_x=model.I0[0, 0]-4*Imx-model.m*span**2/2,
                Ib_y=model.I0[1, 1]-4*Imy-model.m*span**2/2,
                Ib_z=model.I0[2, 2]-4*Imz-model.m*span**2,
                Im_x=Imx, Im_y=Imy, Im_z=Imz,
                wn_mass=float(wn), zeta_mass=float(zeta),
                slider_vel_max=.25, u2_lim=.15)

def ideal_terms(mu, sx, sy, vx=0., vy=0., ax=0., ay=0.):
    """Independent frozen analytical family for discrepancy accounting."""
    M = 1.45; m = mu*M; mb = M-4*m
    own_b = np.diag([.0585, .0585, .06])*mb/1.25
    own_m = np.diag([.0006, .0006, .0012])*m/.05
    c = 2*mu*np.array([sx, sy, 0.]); cd = 2*mu*np.array([vx, vy, 0.]); cdd = 2*mu*np.array([ax, ay, 0.])
    positions = [np.zeros(3), np.array([sx, .2, 0.]), np.array([.2, sy, 0.]),
                 np.array([sx, -.2, 0.]), np.array([-.2, sy, 0.])]
    velocities = [np.zeros(3), np.array([vx, 0., 0.]), np.array([0., vy, 0.]),
                  np.array([vx, 0., 0.]), np.array([0., vy, 0.])]
    accelerations = [np.zeros(3), np.array([ax, 0., 0.]), np.array([0., ay, 0.]),
                     np.array([ax, 0., 0.]), np.array([0., ay, 0.])]
    I, Id, h, hd = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3)
    for i, (p, v, a) in enumerate(zip(positions, velocities, accelerations)):
        mass = mb if i == 0 else m; own = own_b if i == 0 else own_m
        r, rd = p-c, v-cd
        I += own+mass*((r@r)*np.eye(3)-np.outer(r, r))
        Id += mass*(2*(r@rd)*np.eye(3)-np.outer(r, rd)-np.outer(rd, r))
        h += mass*np.cross(r, v); hd += mass*(np.cross(rd, v)+np.cross(r, a))
    return dict(c=c, c_dot=cd, c_ddot=cdd, I=I, I_dot=Id, h_m=h, h_m_dot=hd)

def audit_assembly(model):
    """Independent derivative/geometry checks, not flight or physical evidence."""
    checks = {}
    worst = {name: 0. for name in ("polynomial_c", "polynomial_I", "polynomial_I_dot", "polynomial_h", "polynomial_h_dot", "finite_difference_I_dot", "finite_difference_h_dot")}
    dt = 1e-6
    for sx, sy in [(0., 0.), (.135, 0.), (0., -.135), (.135, -.135), (.037, -.061)]:
        vx, vy, ax, ay = .17, -.11, 1.3, -.9
        fast = model.terms(sx, sy, vx, vy, ax, ay)
        direct = model.terms_from_joints(joint_coordinates(sx, sy), joint_coordinates(vx, vy), joint_coordinates(ax, ay))
        for metric, key in [("polynomial_c", "c"), ("polynomial_I", "I"), ("polynomial_I_dot", "I_dot"), ("polynomial_h", "h_m"), ("polynomial_h_dot", "h_m_dot")]:
            worst[metric] = max(worst[metric], float(np.max(np.abs(fast[key]-direct[key]))))
        plus = model.terms(sx+vx*dt+.5*ax*dt**2, sy+vy*dt+.5*ay*dt**2, vx+ax*dt, vy+ay*dt)
        minus = model.terms(sx-vx*dt+.5*ax*dt**2, sy-vy*dt+.5*ay*dt**2, vx-ax*dt, vy-ay*dt)
        worst["finite_difference_I_dot"] = max(worst["finite_difference_I_dot"], float(np.max(np.abs((plus["I"]-minus["I"])/(2*dt)-fast["I_dot"]))))
        worst["finite_difference_h_dot"] = max(worst["finite_difference_h_dot"], float(np.max(np.abs((plus["h_m"]-minus["h_m"])/(2*dt)-fast["h_m_dot"]))))
    for name, value in worst.items():
        limit = 1e-10 if name.startswith("finite_difference") else 1e-12
        checks[name] = dict(value=value, limit=limit, passed=value <= limit)
    checks["zero_cog"] = dict(value=float(np.linalg.norm(model.c0)), limit=1e-12, passed=np.linalg.norm(model.c0) <= 1e-12)
    expected = np.diag([2*model.mu, 2*model.mu, 0.])[:, :2]
    checks["paired_cog_map"] = dict(value=float(np.max(abs(model.C-expected))), limit=1e-12, passed=np.max(abs(model.C-expected)) <= 1e-12)
    # Static gravity and a freely falling body distinguish load from mass*a.
    force = model.slider_required_forces({}, {}, joint_coordinates(1.3, -.9), [0, 0, 0], np.eye(3), [0, 0, 0], [0, 0, 0])
    expected_forces = {name: model.m*value for name, value in joint_coordinates(1.3, -.9).items()}
    error = max(abs(force[name]-value) for name, value in expected_forces.items())
    checks["horizontal_prismatic_force"] = dict(value=error, limit=1e-12, passed=error <= 1e-12)
    # At a tilted body, stationary held sliders carry projected gravity.
    R = rotation(np.array([.2, -.3, .4]))
    held = model.slider_required_forces({}, {}, {}, [0, 0, 0], R, [0, 0, 0], [0, 0, 0])
    falling = model.slider_required_forces({}, {}, {}, [0, 0, -9.81], R, [0, 0, 0], [0, 0, 0])
    checks["tilted_gravity_load_nonzero"] = dict(value=max(abs(v) for v in held.values()), passed=max(abs(v) for v in held.values()) > 1e-4)
    checks["freefall_zero_joint_load"] = dict(value=max(abs(v) for v in falling.values()), limit=1e-12, passed=max(abs(v) for v in falling.values()) <= 1e-12)
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks,
                scope="Static CAD sums, analytical derivatives, and inverse prismatic force; no physical simulation")
