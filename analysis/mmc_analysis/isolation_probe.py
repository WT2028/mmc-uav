#!/usr/bin/env python3
"""Record an isolated CAD multibody short-window with declared fixed rotors.

The current default plant/controller are unchanged. A generated copy retains
the seven physical links and four force servos, locks the two rotor joints,
and receives a prescribed wrench at base_link's origin. Initial slider
coordinates and rigid-body velocity are established once at startup and the
initialization marker is preserved. All subsequent joint/body states are
physics observations; no virtual states are substituted. This verifies the
internal-mass/CAD mechanism, not motor dynamics or closed-loop flight.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import threading
import time
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from . import cad
from .cad import (CADAssembly,DEFAULT_URDF,build_variant,joint_coordinates,sha256,write_csv,write_json)
WORKSPACE = cad.ROOT
ALIGNMENT_SCRIPT = Path(cad.__file__)


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
SCRIPT = Path(__file__).resolve()
WRENCH_SOURCE = SCRIPT.with_name("isolation_wrench.cc")
COLORS = ("blue", "red", "green", "purple")
WRENCH_FIELDS = ("stamp_s", "phase_s", "thrust_N", "force_body_x_N", "force_body_y_N", "force_body_z_N",
                "moment_body_x_Nm", "moment_body_y_Nm", "moment_body_z_Nm", "force_world_x_N", "force_world_y_N",
                "force_world_z_N", "moment_world_x_Nm", "moment_world_y_Nm", "moment_world_z_Nm")
BODY_FIELDS = ("stamp_s", "x", "y", "z", "qx", "qy", "qz", "qw", "vx_world", "vy_world", "vz_world",
               "p", "q", "r", "ax_world", "ay_world", "az_world", "alpha_body_x", "alpha_body_y", "alpha_body_z")


def q5(x):
    x = min(max(float(x), 0.), 1.)
    return x*x*x*(10+x*(-15+6*x))


def frozen_case(S, T, mu=.05/1.45, aero=True):
    if not (0 <= S <= .135 and T > 0):
        raise ValueError("S must be within the declared .135 m command limit and T positive")
    scale = S/.135
    return dict(S_m=S, T_s=T, mu=mu, horizon_s=.35, wn_rad_s=20., zeta=1.2,
                sx_initial_m=.75*S, sy_initial_m=S,
                sx_final_m=-.25*S, sy_final_m=.075*S, y_delay_s=.25*T,
                phi_initial_rad=-.15*scale, theta_initial_rad=.1*scale, psi_initial_rad=0.,
                omega_initial_body_rad_s=[1.5*scale, -.75*scale, 0.],
                thrust_relative_rise=.1, gravity_world_m_s2=[0., 0., -9.81],
                enable_nominal_aero=aero,
                initial_base_position_world_m=[0., 0., 10.],
                duration_s=max(.45, 1.25*T+.35+.1),
                scope="Generated locked-rotor internal-mass physical probe, not actual rotor flight",
                rotor_joints_locked=True, force_application="Prescribed wrench at base_link origin",
                initializing_only_once=True)


def target_history(case, t):
    sx = case["sx_initial_m"]+(case["sx_final_m"]-case["sx_initial_m"])*q5(t/case["T_s"])
    sy = case["sy_initial_m"]+(case["sy_final_m"]-case["sy_initial_m"])*q5((t-case["y_delay_s"])/case["T_s"])
    return joint_coordinates(sx, sy)


def selected_case(path, key, aero=True):
    """Load a frozen map case rather than silently recomputing its root."""
    values = json.loads(Path(path).read_text())
    chosen = [row for row in values if row["key"] == key]
    if len(chosen) != 1:
        raise ValueError("Frozen case key must match exactly one record")
    row = chosen[0]
    case = frozen_case(row["sy0"], row["transition"], row["mu"], aero)
    case.update(sx_initial_m=row["sx0"], sy_initial_m=row["sy0"],
                sx_final_m=row["sx1"], sy_final_m=row["sy1"],
                y_delay_s=row["delay_y"], phi_initial_rad=row["phi0"],
                theta_initial_rad=row["theta0"], psi_initial_rad=row["psi0"],
                omega_initial_body_rad_s=[row["p0"], row["q0"], row["r0"]],
                wn_rad_s=row["wn"], key=key, frozen_source_case=row,
                frozen_cases_file=str(Path(path).resolve()),
                frozen_cases_sha256=sha256(Path(path)))
    if row["thrust_scale0"] != 1. or row["tau_z"] != 0.:
        raise ValueError("Current isolated protocol supports Mg initial thrust and zero yaw input")
    if any(row.get(k, 0.) != 0. for k in ("vx0", "vy0", "sxv0", "syv0", "orbit_radius")):
        raise ValueError("Frozen case requires an unsupported nonzero initial or orbital input")
    case["thrust_relative_rise"] = row["thrust_scale1"] - 1.
    return case


def mass_capability_case(mu, key, alpha_requirement=.503833, aero=True):
    case = frozen_case(.135,.05,mu,aero)
    case.update(key=key, S_m=None, T_s=.05, sx_initial_m=.135, sx_final_m=.135,
                sy_initial_m=0., sy_final_m=0., y_delay_s=0.,
                phi_initial_rad=0., theta_initial_rad=0., psi_initial_rad=0.,
                omega_initial_body_rad_s=[0.,0.,0.], thrust_relative_rise=0.,
                duration_s=.4, family="CAD_fixed_total_mass_capability",
                task_alpha_requirement_rad_s2=alpha_requirement,
                scope="Mass-capacity torque/initial-alpha test with fixed rotor joints and base-origin Mg; separate from the Fig.8 S-T map")
    return case


def compile_instrumentation(out):
    out.mkdir(parents=True, exist_ok=True)
    library = out/"libjoint_isolation_wrench.so"
    flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "gz-sim8", "gz-transport13", "gz-msgs10", "gz-plugin2-register"], text=True))
    command = ["c++", "-std=c++17", "-shared", "-fPIC", "-O2", str(WRENCH_SOURCE), "-o", str(library), *flags]
    result = subprocess.run(command, capture_output=True, text=True)
    (out/"compile.stdout").write_text(result.stdout)
    (out/"compile.stderr").write_text(result.stderr)
    write_json(out/"compile_command.json", command)
    if result.returncode:
        raise RuntimeError("Test-only instrumentation compile failed: " + result.stderr[-2000:])
    # gz-common's shared-library search rejects this non-ASCII figure path.
    # Retain the build artifact in the output directory and load an identical temporary ASCII copy.
    runtime_dir = Path(tempfile.mkdtemp(prefix="analysis_joint_instrumentation_"))
    runtime_library = runtime_dir/library.name
    shutil.copy2(library, runtime_library)
    write_json(out/"runtime_library.json", dict(path=str(runtime_library),
               sha256=sha256(runtime_library), archived_path=str(library)))
    return runtime_library


def add_value(parent, tag, value):
    ET.SubElement(parent, tag).text = str(value)


def prepare(case, out, instrumentation):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root, metadata = build_variant(case["mu"], wn=case["wn_rad_s"], zeta=case["zeta"])
    ET.indent(root)
    urdf = out/"aligned_source.urdf"
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)
    assembly = CADAssembly(root)
    conversion = subprocess.run(["gz", "sdf", "-p", str(urdf)], capture_output=True, text=True, check=True)
    (out/"conversion.stderr").write_text(conversion.stderr)
    (out/"converted_source.sdf").write_text(conversion.stdout)
    model = ET.fromstring(conversion.stdout).find("model")
    name, world_name = "joint_cad_probe", "joint_cad_isolation"
    model.set("name", name)
    add_value(model, "pose", f"0 0 10 {case['phi_initial_rad']:.17g} {case['theta_initial_rad']:.17g} {case['psi_initial_rad']:.17g}")
    for plugin in list(model.findall("plugin")):
        if plugin.get("name") != "mmc::SliderServo":
            model.remove(plugin)
        else:
            plugin.find("topic").text = f"/model/{name}/joint/{plugin.findtext('joint_name')}/cmd_pos"
    for link in model.findall("link"):
        for sensor in list(link.findall("sensor")):
            link.remove(sensor)
    for joint in model.findall("joint"):
        if joint.get("name") in {"joint_rotor_upper", "joint_rotor_lower"}:
            joint.set("type", "fixed")
            for axis in list(joint.findall("axis")):
                joint.remove(axis)
    for uri in model.findall(".//uri"):
        if "/meshes/" in (uri.text or ""):
            uri.text = str(WORKSPACE/"src/mmc_uav_description/meshes"/uri.text.split("/meshes/")[-1])
    p = ET.SubElement(model, "plugin", filename="gz-sim-joint-state-publisher-system", name="gz::sim::systems::JointStatePublisher")
    add_value(p, "topic", f"/model/{name}/joint_state")
    add_value(p, "update_rate", 1000)
    p = ET.SubElement(model, "plugin", filename="gz-sim-odometry-publisher-system", name="gz::sim::systems::OdometryPublisher")
    for key, value in dict(odom_frame="world", robot_base_frame="base_link", odom_publish_frequency=1000,
                           odom_topic=f"/model/{name}/odometry", dimensions=3).items():
        add_value(p, key, value)
    p = ET.SubElement(model, "plugin", filename=str(instrumentation), name="mmc::analysis::JointIsolationWrench")
    for key, value in dict(thrust0=assembly.M*9.81, thrust_relative_rise=case["thrust_relative_rise"],
                           transition_time=case["T_s"], enable_nominal_aero=str(case["enable_nominal_aero"]).lower(),
                           initial_omega_body=" ".join(map(str, case["omega_initial_body_rad_s"])),
                           topic=f"/model/{name}/prescribed_wrench").items():
        add_value(p, key, value)
    q0 = joint_coordinates(case["sx_initial_m"], case["sy_initial_m"])
    for joint, value in q0.items():
        node = ET.SubElement(p, "initial_joint"); add_value(node, "name", joint); add_value(node, "position", value)
    initial_records = {r["name"]: r for r in assembly.link_states(q0)}
    for link in assembly.links:
        if not np.allclose(link.joint_rotation, np.eye(3), atol=1e-14):
            raise ValueError("Initial velocity instrumentation expects current aligned link orientations")
        # Retain the nominal actual CoM locations for the per-link audit.
        # Only the shared free group's base velocity is commanded once.
        pose = initial_records[link.name]["p"]
        node = ET.SubElement(p, "initial_link")
        add_value(node, "name", link.name); add_value(node, "position", " ".join(f"{v:.17g}" for v in pose))
    # Run the marked initializer before force servos on its one setup step.
    model.remove(p); model.insert(0, p)
    sdf = ET.Element("sdf", version="1.11")
    world = ET.SubElement(sdf, "world", name=world_name)
    add_value(world, "gravity", "0 0 -9.81")
    physics = ET.SubElement(world, "physics", name="1ms", type="ignored")
    add_value(physics, "max_step_size", .001); add_value(physics, "real_time_factor", 1.)
    for filename, system in [("gz-sim-physics-system", "Physics"), ("gz-sim-user-commands-system", "UserCommands")]:
        ET.SubElement(world, "plugin", filename=filename, name="gz::sim::systems::"+system)
    world.append(model); ET.indent(sdf)
    ET.ElementTree(sdf).write(out/"probe_world.sdf", encoding="utf-8", xml_declaration=True)
    snapshots = out/"source_snapshot"
    snapshots.mkdir(exist_ok=True)
    for source in (SCRIPT,ALIGNMENT_SCRIPT,WRENCH_SOURCE):
        shutil.copy2(source,snapshots/source.name)
    shutil.copy2(instrumentation,out/"instrumentation.so")
    config = dict(**case, model=name, world=world_name, model_alignment=metadata,
                  original_urdf_sha256=sha256(DEFAULT_URDF),
                  source_sha256={str(path.relative_to(WORKSPACE)):sha256(path) for path in (SCRIPT, ALIGNMENT_SCRIPT, WRENCH_SOURCE)},
                  planned_initial_joints=q0,
                  raw_data_policy="All joint positions/velocities, odometry, applied servo forces and prescribed body wrench are measured/published physical telemetry; initialization reset explicitly marked.",
                  frame_convention="raw_physical_body.csv uses direct Physics world linear velocity, body angular velocity and body-origin pose; raw_odometry.csv is auxiliary finite-difference odometry with startup artifacts.",
                  interval_convention="PostUpdate state is stamped at the physical interval end. PreUpdate wrench with stamp t drives the interval (t-0.001,t].",
                  replay_policy="Condition full/omission models on first post-initialization measured state and all actual joint/input histories. Locked rotors are an explicit test isolation, not production semantics.",
                  defaults_modified=False)
    config["physical_link_names"] = [link.name for link in assembly.links]
    config["runtime_instrumentation_sha256"] = sha256(instrumentation)
    write_json(out/"protocol.json", config)
    return config


def stamp(message):
    return message.header.stamp.sec+message.header.stamp.nsec*1e-9


def run(out, config, timeout):
    from gz.transport13 import Node
    from gz.msgs10.double_pb2 import Double
    from gz.msgs10.double_v_pb2 import Double_V
    from gz.msgs10.model_pb2 import Model
    from gz.msgs10.odometry_pb2 import Odometry
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.msgs10.boolean_pb2 import Boolean
    os.environ["GZ_PARTITION"] = f"analysis_joint_cad_{os.getpid()}_{time.time_ns()}"
    plugin_path = WORKSPACE/"install/mmc_gazebo_systems/lib"
    os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"] = str(plugin_path)+os.pathsep+os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    node = Node(); lock = threading.RLock()
    records = dict(joints=[], odometry=[], body=[], links=[], wrench=[], commands=[], initialization=[], errors=[])
    servo = {color: [] for color in COLORS}
    state = dict(start=None, done=False)
    name = config["model"]
    publishers = {joint:node.advertise(f"/model/{name}/joint/{joint}/cmd_pos", Double) for joint in joint_coordinates(0, 0)}
    def joints_callback(message):
        try:
            now = stamp(message)
            with lock:
                if state["start"] is None: state["start"] = now
                elapsed = max(0., now-state["start"])
                targets = target_history(config, elapsed)
                for joint, target in targets.items():
                    if not publishers[joint].publish(Double(data=target)):
                        raise RuntimeError("Physical target publication failed")
                records["commands"].append(dict(stamp_s=now, phase_s=elapsed, **targets))
                for joint in message.joint:
                    records["joints"].append(dict(stamp_s=now, joint_name=joint.name,
                                                   position=joint.axis1.position, velocity=joint.axis1.velocity,
                                                   force=joint.axis1.force))
                if elapsed >= config["duration_s"]: state["done"] = True
        except Exception as exc:
            with lock: records["errors"].append(repr(exc)); state["done"] = True
    def odometry_callback(message):
        try:
            p, q = message.pose.position, message.pose.orientation
            v, w = message.twist.linear, message.twist.angular
            row = dict(stamp_s=stamp(message), frame_id="world", child_frame_id="base_link",
                       x=p.x, y=p.y, z=p.z, qx=q.x, qy=q.y, qz=q.z, qw=q.w,
                       vx=v.x, vy=v.y, vz=v.z, p=w.x, q=w.y, r=w.z)
            with lock: records["odometry"].append(row)
        except Exception as exc:
            with lock: records["errors"].append(repr(exc))
    def wrench_callback(message):
        try:
            row = dict(zip(WRENCH_FIELDS, message.data))
            if len(row) != len(WRENCH_FIELDS): raise RuntimeError("Incomplete applied-wrench telemetry")
            with lock:
                records["initialization" if row["phase_s"] < 0 else "wrench"].append(row)
        except Exception as exc:
            with lock: records["errors"].append(repr(exc))
    def body_callback(message):
        try:
            if len(message.data) != len(BODY_FIELDS): raise RuntimeError("Incomplete physical body telemetry")
            with lock: records["body"].append(dict(zip(BODY_FIELDS, message.data)))
        except Exception as exc:
            with lock: records["errors"].append(repr(exc))
    def link_callback(name):
        def callback(message):
            try:
                if len(message.data) != len(BODY_FIELDS): raise RuntimeError("Incomplete physical link telemetry")
                row = dict(link_name=name, **dict(zip(BODY_FIELDS,message.data)))
                with lock: records["links"].append(row)
            except Exception as exc:
                with lock: records["errors"].append(repr(exc))
        return callback
    def servo_callback(color):
        def callback(message):
            try:
                if len(message.data) != 8: raise RuntimeError("Unexpected servo telemetry")
                row = dict(zip(("stamp_s", "position", "velocity", "requested", "target",
                                "applied_force_N", "desired_acceleration", "load_acceleration"), message.data))
                with lock: servo[color].append(row)
            except Exception as exc:
                with lock: records["errors"].append(repr(exc))
        return callback
    callbacks = {c:servo_callback(c) for c in COLORS}
    link_callbacks = {name:link_callback(name) for name in config["physical_link_names"]}
    node.subscribe(Model, f"/model/{name}/joint_state", joints_callback)
    node.subscribe(Odometry, f"/model/{name}/odometry", odometry_callback)
    node.subscribe(Double_V, f"/model/{name}/prescribed_wrench", wrench_callback)
    node.subscribe(Double_V, f"/model/{name}/prescribed_wrench/body_state", body_callback)
    for link,cb in link_callbacks.items():
        node.subscribe(Double_V,f"/model/{name}/prescribed_wrench/link/{link}",cb)
    for c in COLORS:
        node.subscribe(Double_V, f"/model/{name}/joint/joint_slider_{c}/cmd_pos/servo_state", callbacks[c])
    log = (out/"gazebo.log").open("w")
    process = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(out/"probe_world.sdf")],
                               env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    completed = False
    try:
        deadline = time.monotonic()+30
        while not all(pub.has_connections() for pub in publishers.values()):
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Servo instrumentation subscribers unavailable; inspect gazebo.log")
            time.sleep(.05)
        time.sleep(.25)
        for joint, value in config["planned_initial_joints"].items(): publishers[joint].publish(Double(data=value))
        time.sleep(.05)
        ok, response = node.request(f"/world/{config['world']}/control", WorldControl(pause=False), WorldControl, Boolean, 10000)
        if not ok or not response.data: raise RuntimeError("Cannot unpause physical probe")
        deadline = time.monotonic()+timeout
        while True:
            with lock:
                if state["done"]: break
            if process.poll() is not None or time.monotonic() > deadline: raise RuntimeError("Physical probe timed out")
            time.sleep(.02)
        if records["errors"]: raise RuntimeError(str(records["errors"]))
        if len(records["initialization"]) != 1 or len(records["wrench"]) < 2:
            raise RuntimeError("Missing physical initializer/wrench instrumentation; run is invalid")
        if not all(servo.values()) or len(records["body"]) < 2:
            raise RuntimeError("Incomplete physical state/force observations; run is invalid")
        completed = True
        node.request(f"/world/{config['world']}/control", WorldControl(pause=True), WorldControl, Boolean, 10000)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
        log.close()
        for topic in [f"/model/{name}/joint_state", f"/model/{name}/odometry",
                      f"/model/{name}/prescribed_wrench", f"/model/{name}/prescribed_wrench/body_state",
                      *[f"/model/{name}/prescribed_wrench/link/{link}" for link in link_callbacks],
                      *[f"/model/{name}/joint/joint_slider_{c}/cmd_pos/servo_state" for c in COLORS]]:
            node.unsubscribe(topic)
        time.sleep(.05)
        for key, filename in [("joints", "raw_joint_states.csv"), ("odometry", "raw_odometry.csv"),
                               ("body", "raw_physical_body.csv"),
                               ("links", "raw_physical_links.csv"),
                               ("wrench", "raw_wrench.csv"), ("commands", "published_commands.csv"),
                               ("initialization", "initialization_markers.csv")]:
            if records[key]: write_csv(out/filename, sorted(records[key], key=lambda r:r["stamp_s"]))
        for c in COLORS:
            if servo[c]: write_csv(out/f"raw_servo_{c}.csv", sorted(servo[c], key=lambda r:r["stamp_s"]))
        write_json(out/"callback_errors.json", records["errors"])
        summary = dict(completed=completed, errors=records["errors"],
                       initialization_marker_count=len(records["initialization"]),
                       sample_counts={key:len(value) for key, value in records.items()},
                       measured_state_only=True, rotors_locked=True,
                       model_comparison_completed=False,
                       evidence_scope="Raw locked-rotor prescribed-wrench physical observation; no G0-G4 boundary validation claimed until matched replay.",
                       files_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name != "runtime_summary.json"},
                       default_urdf_unchanged=sha256(DEFAULT_URDF) == config["original_urdf_sha256"])
        write_json(out/"runtime_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--S", type=float, default=.135)
    ap.add_argument("--T", type=float)
    ap.add_argument("--mu", type=float, default=.05/1.45)
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--instrumentation-dir", type=Path, default=WORKSPACE/"results/instrumentation")
    ap.add_argument("--enable-nominal-aero", action="store_true", help="Nominal drag is already enabled by default")
    ap.add_argument("--disable-nominal-aero", action="store_true", help="Explicit diagnostic variant; does not match the final map")
    ap.add_argument("--case-config", type=Path, help="Frozen boundary_validation_cases.json")
    ap.add_argument("--case-key", help="Exact key from the frozen case file")
    ap.add_argument("--mass-capability", action="store_true", help="Held .135 m single-axis capacity test; independent from Fig.8")
    ap.add_argument("--alpha-requirement",type=float,default=.503833)
    ap.add_argument("--run", action="store_true", help="Start physical Gazebo; otherwise prepare only")
    ap.add_argument("--timeout", type=float, default=90.)
    args = ap.parse_args()
    if bool(args.case_config) != bool(args.case_key):
        ap.error("--case-config and --case-key must be supplied together")
    if not args.case_config and args.T is None and not args.mass_capability:
        ap.error("--T or a frozen --case-config/--case-key is required")
    if args.enable_nominal_aero and args.disable_nominal_aero:
        ap.error("Aerodynamic options conflict")
    if args.case_dir.exists() and any(args.case_dir.iterdir()):
        raise SystemExit("Choose a new case directory; existing physical evidence is immutable")
    library = compile_instrumentation(args.instrumentation_dir.resolve())
    case = (selected_case(args.case_config, args.case_key, not args.disable_nominal_aero)
            if args.case_config else mass_capability_case(args.mu,f"mass_capacity_mu{args.mu:.12f}",args.alpha_requirement,not args.disable_nominal_aero)
            if args.mass_capability else frozen_case(args.S, args.T, args.mu, not args.disable_nominal_aero))
    config = prepare(case, args.case_dir.resolve(), library)
    if args.run:
        summary = run(args.case_dir.resolve(), config, args.timeout)
        print(json.dumps(dict(completed=summary["completed"], samples=summary["sample_counts"],
                              initialized_once=summary["initialization_marker_count"] == 1,
                              default_urdf_unchanged=summary["default_urdf_unchanged"])))
    else:
        print(json.dumps(dict(prepared=True, simulation_started=False, case_dir=str(args.case_dir.resolve()))))


if __name__ == "__main__":
    main()
