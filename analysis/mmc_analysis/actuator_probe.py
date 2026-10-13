#!/usr/bin/env python3
"""Physical, free-floating Gazebo actuator acceptance probe (force-servo measurements).

The only commanded quantity is the servo's position *target*. Joint positions,
velocities, efforts and body odometry are Gazebo measurements. Model/reference
states never replace measured states. No ROS controller or rotor command runs.
Each case starts a separate paused Gazebo transport partition, connects first,
then unpauses. Commands are scheduled on measured Gazebo simulation stamps.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(__file__).resolve().parents[2]
COLORS = ("blue", "red", "purple", "green")
JOINTS = tuple("joint_slider_" + c for c in COLORS)
PARAMS = dict(wn=20.0, zeta=1.2, moving_mass=0.05, velocity_limit=0.25,
              command_limit=0.135, mechanical_limit=0.15, effort_limit=2.0, load_observer_tau=0.02)
# Frozen before any run. Do not tune acceptance thresholds to observed outputs.
TOL = dict(t90_relative_error=0.05, trajectory_rms_10mm_m=0.0003,
           peak_speed_m_s=0.251, peak_position_m=0.1355,
           steady_error_m=0.0001, system_com_drift_m=0.00005,
           synchronized_sample_gap_s=0.002, maximum_sample_gap_s=0.005,
           effort_limit_N=2.0001, minimum_physical_body_displacement_m=0.00001)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9


def transform(xyz, rpy):
    x, y, z = rpy
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    R = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                  [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz
    return T


def urdf_com_geometry(urdf):
    """Mass-weighted zero-position COM and slider direction in base coordinates.

    All model links are retained, including rotor masses. Actual uncommanded
    rotor coordinates are included: physical body recoil can rotate those joints.
    """
    tree = ET.parse(urdf).getroot()
    frames = {"base_link": np.eye(4)}
    pending = list(tree.findall("joint"))
    while pending:
        progress = False
        for j in pending[:]:
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent not in frames:
                continue
            origin = j.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0]*3
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0]*3
            frames[child] = frames[parent] @ transform(xyz, rpy)
            pending.remove(j); progress = True
        if not progress:
            raise RuntimeError("Unresolved URDF frame graph")
    mass, moment, slider_axes = 0., np.zeros(3), {}
    rotor_geometry = {}
    for link in tree.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        m = float(inertial.find("mass").get("value")); origin = inertial.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0]*3
        moment += m * (frames[link.get("name")] @ np.r_[xyz, 1])[:3]; mass += m
    for j in tree.findall("joint"):
        if j.get("name") in JOINTS:
            child = j.find("child").get("link")
            direction = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            m = float(tree.find(f"link[@name='{child}']/inertial/mass").get("value"))
            slider_axes[j.get("name")] = (m / mass * (frames[child][:3, :3] @ direction)).tolist()
    for j in tree.findall("joint"):
        if j.get("name") in ("joint_rotor_upper", "joint_rotor_lower"):
            child=j.find("child").get("link"); inertial=tree.find(f"link[@name='{child}']/inertial")
            m=float(inertial.find("mass").get("value"))
            local=np.array([float(v) for v in inertial.find("origin").get("xyz","0 0 0").split()])
            axis=np.array([float(v) for v in j.find("axis").get("xyz").split()])
            rotor_geometry[j.get("name")] = dict(mass_ratio=m/mass,
                nominal_inertial_offset_base_m=(frames[child][:3,:3]@local).tolist(),
                axis_base=(frames[child][:3,:3]@axis).tolist())
    return dict(total_mass_kg=mass, nominal_com_base_m=(moment / mass).tolist(),
                slider_mass_weighted_directions=slider_axes, rotor_geometry=rotor_geometry)


def prepare_model(urdf, out, name):
    conversion = subprocess.run(["gz", "sdf", "-p", str(urdf)], capture_output=True, text=True, check=True)
    (out / "sdf_conversion.stderr").write_text(conversion.stderr)
    (out / "source_urdf.sdf").write_text(conversion.stdout)
    root = ET.fromstring(conversion.stdout); model = root.find("model"); model.set("name", name)
    # These are test instrumentation/isolation edits to a generated copy only.
    for plugin in list(model.findall("plugin")):
        model.remove(plugin)
    for link in model.findall("link"):
        for sensor in list(link.findall("sensor")):
            link.remove(sensor)  # no render engine or IMU needed for this probe
        wind = link.find("enable_wind")
        if wind is not None:
            wind.text = "false"
    for uri in model.findall(".//uri"):
        if "/meshes/" in (uri.text or ""):
            uri.text = str(ROOT / "src/mmc_uav_description/meshes" / uri.text.split("/meshes/")[-1])
    # Input physics must already match authorized semantics. Never silently fix
    # the source object's damping, force, velocity or travel in a test copy.
    for joint in JOINTS:
        j = model.find(f"joint[@name='{joint}']")
        limit = j.find("axis/limit")
        expected = dict(lower=-PARAMS["mechanical_limit"], upper=PARAMS["mechanical_limit"],
                        velocity=PARAMS["velocity_limit"], effort=PARAMS["effort_limit"])
        for tag, value in expected.items():
            actual = float(limit.findtext(tag))
            if abs(actual - value) > 1e-9:
                raise RuntimeError(f"Source {joint} {tag}={actual}, expected authorized baseline {value}")
        p = ET.SubElement(model, "plugin", filename="mmc-slider-servo-system", name="mmc::SliderServo")
        for key, value in dict(joint_name=joint, topic=f"/model/{name}/joint/{joint}/cmd_pos", **PARAMS).items():
            ET.SubElement(p, key).text = str(value)
    p = ET.SubElement(model, "plugin", filename="gz-sim-joint-state-publisher-system", name="gz::sim::systems::JointStatePublisher")
    ET.SubElement(p, "topic").text = f"/model/{name}/joint_state"
    ET.SubElement(p, "update_rate").text = "1000"
    p = ET.SubElement(model, "plugin", filename="gz-sim-odometry-publisher-system", name="gz::sim::systems::OdometryPublisher")
    for key, value in dict(odom_frame="world", robot_base_frame="base_link", odom_publish_frequency=1000,
                           odom_topic=f"/model/{name}/odometry", dimensions=3).items():
        ET.SubElement(p, key).text = str(value)
    sdf = ET.Element("sdf", version="1.11"); world = ET.SubElement(sdf, "world", name="actuator_alignment")
    ET.SubElement(world, "gravity").text = "0 0 0"
    physics = ET.SubElement(world, "physics", name="1ms", type="ignored")
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    for filename, system in [("gz-sim-physics-system", "Physics"), ("gz-sim-user-commands-system", "UserCommands")]:
        ET.SubElement(world, "plugin", filename=filename, name="gz::sim::systems::" + system)
    world.append(model); ET.indent(sdf)
    ET.ElementTree(sdf).write(out / "probe_world.sdf", encoding="unicode")
    return urdf_com_geometry(urdf)


def sequence(case):
    active = {"single": ("blue",), "pair": ("blue", "red"), "dual": COLORS,
              "constraints": COLORS}[case]
    amplitudes = [0, .01, 0, -.01, 0] if case != "constraints" else [0, .05, 0, .135, -.135, .20, -.20, 0]
    return [dict(requested={c: float(a if c in active else 0) for c in COLORS},
                 hold_s=.5 if i == 0 else (2.0 if case == "constraints" else 1.5))
            for i, a in enumerate(amplitudes)]


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No samples for {path.name}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_case(out, case, urdf, timeout):
    from gz.transport13 import Node
    from gz.msgs10.double_pb2 import Double
    from gz.msgs10.double_v_pb2 import Double_V
    from gz.msgs10.model_pb2 import Model
    from gz.msgs10.odometry_pb2 import Odometry
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.msgs10.boolean_pb2 import Boolean

    out.mkdir(parents=True, exist_ok=True)
    (out/"source_urdf.urdf").write_bytes(urdf.read_bytes())
    name = "actuator_probe_" + case; geometry = prepare_model(urdf, out, name)
    segments = sequence(case); partition = f"mmc_alignment_{os.getpid()}_{case}_{time.time_ns()}"
    # gz transport partition is read when constructing each Node.
    os.environ["GZ_PARTITION"] = partition
    config = dict(case=case, partition=partition, model=name, params=PARAMS, acceptance_tolerances=TOL,
                  segments=segments, geometry=geometry, source_urdf=str(urdf), source_urdf_sha256=sha(urdf),
                  git_head=git("rev-parse", "HEAD").strip(),
                  generated_data=True, gravity=[0, 0, 0], rotor_commands=False,
                  commanded_quantity="servo target only; actual joint states and base odometry recorded",
                  tolerance_policy="Frozen before run; no result-based relaxation")
    write_json(out / "config.json", config)
    critical = [urdf, Path(__file__), ROOT/"src/mmc_gazebo_systems/src/slider_servo.cc",
                ROOT/"src/mmc_control/mmc_control/mmc_uav.py", ROOT/"src/mmc_control/launch/mmc_launch.py"]
    write_json(out/"source_manifest.json", {str(p.relative_to(ROOT)):sha(p) for p in critical if p.exists()})
    servo_source=ROOT/"src/mmc_gazebo_systems/src/slider_servo.cc"
    if servo_source.exists():
        (out/"slider_servo_source.cc").write_bytes(servo_source.read_bytes())
    log = (out / "gazebo.log").open("w")
    process = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(out / "probe_world.sdf")],
                               env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    node = Node(); lock = threading.RLock(); joint_rows, odom_rows, events, errors = [], [], [], []
    state = dict(next_index=0, next_time=0., start=None, done=False)
    telemetry_rows={c:[] for c in COLORS}
    pubs = {c: node.advertise(f"/model/{name}/joint/joint_slider_{c}/cmd_pos", Double) for c in COLORS}

    def joints_callback(msg):
        try:
            js = {j.name: j for j in msg.joint}
            if not all(j in js for j in JOINTS):
                return
            t = stamp(msg)
            with lock:
                if state["start"] is None:
                    state["start"] = t
                rel = t-state["start"]
                if state["next_index"] < len(segments) and rel >= state["next_time"]:
                    seg = segments[state["next_index"]]
                    for c, target in seg["requested"].items():
                        if not pubs[c].publish(Double(data=target)):
                            raise RuntimeError("command publication failed")
                    events.append(dict(sim_time_s=t, relative_sim_time_s=rel, **seg))
                    state["next_index"] += 1; state["next_time"] = rel + seg["hold_s"]
                if state["next_index"] == len(segments) and rel >= state["next_time"]:
                    state["done"] = True
                row = dict(sim_time_s=t)
                for c in COLORS:
                    axis = js["joint_slider_" + c].axis1
                    row.update({c+"_position_m": axis.position, c+"_velocity_m_s": axis.velocity,
                                c+"_force_N": axis.force})
                for rotor in ("upper", "lower"):
                    axis = js.get("joint_rotor_" + rotor)
                    row["rotor_"+rotor+"_position_rad"] = axis.axis1.position if axis else 0.
                joint_rows.append(row)
        except Exception as exc:
            with lock:
                errors.append(repr(exc)); state["done"] = True

    def odom_callback(msg):
        try:
            pose = msg.pose; q = pose.orientation; p = pose.position
            row = dict(sim_time_s=stamp(msg), base_x_m=p.x, base_y_m=p.y, base_z_m=p.z,
                       base_qx=q.x, base_qy=q.y, base_qz=q.z, base_qw=q.w,
                       base_vx_m_s=msg.twist.linear.x, base_vy_m_s=msg.twist.linear.y,
                       base_vz_m_s=msg.twist.linear.z)
            with lock:
                odom_rows.append(row)
        except Exception as exc:
            with lock:
                errors.append(repr(exc))

    def servo_callback(color):
        def callback(msg):
            try:
                values=list(msg.data)
                if len(values)!=8:
                    raise RuntimeError(f"Unexpected {color} servo telemetry length {len(values)}")
                row=dict(zip(("sim_time_s","position_m","velocity_m_s","requested_m","target_m",
                              "applied_force_N","desired_acceleration_m_s2","load_acceleration_m_s2"), values))
                with lock:
                    telemetry_rows[color].append(row)
            except Exception as exc:
                with lock:
                    errors.append(repr(exc)); state["done"]=True
        return callback

    # Keep callback closures alive for transport's lifetime.
    servo_callbacks={c:servo_callback(c) for c in COLORS}
    for c in COLORS:
        node.subscribe(Double_V, f"/model/{name}/joint/joint_slider_{c}/cmd_pos/servo_state", servo_callbacks[c])
    node.subscribe(Model, f"/model/{name}/joint_state", joints_callback)
    node.subscribe(Odometry, f"/model/{name}/odometry", odom_callback)
    try:
        deadline = time.monotonic()+30
        while not all(pub.has_connections() for pub in pubs.values()):
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Servo subscriptions unavailable; inspect gazebo.log")
            time.sleep(.05)
        # Allow subscriber discovery while paused. No state publication starts
        # until the world is unpaused, so subscription registration precedes it.
        time.sleep(.25)
        for pub in pubs.values():
            pub.publish(Double(data=0))
        ok, response = node.request("/world/actuator_alignment/control", WorldControl(pause=False),
                                    WorldControl, Boolean, 10000)
        if not ok or not response.data:
            raise RuntimeError("Unable to unpause isolated world")
        deadline = time.monotonic()+timeout
        while True:
            with lock:
                done = state["done"]
            if done:
                break
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Simulation incomplete or timed out")
            time.sleep(.02)
        if errors:
            raise RuntimeError(f"Transport callback errors: {errors[:5]}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        log.close()
        # Preserve partial raw evidence on failures as well.
        with lock:
            if joint_rows:
                write_csv(out / "joint_raw.csv", sorted(joint_rows, key=lambda r: r["sim_time_s"]))
            if odom_rows:
                write_csv(out / "odometry_raw.csv", sorted(odom_rows, key=lambda r: r["sim_time_s"]))
            for c, samples in telemetry_rows.items():
                if samples:
                    write_csv(out/("servo_"+c+"_raw.csv"), sorted(samples,key=lambda r:r["sim_time_s"]))
            write_json(out / "events.json", events)
            write_json(out / "callback_errors.json", errors)
    return analyze(out)


def ideal_response(t, target, q0):
    wn, zeta = PARAMS["wn"], PARAMS["zeta"]
    r1 = -wn*(zeta-math.sqrt(zeta*zeta-1)); r2 = -wn*(zeta+math.sqrt(zeta*zeta-1))
    return target + (q0-target)*(r2*np.exp(r1*t)-r1*np.exp(r2*t))/(r2-r1)


def reference_t90():
    lo, hi = 0., 2.
    for _ in range(60):
        mid = (lo+hi)/2
        if ideal_response(np.array([mid]), .01, 0)[0] < .009:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def quaternion_matrix(row):
    x,y,z,w = (row["base_q"+c] for c in "xyzw")
    n = math.sqrt(x*x+y*y+z*z+w*w); x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_csv(path):
    if not path.exists() and path.with_name(path.name+'.gz').exists():
        path = path.with_name(path.name+'.gz')
    with (gzip.open(path,'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def analyze(out):
    cfg = json.loads((out / "config.json").read_text()); events = json.loads((out / "events.json").read_text())
    rows = read_csv(out / "joint_raw.csv"); odom = read_csv(out / "odometry_raw.csv")
    t = np.array([r["sim_time_s"] for r in rows]); ot = np.array([r["sim_time_s"] for r in odom])
    checks = []
    def check(name, value, limit, relation="<="):
        passed = bool(value <= limit) if relation == "<=" else bool(value >= limit)
        checks.append(dict(name=name, value=float(value), relation=relation, limit=float(limit), passed=passed))
    check("all_sequence_events", len(events), len(cfg["segments"]), ">=")
    check("max_joint_sample_gap_s", np.max(np.diff(t)), TOL["maximum_sample_gap_s"])
    check("max_odometry_sample_gap_s", np.max(np.diff(ot)), TOL["maximum_sample_gap_s"])
    nominal_t90 = reference_t90(); step_metrics = []
    for i, event in enumerate(events):
        start = event["sim_time_s"]; end = events[i+1]["sim_time_s"] if i+1<len(events) else start+event["hold_s"]
        ids = np.flatnonzero((t>=start)&(t<end))
        if len(ids)<2:
            continue
        for c in COLORS:
            measured=read_csv(out/("servo_"+c+"_raw.csv"))
            # Match this outgoing command to the first exact plugin receive
            # stamp with that requested value, excluding prior segments.
            requested = event["requested"][c]; target = float(np.clip(requested, -.135, .135))
            received=[r for r in measured if start<=r["sim_time_s"]<end and abs(r["requested_m"]-requested)<1e-9]
            if not received:
                check(f"segment_{i}_{c}_command_received",0,1,">=")
                continue
            received_start=received[0]["sim_time_s"]
            check(f"segment_{i}_{c}_command_latency_s",received_start-start,.05)
            q=np.array([r["position_m"] for r in received]); elapsed=np.array([r["sim_time_s"]-received_start for r in received])
            previous_target = float(np.clip(events[i-1]["requested"][c], -.135, .135)) if i else 0.
            delta = target-previous_target
            steady = float(np.mean(q[elapsed>=elapsed[-1]-.2])); error = abs(steady-target)
            check(f"segment_{i}_{c}_steady_error_m", error, TOL["steady_error_m"])
            if abs(delta)<1e-9:
                continue
            crossing = np.flatnonzero((q-previous_target)*np.sign(delta)>=.9*abs(delta))
            t90 = None
            if len(crossing):
                k = int(crossing[0]); t90=float(elapsed[k])
                if k>0:
                    threshold = previous_target+.9*delta
                    t90=float(elapsed[k-1]+(threshold-q[k-1])/(q[k]-q[k-1])*(elapsed[k]-elapsed[k-1]))
            ideal = ideal_response(elapsed, target, previous_target)
            rms = float(np.sqrt(np.mean((q-ideal)**2)))
            metric = dict(segment=i, joint=c, requested_m=requested, clipped_target_m=target,
                          delta_target_m=delta, command_received_sim_time_s=received_start,
                          command_transport_latency_s=received_start-start, t90_s=t90, ideal_t90_s=nominal_t90,
                          trajectory_rms_vs_unlimited_second_order_m=rms, steady_error_m=error)
            step_metrics.append(metric)
            if abs(abs(delta)-.01)<1e-8:
                check(f"segment_{i}_{c}_t90_relative_error", abs(t90/nominal_t90-1) if t90 else 1., TOL["t90_relative_error"])
                check(f"segment_{i}_{c}_trajectory_rms_m", rms, TOL["trajectory_rms_10mm_m"])
    for c in COLORS:
        check(c+"_peak_speed_m_s", max(abs(r[c+"_velocity_m_s"]) for r in rows), TOL["peak_speed_m_s"])
        check(c+"_peak_position_m", max(abs(r[c+"_position_m"]) for r in rows), TOL["peak_position_m"])
        telemetry=read_csv(out/("servo_"+c+"_raw.csv"))
        check(c+"_applied_force_peak_N",max(abs(r["applied_force_N"]) for r in telemetry),TOL["effort_limit_N"])
        clipping_error=max(abs(r["target_m"]-float(np.clip(r["requested_m"],-.135,.135))) for r in telemetry)
        check(c+"_command_clipping_error_m",clipping_error,1e-12)
        check(c+"_max_telemetry_gap_s",max(np.diff([r["sim_time_s"] for r in telemetry])),TOL["maximum_sample_gap_s"])
        # JointState force may be unavailable/zero on a physics backend. Preserve
        # reported force, and distinguish absent feedback from proven effort.
    rotor_excursion=max(abs(r["rotor_"+rotor+"_position_rad"]-rows[0]["rotor_"+rotor+"_position_rad"]) for r in rows for rotor in ("upper","lower"))
    nearest = np.searchsorted(t,ot); nearest=np.clip(nearest,0,len(t)-1)
    before=np.clip(nearest-1,0,len(t)-1)
    nearest=np.where(abs(t[before]-ot)<abs(t[nearest]-ot), before, nearest)
    gaps=abs(t[nearest]-ot)
    check("joint_odom_nearest_timestamp_gap_s", float(np.max(gaps)), TOL["synchronized_sample_gap_s"])
    geom=cfg["geometry"]
    if "rotor_geometry" not in geom:
        # Backfill the kinematics from the identical archived input, preserving
        # the original config and all numerical tolerances.
        source=out/"source_urdf.urdf"
        if not source.exists():
            source=Path(cfg["source_urdf"])
        if sha(source)!=cfg["source_urdf_sha256"]:
            raise RuntimeError("Cannot upgrade COM kinematics: original URDF source SHA unavailable")
        geom=urdf_com_geometry(source)
    com=[]; merged=[]
    for o,k,gap in zip(odom,nearest,gaps):
        row=rows[int(k)]; cb=np.array(geom["nominal_com_base_m"])
        for c in COLORS:
            cb += np.array(geom["slider_mass_weighted_directions"]["joint_slider_"+c])*row[c+"_position_m"]
        for rotor in ("upper","lower"):
            rg=geom["rotor_geometry"]["joint_rotor_"+rotor]
            axis=np.array(rg["axis_base"]); axis/=np.linalg.norm(axis)
            offset=np.array(rg["nominal_inertial_offset_base_m"]); angle=row["rotor_"+rotor+"_position_rad"]
            rotated=offset*math.cos(angle)+np.cross(axis,offset)*math.sin(angle)+axis*np.dot(axis,offset)*(1-math.cos(angle))
            cb+=rg["mass_ratio"]*(rotated-offset)
        base=np.array([o["base_"+a+"_m"] for a in "xyz"]); world=base+quaternion_matrix(o)@cb
        com.append(world); merged.append(dict(**o, joint_sim_time_s=row["sim_time_s"], synchronization_gap_s=float(gap),
                                             **{k:v for k,v in row.items() if k!="sim_time_s"},
                                             **{"system_com_"+a+"_m":float(world[i]) for i,a in enumerate("xyz")}))
    com=np.array(com); drift=np.linalg.norm(com-com[0],axis=1)
    check("system_com_conservation_peak_drift_m",float(np.max(drift)),TOL["system_com_drift_m"])
    base=np.array([[o["base_"+a+"_m"] for a in "xyz"] for o in odom])
    body_displacement=float(np.max(np.linalg.norm(base-base[0],axis=1)))
    check("physical_base_reaction_peak_displacement_m",body_displacement,TOL["minimum_physical_body_displacement_m"],">=")
    write_csv(out / "joint_odometry_com.csv",merged)
    # This reference alone clips integrated velocity. It does not feed Gazebo.
    qref=np.zeros(4); vref=np.zeros(4); reference_rows=[]; event_index=0
    for i,row in enumerate(rows):
        now=row["sim_time_s"]
        while event_index+1<len(events) and events[event_index+1]["sim_time_s"]<=now:
            event_index+=1
        target=np.array([np.clip(events[event_index]["requested"][c],-.135,.135) for c in COLORS])
        dt=0 if i==0 else now-t[i-1]; n=max(1,math.ceil(dt/.0001)); h=dt/n
        for _ in range(n):
            acc=PARAMS["wn"]**2*(target-qref)-2*PARAMS["zeta"]*PARAMS["wn"]*vref
            vref=np.clip(vref+h*acc,-.25,.25); qref+=h*vref
        reference_rows.append(dict(sim_time_s=now, **{c+"_reference_position_m":float(qref[k]) for k,c in enumerate(COLORS)},
                                    **{c+"_reference_velocity_m_s":float(vref[k]) for k,c in enumerate(COLORS)}))
    write_csv(out / "reference_only.csv", reference_rows)
    force_feedback = {c: max(abs(r[c+"_force_N"]) for r in rows) for c in COLORS}
    result=dict(case=cfg["case"], passed=all(c["passed"] for c in checks), checks=checks,
                samples=len(rows), odometry_samples=len(odom), nominal_t90_s=nominal_t90,
                step_metrics=step_metrics, reported_joint_force_peak_N=force_feedback,
                force_feedback_note="Applied effort is verified from servo telemetry; JointState force is separately preserved.",
                max_system_com_drift_m=float(max(drift)), max_base_displacement_m=body_displacement,
                max_uncommanded_rotor_relative_excursion_rad=rotor_excursion,
                com_method="URDF link masses and zero-position frames + measured prismatic coordinates, rotated by actual base odometry; nearest timestamps retained; actual uncommanded rotor coordinates included",
                reference_method="Unlimited overdamped analytical response for 10 mm acceptance; symplectic Euler <=0.1 ms with velocity clipping for constraint reference only",
                com_kinematics=dict(physical_basis="A free-floating body can rotate uncommanded rotor joints through physical recoil; zero rotor command does not imply stationary rotor coordinates.",
                    calculation="Include actual rotor coordinates in full-mass COM kinematics; record uncommanded relative rotation without enforcing artificial stationary joints.",
                    numerical_tolerances_unchanged=True, original_config_preserved=True),
                evidence_sha256={p.name:sha(p) for p in out.iterdir() if p.suffix in (".csv",".sdf",".json",".patch") and p.name!="metrics.json"})
    write_json(out / "metrics.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("single","pair","dual","constraints","all"), default="all")
    parser.add_argument("--urdf", type=Path, default=ROOT/"src/mmc_uav_description/urdf/mmc_uav.urdf")
    parser.add_argument("--timeout", type=float, default=100)
    parser.add_argument("--analyze-only", action="store_true")
    args=parser.parse_args(); cases=("single","pair","dual","constraints") if args.case=="all" else (args.case,)
    results=[]
    for case in cases:
        out=args.out_dir.resolve()/case
        result=analyze(out) if args.analyze_only else run_case(out,case,args.urdf,args.timeout)
        results.append(result)
        print(json.dumps(dict(case=case,passed=result["passed"],samples=result["samples"],
                              failed_checks=[c for c in result["checks"] if not c["passed"]]),ensure_ascii=False))
    write_json(args.out_dir.resolve()/"suite_summary.json",dict(passed=all(r["passed"] for r in results),
                                                                cases={r["case"]:r["passed"] for r in results}))
    return 0 if all(r["passed"] for r in results) else 2


if __name__=="__main__":
    raise SystemExit(main())
