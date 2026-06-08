# MMC UAV Simulation

ROS 2 simulation source code for a moving-mass-controlled coaxial UAV.  The
repository contains the minimum open-source skeleton for the controller,
custom wind messages, and UAV description resources used in the paper's
Gazebo validation section.

> Status: initial public skeleton.  The full development workspace and paper
> drafting assets are intentionally not included here.

## Repository layout

```text
src/
  mmc_control/          ROS 2 Python controller, launch file, bridge config, visualizer
  mmc_interfaces/       Custom wind command/status messages
  mmc_uav_description/  URDF, meshes, and Gazebo worlds

docs/experiments/       Placeholders for paper validation experiment notes
videos/                 Placeholder for recorded validation videos
```

## What is intentionally excluded

- Historical `fly_data` CSV logs and generated plots.
- Paper PDFs, LaTeX files, review images, and manuscript figures.
- `build/`, `install/`, `log/`, caches, and other development artifacts.
- Vendored copies of standard ROS/Gazebo dependencies.
- Real validation videos for now; they will be recorded and added later.

## Dependencies

This project expects a ROS 2 + Gazebo Sim environment with `colcon` and the
standard ROS/Gazebo bridge packages.  Package names vary by ROS/Gazebo distro,
but the runtime dependencies include:

- ROS 2 Python client library: `rclpy`
- ROS messages: `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `std_msgs`,
  `builtin_interfaces`, `actuator_msgs`
- Gazebo bridge/interfaces: `ros_gz_bridge`, `ros_gz_interfaces`
- Python libraries: `numpy`, `scipy`, `pandas`, `matplotlib`
- Gazebo Python transport bindings used by the wind bridge:
  `python3-gz-transport13`

## Build

From the repository root:

```bash
source /opt/ros/<distro>/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Default simulation entry point

```bash
ros2 launch mmc_control mmc_launch.py
```

The launch file resolves resources from installed package shares, so it should
not depend on the original development workspace path.  Generated flight logs
are written under `fly_data/` relative to `MMC_CONTROL_ROOT` when that
environment variable is set, otherwise relative to the source package root for
source-tree runs or the current working directory for installed runs.

## Paper validation scenarios

The paper validation section uses four scenario families.  Video files are not
included in this initial skeleton; see `docs/experiments/README.md` and
`videos/README.md` for placeholders.

- Experiment A: A--B point transfer and holding.
- Experiment B: A--B point transfer and holding with a yaw-step hold.
- Experiment C: parameter scans under the nominal A--B task.
- Experiment D: A--B point transfer and holding under wind disturbance, with
  NDO on/off comparisons.

## License

MIT.  See `LICENSE`.
