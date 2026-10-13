# Installation

## Environment

The tested system is Ubuntu 22.04 with ROS 2 Humble, Gazebo Harmonic, GCC 11 and
Python 3.10. ROS and Gazebo must be installed with their package repositories
configured before installing the dependencies below.

```bash
sudo apt install build-essential cmake pkg-config python3-colcon-common-extensions \
  python3-venv python3-pip python3-gz-transport13 \
  libgz-sim8-dev libgz-transport13-dev libgz-msgs10-dev \
  ros-humble-ros-base ros-humble-actuator-msgs ros-humble-vision-msgs \
  ros-humble-rclcpp-components ros-humble-yaml-cpp-vendor
```

Use the system Python 3.10 or a virtual environment that can import ROS's system
packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
source /opt/ros/humble/setup.bash
export GZ_VERSION=harmonic
export CMAKE_BUILD_PARALLEL_LEVEL=2
colcon build --executor sequential --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

In a new terminal, activate the same Python environment, source Humble and source
this workspace's `install/setup.bash`. Avoid sourcing another MMC workspace in
that terminal. `ros2 pkg prefix mmc_control` and `ros2 pkg prefix ros_gz_bridge`
should point inside this repository's installation.

The bundled bridge defaults to Harmonic and builds the message families used by
this project, including `EntityWrench`. Its CMake options are
`ROS_GZ_BRIDGE_MMC_MINIMAL=ON` and `ROS_GZ_BRIDGE_BUILD_UPSTREAM_TESTS=OFF`.
The default Humble/Fortress bridge is not a substitute for this build.

## Tested numerical library versions

| Library | Version |
|---|---|
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| pandas | 1.3.5 |
| Matplotlib | 3.8.4 |
| CasADi | 3.7.2 |
| OSQP | 1.1.1 |
| PyYAML | 6.0.3 |

The verified Gazebo components are gz-sim 8.15.0, gz-transport 13.6.0 and gz-msgs
10.4.0. Exact numerical trajectories can vary with simulator scheduling and
library versions; the provided records and metrics give a fixed reference.

## Runtime

A headless run needs no display. `--gui` or the default demonstration requires a
working display/OpenGL setup. Gazebo plugins are found through the environment
hooks installed by `mmc_gazebo_systems`.

Run one simulation at a time. The runner uses a separate Gazebo partition and ROS
domain, single-threaded BLAS, and a bounded wall-clock timeout. `--domain` selects
an explicit unused domain when another ROS application is present. Keep four
physical CPU cores available for simulation and use a modest build parallelism.

Optional tools such as RViz, PlotJuggler and keyboard teleoperation are not needed
for the headless paper experiments.
