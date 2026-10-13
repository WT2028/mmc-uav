# Third-party notices

## ROS/Gazebo bridge

`src/ros_gz_bridge` and `src/ros_gz_interfaces` contain code from the ROS/Gazebo
integration project maintained by Open Robotics and its contributors:
<https://github.com/gazebosim/ros_gz>.

Both packages identify version 0.244.23. They are distributed under Apache 2.0,
with the original copyright notices, package metadata and license text retained
in their directories. Local adaptations select Gazebo Harmonic, support the MMC
message subset and entity-wrench path, disable unnecessary upstream test builds,
and initialize the bridge topic-expansion option explicitly.

The root MIT license applies to first-party MMC code and simulation records and
does not replace these Apache 2.0 licenses. ROS, Gazebo, numerical libraries and
other separately installed dependencies retain their own licenses.

## Model exporter notice

The URDF retains the SolidWorks-to-URDF exporter notice included by the model
export tool. CAD geometry and the MMC simulation configuration are supplied as
project assets.
