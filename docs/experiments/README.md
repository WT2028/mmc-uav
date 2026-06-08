# Validation experiment placeholders

This directory documents the four simulation experiment families used for the
paper validation section.  The actual videos will be recorded later and placed
under `videos/`.

## Experiment A — nominal A--B point transfer and holding

Purpose: validate nominal spatial tracking and final holding accuracy.

Suggested launch:

```bash
ros2 launch mmc_control mmc_launch.py \
  auto_scene_mode:=hover_to_point_hold \
  auto_scene_target_x:=0.0 \
  auto_scene_target_y:=3.0 \
  auto_scene_target_z:=2.0
```

## Experiment B — A--B transfer plus yaw-step hold

Purpose: validate translation--yaw coordination under the shared constrained
input domain.

Suggested launch:

```bash
ros2 launch mmc_control mmc_launch.py \
  auto_scene_mode:=hover_to_point_yaw_step_hold \
  auto_scene_yaw_step_deg:=90.0 \
  auto_scene_yaw_ramp_duration:=5.0
```

## Experiment C — parameter scans under the nominal A--B task

Purpose: compare moving-mass actuator bandwidth and individual moving-mass
settings under the same point-transfer task.

Example bandwidth scan:

```bash
for wn in 10.0 20.0 30.0; do
  ros2 launch mmc_control mmc_launch.py \
    auto_scene_mode:=hover_to_point_hold \
    slider_wn_mass:=${wn}
done
```

Example mass-ratio scan:

```bash
for m in 0.035 0.050 0.075; do
  ros2 launch mmc_control mmc_launch.py \
    auto_scene_mode:=hover_to_point_hold \
    moving_mass_kg:=${m}
done
```

## Experiment D — wind disturbance and NDO on/off comparison

Purpose: compare disturbance-compensation benefit and resource cost under
constant wind with turbulence.

Example for the 3 m/s wind world:

```bash
WORLD=$(ros2 pkg prefix mmc_uav_description)/share/mmc_uav_description/worlds/wind_x3_hover_hold_world.sdf

ros2 launch mmc_control mmc_launch.py \
  world_sdf_path:=${WORLD} \
  enable_wind_bridge:=true \
  ndo_enabled:=true

ros2 launch mmc_control mmc_launch.py \
  world_sdf_path:=${WORLD} \
  enable_wind_bridge:=true \
  ndo_enabled:=false
```

For the 5 m/s case, replace the world with
`wind_x5_hover_hold_world.sdf`.
