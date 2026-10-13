# Recorded simulation videos

These videos visualize the paper's supplied simulation records with the CAD
meshes and the Gazebo Harmonic Ogre2 rendering engine. No controller or physics
simulation is rerun when making a video. Body position and attitude, and moving
mass displacements, come from the recorded measurements. The following camera
preserves world orientation; the XY inset makes translation and target holding
visible throughout the recording.

## Coverage

All 17 closed-loop profiles have a full video through controller time 70 s, at
normal speed. The shared nominal run is used for the mass and bandwidth families.
Wind videos show the first recorded repetition for each controller, while the
paper's metrics use all three repetitions. The complete UMPC holding segment is
included. See the [repository video index](../README.md#simulation-videos) for
links to every profile.

`isolated_actuators.mp4` contains the complete single, paired, dual-axis and
constrained actuator tests at half speed. `multibody_checks.mp4` contains all
thirteen 0.35 s physical tests at 20x slow motion. Their chapter boundaries and
source-file checksums are listed in [the video manifest](manifest.json).

Frames are interpolated at 20 fps. Attitude interpolation uses quaternions. Four
independent mass positions are used where the distributed records provide them;
otherwise each pair uses its recorded paired displacement. For flight records
without rotor phase, visual phase is integrated from measured speed using the
model's 10:1 rotor visualization slowdown. The coordinated and isolated tests
use recorded rotor joint angles. The displayed time and state values are taken
from the same history as the rendered vehicle.

Overlays show the experiment, simulation time, position (XYZ, metres), and
roll/pitch/yaw (RPY, degrees). Slowed clips also show their playback rate. In the
XY inset, +X points right and +Y points up; A and B mark the start and target,
and teal traces the recorded motion. A scale bar indicates distance. Comparison
profiles share the same map range, including the complete pressure-test motion.

## Render the videos

The normal numerical analyses do not require video tools. Rendering additionally
uses FFmpeg, Pillow, the Gazebo Rendering 8 development headers and Ogre2 plugin:

```bash
sudo apt install ffmpeg libgz-rendering8-dev libgz-common5-graphics-dev
python3 -m pip install Pillow
OPENBLAS_NUM_THREADS=1 python3 scripts/video/record.py --all
```

A working Ogre2/EGL graphics driver is required. Rendering uses offscreen
cameras and does not open or record the desktop. Generated clips, poses and
render logs go into `results/`; supplied video files are left untouched by these
commands.

```bash
python3 scripts/video/record.py coordinated_motion
python3 scripts/video/record.py wind_cmpc_ndo --repeat 1
python3 scripts/video/record.py actuator_single actuator_pair actuator_dual actuator_constraints
python3 scripts/video/record.py multibody_map_weak multibody_map_fast multibody_map_slow
```

For the complete positioning and multibody compilations, preview GIFs, and the
source manifest, run the packaging script after the closed-loop render:

```bash
python3 scripts/video/package.py
```

The command renders all four actuator tests and all thirteen multibody tests
from their supplied histories, concatenates each family without dropping frames,
and writes everything under `results/video_package/`. No simulation records are
modified. `--output` selects a different directory for deliberate publication.
