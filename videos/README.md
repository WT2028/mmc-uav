# Recorded simulation videos

These videos replay the paper's Gazebo simulation records using the CAD meshes
and the Gazebo Harmonic Ogre2 rendering engine. Recorded body poses and
moving-mass displacements drive the animation. A tracking camera preserves
world orientation, while the XY inset shows translation and target holding.

## Coverage

All 17 closed-loop profiles have a full video through controller time 70 s, at
normal speed. The shared nominal run is used for the mass and bandwidth families.
Wind videos show the first recorded repetition for each controller; the paper's
metrics use all three repetitions. The
[repository video index](../README.md#simulation-videos) links to every profile.

`isolated_actuators.mp4` contains the complete single, paired, dual-axis and
constrained actuator tests at half speed. `multibody_checks.mp4` contains all
thirteen 0.35 s Gazebo multibody tests at 20x slow motion. Their chapter boundaries
and source-file checksums are listed in [the video manifest](manifest.json).

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
and teal traces the recorded motion. A scale bar indicates distance. Controller
comparisons use a common map range within each scenario.

## Render the videos

The normal numerical analyses do not require video tools. Rendering additionally
uses FFmpeg, Pillow, the Gazebo Rendering 8 development headers and Ogre2 plugin:

```bash
sudo apt install ffmpeg libgz-rendering8-dev libgz-common5-graphics-dev
python3 -m pip install Pillow
OPENBLAS_NUM_THREADS=1 python3 scripts/video/record.py --all
```

A working Ogre2/EGL graphics driver is required. Offscreen cameras render the
videos directly. Generated clips, poses and render logs go into `results/`.

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
from their supplied histories, joins the complete clips by family, and writes
the videos, previews and manifest to `results/video_package/`. Use `--output`
to select another destination.
