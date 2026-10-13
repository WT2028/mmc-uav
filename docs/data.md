# Data and numerical methods

`data/closed_loop/index.json` links each of the 25 recorded runs to an experiment
profile, repeat number, numeric files, checksums and unrounded reference metrics.
The CSV values retain their recorded precision. Display rounding is applied only
to reports; it does not change parameters or calculations.

## Closed-loop records

- `flight.csv.gz`: all recorded time samples for the selected position, attitude,
  reference, actuator, rotor, observer, solver and timing signals. `Time` is
  controller time in seconds; position is in metres, angles in degrees, angular
  rates in degrees/s, rail speed in m/s and rotor speed in rad/s.
- `servo.csv.gz`: independent four-rail physical telemetry for the parameter and
  pressure comparisons. Its native simulation clock is aligned to controller
  time using the recorded clock offset. Position, velocity and force are actual
  Gazebo states/forces, in m, m/s and N.
- Metadata contains the SHA-256 of each distributed container and the original
  flight container. The distributed CSV is a column projection retaining every
  sample and its original numeric text. Machine-specific paths and repeated
  status strings are omitted.

Position RMSE integrates squared three-dimensional error against the declared
continuous reference using trapezoidal time weighting and interpolated window
endpoints. The wind/scan task uses a quintic reference; the pressure task uses a
seventh-order reference. Attitude RMSE uses wrapped angle errors against the
actual inner-loop reference. Rotor utilization is measured speed / 450 rad/s.

Pressure occupancy interpolates each physical rail onto their common union of
timestamps, tests whether any rail reaches the 98% threshold, and integrates
that binary sequence. Parameter-scan occupancy integrates threshold crossings of
the piecewise-linear maximum rail magnitude. These definitions are kept
explicit in `analysis/closed_loop.py`.

```bash
python3 -m analysis.closed_loop
```

This independently calculates the reported metrics and checks 100 values against
the included references with absolute tolerance 1e-6. It writes per-run values,
group means/ranges and every check to `results/closed_loop_metrics.json`.

## Actuator and multibody records

`data/actuator/` contains measured joint, servo and body histories for single,
paired, dual-axis and constrained positioning tests, with command events and the
input URDF. `data/multibody/` contains thirteen isolated physical tests spanning
representative cases and both sides of the coupling boundaries. Each includes
the initial-state marker, raw joint/body motion, wrench, servo states and URDF.

`analysis/mmc_analysis/` contains the six-DOF equations, CAD assembly calculation,
finite-horizon omission models, force-servo probe code and replay integration.
The replay compares a model propagated from its initial condition with the later
Gazebo body measurements; the measured body state is not repeatedly reset into
the model. The records therefore support a numerical model-consistency check.

The repository provides numerical simulation evidence. Physical mechanism
photographs in the paper describe the implementation; the datasets here are
Gazebo measurements. Recorded full holding histories and experiment repetition
counts remain available when interpreting the reported windows.

Generated results, builds and new simulation logs are ignored by Git. The
supplied datasets are versioned; use a new output directory for each run.

Run `python3 scripts/verify_sources.py` to check all distributed source, model,
configuration and data files against `reproducibility.json`.
