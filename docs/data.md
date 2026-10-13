# Data and numerical methods

The supplied Gazebo records support the paper's actuator tests, dynamics
comparisons, and closed-loop evaluations. `data/closed_loop/index.json` links
each of the 25 flight records to its experiment profile, repeat number, numeric
files, checksums and reference metrics. Calculations use the stored full-precision
values; rounding is applied to displayed results.

## Closed-loop records

- `flight.csv.gz`: all recorded time samples for the selected position, attitude,
  reference, actuator, rotor, observer, solver and timing signals. `Time` is
  controller time in seconds; position is in metres, angles in degrees, angular
  rates in degrees/s, rail speed in m/s and rotor speed in rad/s.
- `servo.csv.gz`: independent four-rail Gazebo telemetry for the parameter and
  pressure comparisons. Its native simulation clock is aligned to controller
  time using the recorded clock offset. Position, velocity and force are actual
  Gazebo states/forces, in m, m/s and N.
- Metadata contains the SHA-256 of each distributed container and the original
  flight container. The distributed CSV is a column projection retaining every
  sample and its original numeric text for the analysis signals.

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

`data/actuator/` contains Gazebo joint, servo and body histories for single,
paired, dual-axis and constrained positioning tests, with command events and the
input URDF. `data/multibody/` contains thirteen isolated Gazebo tests spanning
representative cases and both sides of the coupling boundaries. Each includes
the initial-state marker, raw joint/body motion, wrench, servo states and URDF.

`analysis/mmc_analysis/` contains the six-DOF equations, CAD assembly calculation,
finite-horizon omission models, force-servo probe code and replay integration.
Starting from the measured initial state, the model propagates body motion
continuously using the recorded joint histories and applied inputs. Comparing
these predictions with Gazebo body measurements checks the analytical dynamics
against multibody simulation under matched inputs. The
[experiment guide](experiments/README.md) specifies the evaluation windows and
repetition counts.

Generated results, builds and new simulation logs are ignored by Git. The
supplied datasets are versioned; use a new output directory for each run.

Run `python3 scripts/verify_sources.py` to check all distributed source, model,
configuration and data files against `reproducibility.json`.
