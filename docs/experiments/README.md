# Paper experiment guide

Run commands from the repository root after building and sourcing the workspace.
`experiments/configs/*.json` is the authoritative configuration for each profile;
`experiments/models/` contains its actual URDF and Gazebo world.

| Paper section | Profiles | Task and conditions | Recorded runs |
|---|---|---|---|
| V.B | `coordinated_motion` | (0,0,2)→(3,3,2) m, 4 s transfer, 40° yaw, calm air | 1 |
| V.C | `scan_nominal`, `scan_mu_low`, `scan_mu_high` | Fixed-total-mass mass-ratio family, calm air, NDO off | 1 per point |
| V.C | `scan_nominal`, `scan_M_low`, `scan_M_high` | Fixed-mass-ratio total-mass family, calm air, NDO off | 1 per point |
| V.C | `scan_omega10`, `scan_nominal`, `scan_omega30` | 10/20/30 rad/s positioning bandwidth, calm air, NDO off | 1 per point |
| V.C | `efficiency_k75`, `efficiency_k85`, `efficiency_k95` | Physical lower-rotor coefficient 0.75/0.85/0.95; nominal controller coefficient 0.85 | 1 per point |
| V.D | `wind_pid`, `wind_lqr`, `wind_cmpc`, `wind_cmpc_ndo` | 3 m Y transfer in 6 s; +X wind ramps to 5 m/s from t=8 s | 3 per controller |
| V.D | `pressure_cmpc`, `pressure_umpc` | 1.5 m X transfer in 2.95 s, seventh-order reference, zero wind, NDO off | 1 per controller |

The nominal mass/bandwidth point is shared across the V.C families, giving seven
independent parameter-scan runs. The coordinated-motion experiment uses its own
controller implementation, `coordinated_control.py`; the comparison and scan
profiles use the explicitly configured experiment factories. This keeps every
experiment tied to the controller used for its supplied records.

## Running a profile

```bash
python3 scripts/run_experiment.py wind_cmpc --output results/wind_cmpc_01
python3 scripts/run_experiment.py wind_cmpc --output results/wind_cmpc_02
python3 scripts/run_experiment.py wind_cmpc --output results/wind_cmpc_03
```

Each invocation is an independent reset. The controller runs for 70 s of its own
time. The output contains `configuration.json`, effective ROS parameters,
controller design, loaded-source hash, flight CSV, independent compressed sensor
histories and completion status. The MPC
profiles also save integral and solver-coordinate diagnostics. A completed
capture indicates that recording finished; tracking performance is calculated
from the measurements.

The comparison factories share the measured-state input, rotor allocation and
physical protection. Within the pressure pair, model, costs, reference,
outer-loop settings and NDO state are common; the predictive inequalities are
the controlled difference. Wind NDO on/off profiles share the same MPC design.
The mass/bandwidth scans align the prediction parameters to the CAD mass family
and retain the nominal effective cost and reference-coordinator settings.

## Windows and limits

- Wind, parameter and efficiency comparisons: controller t∈[8,23] s.
- Pressure comparison: t∈[8,18] s, with the complete later holding record included.
- Parameter-scan hold window: t∈[23,70] s.
- Mechanical travel: ±0.15 m; working command: ±0.135 m; rail speed: ±0.25 m/s;
  applied force: ±2 N.
- Near-boundary occupancy uses 98% of the specified working-position or speed
  bound. The constrained prediction uses the declared nominal force margin.

The pressure comparison is a single-run demanding case. Its UMPC record includes
the subsequent holding failure; both complete traces are retained. The wind
comparison reports the mean and range over three independent runs. Different
experiments have different declared tasks and should not be pooled as repeats.

## Isolated actuator and model tests (V.A)

Recompute the actuator measurements without launching Gazebo:

```bash
mkdir -p results
cp -r data/actuator results/actuator_check
python3 -m analysis.mmc_analysis.actuator_probe \
  --out-dir results/actuator_check --analyze-only
```

Collect a new zero-gravity physical actuator test with the installed force servo:

```bash
python3 -m analysis.mmc_analysis.actuator_probe \
  --out-dir results/new_actuator --case all
```

Recompute the thirteen recorded CAD multibody comparisons:

```bash
python3 -m analysis.mechanisms --multibody
```

Prepare and run an independent short multibody test:

```bash
python3 -m analysis.mmc_analysis.isolation_probe \
  --case-config experiments/multibody_cases.json --case-key map_fast \
  --case-dir results/new_multibody --run
python3 -m analysis.mmc_analysis.replay --case-dir results/new_multibody \
  --aerodynamic-policy nominal --output-dir results/new_multibody_replay
```

These short tests fix the relative rotor joints, apply the declared body-origin
wrench and measure the moving masses and body motion. The replay starts from one
measured initial state and uses the observed rail histories and thrust; each
model evaluates its own state-dependent aerodynamic loads thereafter.

## Design criteria and coupling calculations (III)

```bash
python3 -m analysis.mechanisms
python3 -m analysis.mechanisms --map
```

The first command evaluates the prescribed positioning/attitude requirements and
force/speed-limited responses. The second samples the declared amplitude/time
family using the full model and explicit coupling-term omissions. Its position
and attitude budgets are 2 mm and 0.2° over 0.35 s. `--grid` changes sampling
density; the individual equations and integrator remain unchanged. The default
paper grid has 16 amplitudes from 0 to 0.135 m and 13 logarithmically spaced
transition times from 0.05 to 1.6 s. `--grid N` optionally uses N points on each
axis over the same ranges. The output includes the complete sampling protocol.

The coordinated-flight resource metrics and fifteen local coupling windows can
be recomputed from the independent sensor histories:

```bash
python3 -m analysis.mmc_analysis.coordination_metrics \
  --case-dir data/coordination --output-dir results/coordination_metrics
python3 -m analysis.mmc_analysis.coordination_replay \
  --case-dir data/coordination --output-dir results/coordination_replay
```

The fifteen windows retain both cases exceeding the position budget; the
reported eligible-window count is thirteen. The output records every window.

The physical bandwidth checks cover 10 and 30 rad/s and both sides of the
analytical response-time and speed thresholds:

```bash
python3 -m analysis.mmc_analysis.bandwidth_probe
python3 -m analysis.mmc_analysis.geometry_scaling
```

The first command compares the supplied measurements with the positioning
models. Add `--run` and a new `--output` directory to collect new physical probes.
The second exports the fixed-body-mass and body-size scaling tables used by the
mechanism analysis.

To calculate metrics from a newly completed wind, scan or pressure run:

```bash
python3 -m analysis.closed_loop --run-dir results/wind_cmpc_01 \
  --output results/wind_cmpc_01/metrics.json
```

The runner records the four physical servo channels as well as joint states,
odometry, IMU and wind. These are passive observations and do not enter the
control law. A shortened installation check must cover the complete evaluation
window before metrics can be calculated.
