#!/usr/bin/env python3
"""Launch one declared experiment in a fresh ROS/Gazebo domain."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', nargs='?')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--duration', type=float, help='Override duration for an explicitly shortened test')
    parser.add_argument('--timeout', type=float, default=900., help='Wall-clock timeout in seconds')
    parser.add_argument('--domain', type=int, default=None, help='Unused ROS_DOMAIN_ID (default: selected from PID)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    index = json.loads((ROOT / 'experiments/index.json').read_text())
    if args.list:
        for row in index:
            print(f"{row['name']:24} section {row['paper_section']}  n={row['recorded_repetitions']}")
        return
    if args.profile not in {row['name'] for row in index}:
        parser.error('Select a profile shown by --list')
    config = json.loads((ROOT / 'experiments/configs' / (args.profile + '.json')).read_text())
    if args.duration is not None:
        if args.duration <= 0:
            parser.error('--duration must be positive')
        config['duration_controller_s'] = args.duration
        config['duration_overridden'] = True
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    out = (args.output or ROOT / 'results' / (args.profile + '_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    launch = dict(config['launch_arguments'])
    launch.update(enable_gui=str(args.gui).lower(), enable_rviz='false', enable_plotjuggler='false',
                  controller_executable='experiment_node', world_sdf_path=str(ROOT / config['world']),
                  uav_model_path=str(ROOT / config['model']))
    command = ['ros2', 'launch', 'mmc_control', 'mmc_launch.py',
               *(f'{key}:={value}' for key, value in launch.items())]
    if args.dry_run:
        print(json.dumps({'command': command, 'output': str(out), 'configuration': config}, indent=2))
        return
    if out.exists():
        parser.error('Output directory already exists; select a new path')
    out.mkdir(parents=True)
    (out / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'protocol.json').write_text(json.dumps(config['protocol'], indent=2) + '\n')
    shutil.copy2(ROOT / config['model'], out / 'effective_model.urdf')
    parameters = dict(config['ros_parameters'], world_sdf_path=str(ROOT / config['world']))
    (out / 'parameters.yaml').write_text(yaml.safe_dump({'/**': {'ros__parameters': parameters}}))
    domain = args.domain if args.domain is not None else 80 + os.getpid() % 120
    if not 0 <= domain <= 232:
        parser.error('--domain must be between 0 and 232')
    env = dict(os.environ, MMC_EXPERIMENT_CONFIG=str(out / 'configuration.json'),
               MMC_EXPERIMENT_OUT=str(out), MMC_CONTROL_ROOT=str(out), ROS_DOMAIN_ID=str(domain),
               GZ_PARTITION='mmc_' + uuid.uuid4().hex, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    (out / 'command.json').write_text(json.dumps({'command': command, 'ROS_DOMAIN_ID': domain,
                                                 'GZ_PARTITION': env['GZ_PARTITION']}, indent=2) + '\n')
    process = recorder = None
    try:
        with (out / 'launch.log').open('w') as log:
            recorder = subprocess.Popen([sys.executable,str(ROOT/'scripts/record_sensors.py'),str(out)],
                                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started = time.monotonic()
            while not (out/'recorder_ready.json').exists():
                if recorder.poll() is not None or time.monotonic()-started>15:
                    raise RuntimeError(f'Sensor recorder did not start; see {out}/launch.log')
                time.sleep(.1)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            started = time.monotonic()
            while not (out / 'completion.json').exists():
                if recorder.poll() is not None:
                    raise RuntimeError(f'Sensor recorder exited; see {out}/launch.log')
                if process.poll() is not None:
                    raise RuntimeError(f'Launch exited ({process.returncode}); see {out}/launch.log')
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(f'Wall-clock timeout; see {out}/launch.log')
                time.sleep(.5)
    finally:
        for child in (process,recorder):
            if child is None:
                continue
            try:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            except ProcessLookupError:
                pass
    completion = json.loads((out / 'completion.json').read_text())
    if not completion['reached_duration'] or not completion['flight_logs']:
        raise RuntimeError(f'Experiment did not finish: {out}')
    capture=json.loads((out/'sensor_capture.json').read_text())
    if capture['errors'] or not all(capture['counts'].values()):
        raise RuntimeError(f'Incomplete sensor capture: {out}')
    print(out)


if __name__ == '__main__':
    main()
