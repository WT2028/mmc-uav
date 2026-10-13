"""Run a declared paper experiment through the normal ROS controller node."""
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import rclpy


def main():
    config = json.loads(Path(os.environ['MMC_EXPERIMENT_CONFIG']).read_text())
    out = Path(os.environ['MMC_EXPERIMENT_OUT']).resolve()
    os.environ['MMC_COMPARISON_OUT'] = str(out)
    if config['family'] == 'coordination':
        from mmc_control.experiments import coordinated_control as control
        factory = None
    else:
        from mmc_control import mmc_uav as control
        from mmc_control.experiments import controller_factory as factory
        if config['family'] == 'pressure':
            from mmc_control.experiments import pressure_factory as factory

    if config['family'] == 'scan':
        params = config['model_alignment']['prediction_params']
        original_params_init = control.Params.__init__
        def aligned_params_init(self, *args, **kwargs):
            original_params_init(self, *args, **kwargs)
            for key, value in params.items():
                setattr(self, key, value)
        control.Params.__init__ = aligned_params_init
        original_cost = control.build_slider_actuator_cost_profile
        def fixed_cost(physical, **kwargs):
            nominal = copy.copy(physical)
            nominal.wn_mass, nominal.zeta_mass = 20., 1.2
            profile = original_cost(nominal, **kwargs)
            return dataclasses.replace(profile, wn_mass=physical.wn_mass,
                                       zeta_mass=physical.zeta_mass)
        control.build_slider_actuator_cost_profile = fixed_cost

    old_init = control.MMCUAVROS2Controller.__init__
    old_inner = control.MMCUAVROS2Controller.inner_loop_cb
    reached_duration = False
    def configured_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if factory is not None:
            if config['controller'] != 'constrained_mpc' and self.ndo_enabled:
                raise ValueError('Baseline and unconstrained controllers require NDO off')
            if config['family'] == 'scan':
                for coordinator in (self.pitch_axis_coordinator, self.roll_axis_coordinator):
                    coordinator.cfg.ref_wn, coordinator.cfg.ref_zeta = 32., 1.2
            self.att_mpc = factory.make_controller(config['controller'], self.att_mpc,
                                                   config['configuration'], node=self)
            design = self.att_mpc.design_record()
            if config['family'] == 'scan':
                owner = self.att_mpc.production_mpc
                actual = {'Q': np.diag(owner.Q_y).tolist(), 'R': np.diag(owner.R_u).tolist(),
                          'Rdelta': np.diag(owner.R_delta).tolist()}
                expected = {'Q': [1400,1400,600,100,100,200,.05,.2,.05,.2],
                            'R': [1000,.15,.15], 'Rdelta': [1,1.2,1.2]}
                if any(not np.allclose(actual[k], expected[k], rtol=0, atol=1e-12) for k in expected):
                    raise ValueError('Effective scan weights do not match the declared design')
                design['effective_scan_cost'] = actual
            (out / 'controller_design.json').write_text(json.dumps(design, indent=2) + '\n')
        parameters = {name: parameter.value for name, parameter in self._parameters.items()}
        (out / 'effective_ros_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')

    def bounded_inner(self, *args, **kwargs):
        nonlocal reached_duration
        result = old_inner(self, *args, **kwargs)
        if self.flight_data_log and self._elapsed_sec() >= config['duration_controller_s']:
            reached_duration = True
            raise control.ExternalShutdownException()
        return result

    control.MMCUAVROS2Controller.__init__ = configured_init
    control.MMCUAVROS2Controller.inner_loop_cb = bounded_inner
    if factory is not None:
        trigger = config['protocol']['wind_trigger_controller_s']
        control.MMCUAVROS2Controller._move_start_ready_for_runtime_wind_activation = (
            lambda self, summary, t: float(t) >= trigger)
    control.get_default_log_dir = lambda: out
    source = Path(control.__file__)
    (out / 'loaded_controller.json').write_text(json.dumps({
        'module': control.__name__, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'profile': config['name']}, indent=2) + '\n')
    try:
        control.main(args=[*sys.argv[1:], '--ros-args', '--params-file', str(out / 'parameters.yaml')])
    finally:
        if factory is not None:
            factory.finalize(out)
        (out / 'completion.json').write_text(json.dumps({
            'requested_controller_duration_s': config['duration_controller_s'],
            'reached_duration': reached_duration,
            'flight_logs': [p.name for p in out.glob('mmc_flight*.csv')]}, indent=2) + '\n')
