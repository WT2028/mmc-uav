#!/usr/bin/env python3
"""Common task and outer-controller settings for the actuator-limited comparison."""
from __future__ import annotations

import importlib.util
import copy
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys
import types

from .actuator_common import ADAPTER, read, save, sha
from .actuator_reference import task_functions
from .actuator_outer_braking import install_shared_outer, validate_shared_outer
from .actuator_terminal_capture import SETTING as CAPTURE_SETTING, install_terminal_capture
from .actuator_reference_budget import install_shared_reference, validate_shared_reference


from . import controller_factory as original


def synchronize_reference_horizon(node, selected):
    """Keep existing coordinator algorithms/state; match selected MPC row count."""
    if node is None:
        return None
    horizon, sample = int(selected.Np), float(selected.dt)
    records = {}
    for name in ('pitch_axis_coordinator', 'roll_axis_coordinator'):
        coordinator = getattr(node, name)
        before = asdict(coordinator.cfg)
        if float(coordinator.cfg.dt) != sample or sample != .01:
            raise ValueError('Actuator reference and selected MPC must share the unchanged 10 ms sample')
        state = coordinator.state
        if int(coordinator.cfg.prediction_horizon) != horizon:
            coordinator.cfg = replace(coordinator.cfg, prediction_horizon=horizon)
        after = asdict(coordinator.cfg)
        changed = [key for key in before if before[key] != after[key]]
        if set(changed) - {'prediction_horizon'} or coordinator.state is not state:
            raise ValueError('Only coordinator reference row count may change in the experiment entry')
        records[name] = {'before': before, 'after': after, 'changed_fields': changed,
            'sample_time_s_verified_unchanged': after['dt'] == before['dt'] == sample,
            'existing_coordinator_state_preserved': coordinator.state is state,
            'configuration_updated_with_dataclasses_replace': bool(changed)}
    return {'schema': 'mmc-task-reference-horizon-synchronization-v1',
        'selected_prediction_steps': horizon, 'selected_model_sample_time_s': sample,
        'coordinators': records, 'all_passed': all(
            row['after']['prediction_horizon'] == horizon
            and row['sample_time_s_verified_unchanged'] and row['existing_coordinator_state_preserved']
            for row in records.values()),
        'control_or_coordinator_algorithm_modified': False,
        'coordinator_limits_or_gains_modified': False,
        'purpose': 'Match the existing production reference sequence row count to the declared common experimental MPC horizon before any control callback runs'}


def make_controller(kind, production_mpc, configuration_dict=None, node=None):
    full_configuration = configuration_dict or {}
    shared_outer = full_configuration.get('shared_outer')
    shared_reference = full_configuration.get('shared_reference')
    outer_proof = None
    if shared_outer is not None:
        validate_shared_outer(shared_outer)
        if node is None:
            raise ValueError('The declared shared outer override needs the existing actual node')
        outer_proof = install_shared_outer(node.outer, shared_outer)
        if CAPTURE_SETTING in shared_outer:
            outer_proof['all_existing_outer_attributes_and_state_unchanged_before_terminal_capture_install'] = outer_proof['all_existing_outer_attributes_and_state_unchanged_at_install']
            capture_proof = install_terminal_capture(node.outer, {CAPTURE_SETTING: shared_outer[CAPTURE_SETTING]})
            outer_proof['configuration'].update(capture_proof['configuration'])
            outer_proof['terminal_capture_provenance'] = capture_proof
            outer_proof['all_existing_outer_attributes_and_state_unchanged_at_install'] = (
                outer_proof['all_existing_outer_attributes_and_state_unchanged_at_install']
                and not capture_proof['changed_existing_outer_attributes'])
            if 'horizontal_gain_provenance' in outer_proof:
                outer_proof['all_existing_outer_attributes_except_declared_horizontal_gain_unchanged_at_install'] = not capture_proof['changed_existing_outer_attributes']
            outer_proof['all_existing_outer_attributes_except_declared_horizontal_gain_and_terminal_capture_unchanged_at_install'] = capture_proof['all_other_existing_outer_attributes_and_state_preserved_by_identity']
            outer_proof['terminal_capture_installed_after_horizontal_gain'] = 'horizontal_gain_provenance' in outer_proof
            outer_proof['maintained'][-2] = 'All undeclared outer gains, original stationary damping gain/equation, integral washout radius, integrals, filters, deadbands and authority'
    if shared_reference is not None:
        validate_shared_reference(shared_reference)
    core_configuration = {key: value for key, value in full_configuration.items() if key not in ('shared_outer', 'shared_reference')}
    selected = original.make_controller(kind, production_mpc, core_configuration, node=node)
    selected.configuration = full_configuration
    horizon_proof = synchronize_reference_horizon(node, selected)
    reference_proof = None
    if shared_reference is not None:
        reference_proof = install_shared_reference(node, selected, shared_reference,
            Path(original.control.__file__).resolve())
    out = Path(os.environ['MMC_COMPARISON_OUT'])
    if outer_proof is not None:
        outer_proof.update(full_declared_shared_configuration=copy.deepcopy(full_configuration),
            original_factory_supported_configuration=copy.deepcopy(core_configuration),
            wrapper_file=str(Path(__file__).resolve()), wrapper_sha256=sha(Path(__file__)))
        save(out / 'shared_outer_provenance.json', outer_proof)
        if 'terminal_capture_provenance' in outer_proof:
            save(out / 'shared_terminal_capture_provenance.json', outer_proof['terminal_capture_provenance'])
    if reference_proof is not None:
        reference_proof.update(full_declared_shared_configuration=copy.deepcopy(full_configuration),
            original_factory_supported_configuration=copy.deepcopy(core_configuration),
            wrapper_file=str(Path(__file__).resolve()), wrapper_sha256=sha(Path(__file__)),
            original_control_adapter_file=str(ADAPTER), original_control_adapter_sha256=sha(ADAPTER))
        save(out / 'shared_reference_provenance.json', reference_proof)
    if outer_proof is not None or reference_proof is not None:
        loaded = read(out / 'loaded_tracking_design.json')
        loaded.update(full_declared_shared_configuration=copy.deepcopy(full_configuration),
            original_factory_supported_configuration=copy.deepcopy(core_configuration),
            shared_outer_provenance_file='shared_outer_provenance.json' if outer_proof else None,
            shared_reference_provenance_file='shared_reference_provenance.json' if reference_proof else None)
        save(out / 'loaded_tracking_design.json', loaded)
    previous_record = selected.design_record
    def record():
        result = previous_record()
        result['configuration'] = copy.deepcopy(full_configuration)
        result['shared_design']['configuration'] = copy.deepcopy(full_configuration)
        if outer_proof is not None:
            result['shared_design']['shared_outer'] = copy.deepcopy(shared_outer)
            result['shared_design']['original_factory_supported_configuration'] = copy.deepcopy(core_configuration)
            result['shared_outer'] = copy.deepcopy(outer_proof)
            result['effective_outer_law_identical_to_common'] = False
        if reference_proof is not None:
            result['shared_design']['shared_reference'] = copy.deepcopy(shared_reference)
            result['shared_design']['original_factory_supported_configuration'] = copy.deepcopy(core_configuration)
            result['shared_reference'] = copy.deepcopy(reference_proof)
            result['effective_coordinator_reference_identical_to_common'] = False
        return result
    selected.design_record = record
    task = read(out / 'protocol.json')
    needs_declared_task = task.get('trajectory_mode') == 'point_round_trip' or task.get('polynomial_order', 5) != 5
    if needs_declared_task:
        if node is None:
            raise ValueError('The declared task-only planner wrapper needs the actual existing node')
        def functions(self):
            return task_functions(task)
        node._default_trajectory_functions = types.MethodType(functions, node)
        installed = True
    else:
        installed = False
    if horizon_proof is not None:
        horizon_proof.update(wrapper_file=str(Path(__file__).resolve()), wrapper_sha256=sha(Path(__file__)),
            original_control_adapter_file=str(ADAPTER), original_control_adapter_sha256=sha(ADAPTER))
        save(out / 'reference_horizon_provenance.json', horizon_proof)
    task_proof = {
        'schema': 'mmc-task-task-only-reference-wrapper-v1',
        'wrapper_file': str(Path(__file__).resolve()), 'wrapper_sha256': sha(Path(__file__)),
        'original_control_adapter_file': str(ADAPTER), 'original_control_adapter_sha256': sha(ADAPTER),
        'protocol_file_sha256': sha(out / 'protocol.json'),
        'trajectory_mode': task.get('trajectory_mode', 'point_hold'),
        'reference_functions_overridden_for_round_trip': installed and task.get('trajectory_mode') == 'point_round_trip',
        'reference_functions_overridden_for_declared_polynomial': installed and task.get('polynomial_order', 5) != 5,
        'actual_polynomial_order': task.get('polynomial_order', 5),
        'position_blend_coefficients_ascending': task.get('position_blend_coefficients_ascending', [0., 0., 0., 10., -15., 6.]),
        'hook': 'Instance _default_trajectory_functions; existing UniversalTrajectoryPlanner set_mission/get_ref unchanged',
        'common_outer_and_inner_preview_receive_same_planner': True,
        'reference_horizon_provenance_file': 'reference_horizon_provenance.json' if horizon_proof else None,
        'controller_law_override_in_this_wrapper': outer_proof is not None,
        'attitude_MPC_control_law_override_in_this_wrapper': False,
        'outer_control_law_override_in_this_wrapper': outer_proof is not None,
        'shared_outer_configuration': copy.deepcopy(shared_outer),
        'shared_outer_provenance_file': 'shared_outer_provenance.json' if outer_proof else None,
        'coordinator_reference_parameters_overridden_in_this_wrapper': reference_proof is not None,
        'shared_reference_configuration': copy.deepcopy(shared_reference),
        'shared_reference_provenance_file': 'shared_reference_provenance.json' if reference_proof else None,
        'truth_or_observer_inputs_used': False}
    if reference_proof is not None and shared_reference.get('monotonic_direction_reset_enabled', True) is False:
        task_proof.update(controller_law_override_in_this_wrapper=True,
            coordinator_axis_shaping_algorithm_overridden_in_this_wrapper=True,
            shared_reference_axis_shaping_algorithm_modified=True,
            instance_overridden_coordinator_method='_update_axis_state')
    save(out / 'task_reference_provenance.json', task_proof)
    return selected


def finalize(out):
    return original.finalize(out)
