#!/usr/bin/env python3
"""Declared common reference budgets and a guarded optional instance hook."""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
from types import MethodType

ORIGINAL_CONTROL_SHA256 = '266c2773f4eeb2eb790f3b0ec1dacc00b6c248236c5e76f04e7a7b6b96654c10'
FIELD_MAP = {'slider_rate_limit_m_s': 'slider_rate_limit',
             'angle_accel_limit_rad_s2': 'ref_accel_limit'}
OPTIONAL_FIELD_MAP = {'reference_natural_frequency_rad_s': 'ref_wn'}
METHODS = ('_update_axis_state', '_update_slider_state', 'build_reference_sequence')
ANGLE_ACCEL_SCOPE = ('Acceleration term in the existing shaped-angle update before its original monotonicity/rate clipping; '
                     'actual reference differences can also reflect unchanged reversal resets, measurement blending and MPC integral-rate trim')
DIRECTION_RESET_DISABLED_ACCEL_SCOPE = ('Acceleration term in the same damped shaped-angle update with the two raw-direction angle-hold/rate-reset branches removed; '
    'rate clipping is preserved, while final reference differences can still reflect unchanged measurement blending and MPC integral-rate trim')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_shared_reference(settings):
    optional = {'monotonic_direction_reset_enabled'} | set(OPTIONAL_FIELD_MAP)
    if not isinstance(settings, dict) or not set(FIELD_MAP) <= set(settings) or set(settings) - set(FIELD_MAP) - optional:
        raise ValueError('shared_reference requires both budgets and only the optional direction-reset flag and declared natural frequency')
    result = {}
    for name, maximum in (('slider_rate_limit_m_s', .25), ('angle_accel_limit_rad_s2', 320.)):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0. < value <= maximum:
            raise ValueError(name + ' must be finite, positive and no greater than ' + str(maximum))
        result[name] = float(value)
    if 'monotonic_direction_reset_enabled' in settings:
        value = settings['monotonic_direction_reset_enabled']
        if type(value) is not bool:
            raise ValueError('monotonic_direction_reset_enabled must be a strict boolean')
        result['monotonic_direction_reset_enabled'] = value
    if 'reference_natural_frequency_rad_s' in settings:
        result['reference_natural_frequency_rad_s'] = _validated_reference_frequency(settings['reference_natural_frequency_rad_s'])
    return result


def _validated_reference_frequency(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0. < value <= 32.:
        raise ValueError('reference_natural_frequency_rad_s must be finite, positive and no greater than the original32rad/s')
    return float(value)


def configuration_field_mapping(settings):
    return {**FIELD_MAP, **{key: field for key, field in OPTIONAL_FIELD_MAP.items() if key in settings}}


def angle_acceleration_budget_scope(settings):
    scope = ANGLE_ACCEL_SCOPE if settings.get('monotonic_direction_reset_enabled', True) else DIRECTION_RESET_DISABLED_ACCEL_SCOPE
    if 'reference_natural_frequency_rad_s' in settings:
        scope += '; the same damped equation uses the explicitly declared ref_wn natural-frequency parameter'
    return scope


def reference_change_scope(settings):
    if 'reference_natural_frequency_rad_s' in settings:
        if settings.get('monotonic_direction_reset_enabled', True):
            return ('Only slider_rate_limit, ref_accel_limit and the declared ref_wn natural-frequency parameter on both existing coordinator instances; '
                    'all other parameters, state and algorithms preserved for both MPC variants')
        return ('Same slider_rate_limit, ref_accel_limit and declared ref_wn natural-frequency parameter on both existing coordinator instances plus deletion of the two raw-direction monotonicity branches that hold angle_next and reset rate_next; '
                'the damped equation, rate clipping, slider method, measurement blending, state, remaining parameters and existing preview path are preserved for both MPC variants')
    if settings.get('monotonic_direction_reset_enabled', True):
        return 'Only slider_rate_limit and ref_accel_limit on both existing coordinator instances; all other parameters, state and algorithms preserved for both MPC variants'
    return ('Same slider_rate_limit and ref_accel_limit on both existing coordinator instances plus deletion of the two raw-direction monotonicity branches that hold angle_next and reset rate_next; '
            'the damped update, rate clipping, slider method, measurement blending, state, other parameters and existing preview path are preserved for both MPC variants')


def _axis_method_without_direction_reset(cls):
    """Delete exactly the final if/elif; retain every other original AST node."""
    original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_update_axis_state')
    expected = ast.parse('''if raw_target > prev_raw_target + 1e-9 and angle_next < angle_prev:
    angle_next = angle_prev
    rate_next = max(0.0, rate_next)
elif raw_target < prev_raw_target - 1e-9 and angle_next > angle_prev:
    angle_next = angle_prev
    rate_next = min(0.0, rate_next)
''').body[0]
    if (not isinstance(original.body[-1], ast.Return)
            or ast.dump(original.body[-2], include_attributes=False) != ast.dump(expected, include_attributes=False)):
        raise ValueError('The immutable original method no longer ends with exactly the two declared raw-direction branches')
    modified = copy.deepcopy(original)
    del modified.body[-2]
    return original, modified, expected


def source_contract(control_source, helper_source=None, monotonic_direction_reset_enabled=True, reference_natural_frequency_rad_s=None):
    if type(monotonic_direction_reset_enabled) is not bool:
        raise ValueError('Direction-reset source contract requires a strict boolean')
    control_source = Path(control_source)
    if digest(control_source) != ORIGINAL_CONTROL_SHA256:
        raise ValueError('Reference budgets require the unchanged original production control source')
    tree = ast.parse(control_source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AxisManeuverCoordinator')
    methods = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in METHODS}
    if set(methods) != set(METHODS):
        raise ValueError('Missing original coordinator algorithm')
    result = {'original_control_source_sha256': digest(control_source),
            'original_coordinator_method_AST_sha256': methods,
            'helper_sha256': digest(helper_source or Path(__file__)),
            'configuration_field_mapping': dict(FIELD_MAP),
            'production_coordinator_algorithms_modified': False}
    if reference_natural_frequency_rad_s is not None:
        frequency = _validated_reference_frequency(reference_natural_frequency_rad_s)
        result.update(configuration_field_mapping={**FIELD_MAP, **OPTIONAL_FIELD_MAP},
            declared_reference_natural_frequency_rad_s=frequency,
            reference_natural_frequency_change_is_configuration_only=True)
    if not monotonic_direction_reset_enabled:
        original, modified, branch = _axis_method_without_direction_reset(cls)
        result.update(production_coordinator_algorithms_modified=True,
            instance_overridden_method='_update_axis_state',
            removed_raw_direction_monotonicity_branch_count=2,
            removed_branch_AST_sha256=hashlib.sha256(ast.dump(branch, include_attributes=False).encode()).hexdigest(),
            modified_axis_method_AST_sha256=hashlib.sha256(ast.dump(modified, include_attributes=False).encode()).hexdigest(),
            all_original_AST_nodes_except_declared_if_elif_preserved=True,
            original_axis_method_AST_sha256=hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
            original_slider_and_reference_sequence_algorithms_preserved=True)
    return result


def _compile_instance_axis_method(control_source, original_method):
    tree = ast.parse(Path(control_source).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AxisManeuverCoordinator')
    _, modified, _ = _axis_method_without_direction_reset(cls)
    namespace = dict(original_method.__globals__)
    module = ast.fix_missing_locations(ast.Module(body=[modified], type_ignores=[]))
    exec(compile(module, str(Path(__file__).resolve()), 'exec'), namespace)
    return namespace['_update_axis_state']


def install_shared_reference(node, selected, settings, control_source):
    values = validate_shared_reference(settings)
    if node is None:
        raise ValueError('Shared reference budgets require the existing actual node')
    if values['slider_rate_limit_m_s'] > float(selected.P.slider_vel_max):
        raise ValueError('Reference rail speed budget exceeds the unchanged physical model speed')
    reset_enabled = values.get('monotonic_direction_reset_enabled', True)
    mapping = configuration_field_mapping(values)
    contract = source_contract(control_source, monotonic_direction_reset_enabled=reset_enabled,
                               reference_natural_frequency_rad_s=values.get('reference_natural_frequency_rad_s'))
    override = None
    if not reset_enabled:
        override = _compile_instance_axis_method(control_source, node.pitch_axis_coordinator._update_axis_state.__func__)
    records = {}
    for name in ('pitch_axis_coordinator', 'roll_axis_coordinator'):
        coordinator = getattr(node, name)
        before = asdict(coordinator.cfg)
        state, state_values = coordinator.state, copy.deepcopy(asdict(coordinator.state))
        methods = {name: getattr(coordinator, name).__func__ for name in METHODS}
        replacements = {field: values[key] for key, field in mapping.items()}
        if any(value > before[field] for field, value in replacements.items()):
            raise ValueError('Reference budget must not expand the existing coordinator capability')
        coordinator.cfg = replace(coordinator.cfg, **replacements)
        if override is not None:
            coordinator._update_axis_state = MethodType(override, coordinator)
        after = asdict(coordinator.cfg)
        changed = [key for key in before if before[key] != after[key]]
        unchanged_state = coordinator.state is state and asdict(coordinator.state) == state_values
        unchanged_methods = all(getattr(coordinator, key).__func__ is value for key, value in methods.items())
        other_methods = all(getattr(coordinator, key).__func__ is methods[key] for key in METHODS if key != '_update_axis_state')
        class_preserved = all(getattr(type(coordinator), key) is methods[key] for key in METHODS)
        correct_override = (coordinator._update_axis_state.__func__ is override
                            and coordinator._update_axis_state.__self__ is coordinator) if override is not None else unchanged_methods
        if (set(changed) - set(mapping.values()) or not unchanged_state or not other_methods
                or not class_preserved or not correct_override or unchanged_methods is not reset_enabled):
            raise ValueError('Reference budget must preserve all other coordinator parameters, state and methods')
        records[name] = {'before': before, 'after': after, 'changed_fields': changed,
            'all_other_configuration_fields_identical': all(before[key] == after[key] for key in before if key not in mapping.values()),
            'existing_state_identity_and_values_preserved': unchanged_state,
            'existing_production_coordinator_methods_preserved': unchanged_methods,
            'configuration_updated_with_dataclasses_replace': True}
        if override is not None:
            records[name].update(instance_axis_method_overridden=True,
                instance_axis_method_bound_to_existing_coordinator=True,
                instance_axis_method_AST_sha256=contract['modified_axis_method_AST_sha256'],
                original_slider_and_reference_sequence_methods_preserved=other_methods,
                original_global_coordinator_class_methods_preserved=class_preserved,
                existing_build_reference_sequence_dispatches_same_instance_axis_method=True)
    result = {'schema': 'mmc-task-shared-reference-budget-provenance-v1',
        'configuration': values, 'configuration_field_mapping': mapping, 'coordinators': records,
        'all_passed': True, 'instance_configuration_only': reset_enabled,
        'production_coordinator_algorithm_or_state_modified': not reset_enabled,
        'original_class_or_production_files_modified': False,
        'source_contract': contract,
        'helper_file': str(Path(__file__).resolve()), 'helper_sha256': digest(Path(__file__)),
        'angle_acceleration_budget_scope': angle_acceleration_budget_scope(values),
        'final_reference_is_guaranteed_physically_feasible': False,
        'MPC_objective_model_inequalities_or_physical_protection_modified_by_budget': False,
        'same_existing_actual_outer_and_inner_preview_path': True,
        'effective_coordinator_reference_identical_to_common': False}
    if override is not None:
        result.update(monotonic_direction_reset_enabled=False,
            shared_axis_shaping_algorithm_modified=True,
            existing_coordinator_state_modified=False,
            instance_method_only_override=True,
            both_axis_instances_share_identical_override_function=True,
            angle_and_rate_direction_reset_branches_removed_together=True,
            all_other_coordinator_algorithm_AST_nodes_preserved=True,
            algorithm_change_scope=reference_change_scope(values),
            production_preview_uses_existing_instance_dispatch_not_new_copy_path=True)
    if 'reference_natural_frequency_rad_s' in values:
        result.update(reference_natural_frequency_applied_equally_to_both_axes=True,
            reference_natural_frequency_change_is_configuration_only=True,
            reference_natural_frequency_original_rad_s={key: row['before']['ref_wn'] for key, row in records.items()})
    return result
