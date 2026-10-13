#!/usr/bin/env python3
"""Declared shared instance-only stationary-braking distance regularization.

No production module, class or physical limit is changed. A source/AST contract
permits exactly one denominator-floor change; an explicit optional horizontal
proportional gain updates only kp_x/kp_y with separate configuration provenance.
in the original production method. The bound method stores its setting on
the outer instance so Python deepcopy preserves the same method and rebinds
it to the independent future outer instance, without an original-self closure.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
from types import MethodType


PRODUCTION_SHA256 = '266c2773f4eeb2eb790f3b0ec1dacc00b6c248236c5e76f04e7a7b6b96654c10'
SETTING = 'stationary_braking_distance_floor_m'
INSTANCE_SETTING = '_mmc_stationary_braking_distance_floor_m'
MAXIMUM_FLOOR_M = .03


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_shared_outer(settings):
    from .actuator_outer_gains import SETTING as GAIN_SETTING, validate_horizontal_gain
    from .actuator_terminal_capture import SETTING as CAPTURE_SETTING, validate_terminal_capture
    if not isinstance(settings, dict) or SETTING not in settings or set(settings) - {SETTING, GAIN_SETTING, CAPTURE_SETTING}:
        raise ValueError('shared_outer admits its explicit braking floor and optional common horizontal proportional gain')
    value = settings[SETTING]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1e-6 <= value <= MAXIMUM_FLOOR_M:
        raise ValueError('Shared stationary braking floor must be finite in [1e-6,0.03] m')
    validate_horizontal_gain({key: val for key, val in settings.items() if key == GAIN_SETTING})
    validate_terminal_capture({key: val for key, val in settings.items() if key == CAPTURE_SETTING})
    return float(value)


def clamp(value, minlim, maxlim):
    if value <= minlim:
        return minlim
    elif value >= maxlim:
        return maxlim
    else:
        return value


def _stationary_target_braking_with_floor(
    self,
    pos_err: float,
    vel: float,
    accel_cmd: float,
    accel_limit: float,
) -> float:
    accel_limit = max(float(accel_limit), 1e-6)
    pos_err = float(pos_err)
    vel = float(vel)
    accel_cmd = float(accel_cmd)

    if abs(pos_err) <= 1e-6 or abs(vel) <= 1e-6:
        return accel_cmd
    if pos_err * vel <= 0.0:
        return accel_cmd

    stopping_distance = vel * vel / (2.0 * accel_limit)
    if stopping_distance <= abs(pos_err):
        return accel_cmd

    required_brake = clamp(
        vel * vel / (2.0 * max(abs(pos_err), self._mmc_stationary_braking_distance_floor_m)),
        0.0,
        accel_limit,
    )
    desired_accel = -math.copysign(required_brake, vel)
    if vel > 0.0:
        return min(accel_cmd, desired_accel)
    return max(accel_cmd, desired_accel)


def source_contract(source_path, helper_path=None):
    """Prove that the declared denominator argument is the only AST change."""
    source_path = Path(source_path)
    helper_path = Path(__file__) if helper_path is None else Path(helper_path)
    if sha(source_path) != PRODUCTION_SHA256:
        raise ValueError('Shared braking patch requires the guarded original common production source')
    tree = ast.parse(source_path.read_text())
    outer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OuterPosController')
    method = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == '_apply_stationary_target_braking')
    original = copy.deepcopy(method)
    replacements = 0
    for node in ast.walk(method):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'max'
                and len(node.args) == 2 and ast.unparse(node.args[0]) == 'abs(pos_err)'
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == 1e-6):
            node.args[1] = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=INSTANCE_SETTING, ctx=ast.Load())
            replacements += 1
    method.name = '_stationary_target_braking_with_floor'
    helper_tree = ast.parse(helper_path.read_text())
    helper = next(node for node in helper_tree.body if isinstance(node, ast.FunctionDef) and node.name == method.name)
    original_clamp = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'clamp')
    helper_clamp = next(node for node in helper_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'clamp')
    if replacements != 1 or ast.dump(method, include_attributes=False) != ast.dump(helper, include_attributes=False):
        raise ValueError('Shared braking implementation differs beyond the one approved denominator floor')
    if ast.dump(original_clamp, include_attributes=False) != ast.dump(helper_clamp, include_attributes=False):
        raise ValueError('Shared braking helper clamp differs from original production clamp')
    return {'all_passed': True, 'original_production_source_file': str(source_path.absolute()),
        'original_production_source_sha256': sha(source_path),
        'original_braking_method_AST_sha256': hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        'new_braking_method_AST_sha256': hashlib.sha256(ast.dump(helper, include_attributes=False).encode()).hexdigest(),
        'exact_AST_denominator_argument_replacements': replacements,
        'all_other_braking_AST_and_clamp_identical': True,
        'helper_file': str(helper_path.resolve()), 'helper_sha256': sha(helper_path),
        'denominator_before': '2*max(abs(pos_err),1e-6)',
        'denominator_after': '2*max(abs(pos_err),declared_stationary_braking_distance_floor_m)'}


def install_shared_outer(outer, settings):
    floor = validate_shared_outer(settings)
    source = inspect.getsourcefile(type(outer))
    if source is None:
        raise ValueError('Actual production outer class has no source file')
    proof = source_contract(source)
    original_class_method = type(outer)._apply_stationary_target_braking
    before = dict(vars(outer))
    if INSTANCE_SETTING in before or '_apply_stationary_target_braking' in before:
        raise ValueError('Install the shared braking patch once on a fresh actual outer instance')
    if floor > float(outer.hold_xy_err_deadband):
        raise ValueError('Shared braking floor exceeds the original actual hold-position deadband')
    setattr(outer, INSTANCE_SETTING, floor)
    outer._apply_stationary_target_braking = MethodType(_stationary_target_braking_with_floor, outer)
    unchanged = all(vars(outer)[key] is value for key, value in before.items())
    clone = copy.deepcopy(outer)
    copy_ok = (clone is not outer and clone._apply_stationary_target_braking.__self__ is clone
        and clone._apply_stationary_target_braking.__func__ is _stationary_target_braking_with_floor
        and getattr(clone, INSTANCE_SETTING) == floor)
    if not unchanged or not copy_ok or type(outer)._apply_stationary_target_braking is not original_class_method:
        raise ValueError('Shared outer patch changed state/class attributes or did not survive independent deepcopy')
    result = {'schema': 'mmc-task-shared-outer-braking-provenance-v1',
        'all_passed': True, 'configuration': {SETTING: floor}, 'source_contract': proof,
        'instance_only_override': True, 'class_or_production_files_modified': False,
        'all_existing_outer_attributes_and_state_unchanged_at_install': unchanged,
        'preserved_existing_outer_attribute_names': sorted(before),
        'only_added_outer_instance_attributes': [INSTANCE_SETTING, '_apply_stationary_target_braking'],
        'future_outer_deepcopy_same_method_and_setting_bound_to_independent_clone': copy_ok,
        'global_original_outer_class_method_unchanged': True,
        'outer_law_identical_to_common': False,
        'original_hold_xy_err_deadband_m': float(outer.hold_xy_err_deadband),
        'effective_outer_acceleration_limits_m_s2': [float(outer.ax_cmd_limit), float(outer.ay_cmd_limit)],
        'outer_attitude_limits_rad': [float(outer.roll_limit), float(outer.pitch_limit)],
        'maintained': ['All original braking trigger/early-return tests', 'Stopping-distance trigger formula',
            'Original acceleration clamp and sign selection', 'Far field abs(pos_err)>=floor',
            'All other outer gains, integrals, damping, deadbands and authority',
            'Production physical plant and downstream position/speed/force protection']}
    from .actuator_outer_gains import SETTING as GAIN_SETTING, install_horizontal_gain
    if GAIN_SETTING in settings:
        gain_proof = install_horizontal_gain(outer, {GAIN_SETTING: settings[GAIN_SETTING]})
        result['configuration'][GAIN_SETTING] = gain_proof['configuration'][GAIN_SETTING]
        result['horizontal_gain_provenance'] = gain_proof
        result['all_existing_outer_attributes_and_state_unchanged_at_install'] = not gain_proof['changed_existing_outer_attributes']
        result['all_existing_outer_attributes_except_declared_horizontal_gain_unchanged_at_install'] = gain_proof['all_other_existing_outer_attributes_and_state_preserved_by_identity']
        result['declared_horizontal_gain_applied_equally_xy'] = gain_proof['actual_horizontal_gains_applied_equally_xy']
        result['maintained'][-2] = 'All undeclared outer gains, all integrals, damping, filters, deadbands and authority'
        clone = copy.deepcopy(outer)
        result['future_outer_deepcopy_same_method_and_setting_bound_to_independent_clone'] = (
            clone._apply_stationary_target_braking.__self__ is clone
            and clone._apply_stationary_target_braking.__func__ is _stationary_target_braking_with_floor
            and getattr(clone, INSTANCE_SETTING) == floor
            and clone.kp_x == clone.kp_y == result['configuration'][GAIN_SETTING])
        if not result['future_outer_deepcopy_same_method_and_setting_bound_to_independent_clone']:
            raise ValueError('Declared outer gain and braking floor did not survive independent clone')
    return result
