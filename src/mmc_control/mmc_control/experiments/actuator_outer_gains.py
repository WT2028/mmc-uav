#!/usr/bin/env python3
"""Declared common horizontal proportional gain on the existing outer instance.

Only kp_x and kp_y may change. Original source/classes/methods, all other
attributes and controller state remain untouched. An absent optional key
causes no attribute write and adds no default configuration or provenance.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import math
from pathlib import Path


PRODUCTION_SHA256 = 'c3b7204959b0f77568dc83d54f7cf157eb1df3d294ed88dc281188d4a5c188a0'
SETTING = 'horizontal_position_proportional_gain'
ORIGINAL_GAIN = 2.2
ATTRIBUTE_MAPPING = {SETTING: ['kp_x', 'kp_y']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def validate_horizontal_gain(settings):
    if not isinstance(settings, dict) or set(settings) - {SETTING}:
        raise ValueError('Horizontal outer gain admits only its declared proportional-gain key')
    if SETTING not in settings:
        return {}
    value = settings[SETTING]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0. < value <= ORIGINAL_GAIN):
        raise ValueError('Shared horizontal proportional gain must be finite in (0,2.2] s^-2')
    return {SETTING: float(value)}


def source_contract(source_path, helper_path=None):
    source_path = Path(source_path)
    helper_path = Path(__file__) if helper_path is None else Path(helper_path)
    if sha(source_path) != PRODUCTION_SHA256:
        raise ValueError('Horizontal gain requires the guarded original common outer source')
    tree = ast.parse(source_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OuterPosController')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    compute = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'compute')
    defaults = dict(zip([node.arg for node in init.args.args][-len(init.args.defaults):], init.args.defaults))
    if ast.literal_eval(defaults['kp']) != (ORIGINAL_GAIN, ORIGINAL_GAIN, 4.3):
        raise ValueError('Original outer proportional gains no longer match the declared source')
    gain_defaults = {f'{kind}_{axis}': value for kind in ('kp', 'ki', 'kd')
                     for axis, value in zip(('x', 'y', 'z'), ast.literal_eval(defaults[kind]))}
    original_attributes = sorted({node.attr for assignment in ast.walk(init)
        if isinstance(assignment, ast.Assign) for target in assignment.targets
        for node in ast.walk(target) if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == 'self'} - {'P'})
    helper_tree = ast.parse(helper_path.read_text())
    install = next(node for node in helper_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'install_horizontal_gain')
    writes = [target.attr for node in ast.walk(install) if isinstance(node, (ast.Assign, ast.AnnAssign))
              for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
              if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'outer']
    if sorted(writes) != ['kp_x', 'kp_y'] or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ('setattr', 'delattr') for node in ast.walk(install)):
        raise ValueError('Gain helper may write exactly kp_x/kp_y and no other outer attributes')
    return {'all_passed': True,
        'original_production_source_sha256': sha(source_path),
        'production_outer_init_AST_sha256': ast_sha(init),
        'production_outer_compute_AST_sha256': ast_sha(compute),
        'gain_application_AST_sha256': ast_sha(install),
        'helper_file': str(helper_path.resolve()), 'helper_sha256': sha(helper_path),
        'configuration_attribute_mapping': copy.deepcopy(ATTRIBUTE_MAPPING),
        'original_outer_gain_defaults': gain_defaults,
        'original_outer_scalar_attribute_names': original_attributes,
        'original_horizontal_proportional_gain': ORIGINAL_GAIN,
        'gain_unit': 's^-2', 'only_outer_attribute_writes': sorted(writes),
        'gain_change_is_configuration_only': True,
        'original_outer_methods_classes_and_source_files_modified': False}


def scalar_attributes(attributes):
    return {name: value for name, value in attributes.items()
            if value is None or isinstance(value, (bool, int, float, str))}


def install_horizontal_gain(outer, settings):
    declared = validate_horizontal_gain(settings)
    if not declared:
        return None
    source = inspect.getsourcefile(type(outer))
    if source is None:
        raise ValueError('Actual outer class has no production source file')
    contract = source_contract(source)
    before = dict(vars(outer))
    if before.get('kp_x') != ORIGINAL_GAIN or before.get('kp_y') != ORIGINAL_GAIN:
        raise ValueError('Install declared horizontal gain on the original common 2.2/2.2 instance')
    methods = {name: value for name, value in vars(type(outer)).items() if callable(value)}
    gain = declared[SETTING]
    outer.kp_x = gain
    outer.kp_y = gain
    after = dict(vars(outer))
    changed = sorted(name for name in before if name in ('kp_x', 'kp_y') and before[name] != after[name])
    preserved = all(after[name] is value for name, value in before.items() if name not in ('kp_x', 'kp_y'))
    class_preserved = all(vars(type(outer)).get(name) is value for name, value in methods.items())
    clone = copy.deepcopy(outer)
    copy_ok = (clone is not outer and clone.kp_x == clone.kp_y == gain
        and all(getattr(clone, name) == value for name, value in scalar_attributes(after).items()))
    if (set(before) != set(after) or not preserved or not class_preserved or not copy_ok
            or outer.kp_x != outer.kp_y or outer.kp_x != gain):
        raise ValueError('Horizontal gain changed undeclared outer attributes/state/class or clone settings')
    return {'schema': 'mmc-task-shared-horizontal-outer-gain-provenance-v1',
        'all_passed': True, 'configuration': declared, 'source_contract': contract,
        'before': scalar_attributes(before), 'after': scalar_attributes(after),
        'changed_existing_outer_attributes': changed,
        'existing_outer_attribute_names': sorted(before),
        'non_scalar_existing_outer_attribute_names': sorted(set(before) - set(scalar_attributes(before))),
        'all_other_existing_outer_attributes_and_state_preserved_by_identity': preserved,
        'all_original_outer_class_methods_preserved': class_preserved,
        'no_outer_attribute_added_or_removed': set(before) == set(after),
        'actual_horizontal_gains_applied_equally_xy': outer.kp_x == outer.kp_y == gain,
        'independent_deepcopy_preserves_all_scalar_outer_settings_and_state': copy_ok,
        'production_preview_path_or_copy_mechanism_changed': False,
        'original_outer_methods_or_state_algorithms_modified': False,
        'physical_or_authority_limits_modified': False}


def verify_gain_provenance(proof, settings, contract):
    """Validate captured full before/after evidence without importing ROS."""
    declared = validate_horizontal_gain(settings)
    if not declared:
        return proof is None
    if not isinstance(proof, dict) or not isinstance(contract, dict):
        return False
    before, after = proof.get('before'), proof.get('after')
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after):
        return False
    gain = declared[SETTING]
    expected_changed = sorted(name for name in ('kp_x', 'kp_y') if before.get(name) != gain)
    actual_changed = sorted(name for name in before if before[name] != after[name])
    observed_contract = proof.get('source_contract', {})
    return (proof.get('schema') == 'mmc-task-shared-horizontal-outer-gain-provenance-v1'
        and proof.get('all_passed') is True and proof.get('configuration') == declared
        and before.get('kp_x') == before.get('kp_y') == ORIGINAL_GAIN
        and all(before.get(name) == value for name, value in contract['original_outer_gain_defaults'].items())
        and set(contract['original_outer_scalar_attribute_names']) <= set(before)
        and set(proof.get('existing_outer_attribute_names', []))
            == set(before) | set(proof.get('non_scalar_existing_outer_attribute_names', []))
        and after.get('kp_x') == after.get('kp_y') == gain
        and actual_changed == expected_changed == proof.get('changed_existing_outer_attributes')
        and all(proof.get(name) is True for name in (
            'all_other_existing_outer_attributes_and_state_preserved_by_identity',
            'all_original_outer_class_methods_preserved', 'no_outer_attribute_added_or_removed',
            'actual_horizontal_gains_applied_equally_xy',
            'independent_deepcopy_preserves_all_scalar_outer_settings_and_state'))
        and all(proof.get(name) is False for name in (
            'production_preview_path_or_copy_mechanism_changed',
            'original_outer_methods_or_state_algorithms_modified', 'physical_or_authority_limits_modified'))
        and all(observed_contract.get(name) == value for name, value in contract.items() if name != 'helper_file'))
