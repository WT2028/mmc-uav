#!/usr/bin/env python3
"""Optional common terminal capture radius on the existing production outer.

The only write is terminal_hover_capture_radius. The original stationary
trigger, linear capture blend, damping, integral radius, methods and state stay
unchanged. Missing settings cause no write and no provenance/default addition.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import math
from pathlib import Path


PRODUCTION_SHA256 = '266c2773f4eeb2eb790f3b0ec1dacc00b6c248236c5e76f04e7a7b6b96654c10'
SETTING = 'terminal_capture_radius_m'
ATTRIBUTE = 'terminal_hover_capture_radius'
ORIGINAL_RADIUS_M = 1.0
MAXIMUM_RADIUS_M = 8.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def validate_terminal_capture(settings):
    if not isinstance(settings, dict) or set(settings) - {SETTING}:
        raise ValueError('Terminal capture admits only its explicit radius key')
    if SETTING not in settings:
        return {}
    value = settings[SETTING]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not ORIGINAL_RADIUS_M <= value <= MAXIMUM_RADIUS_M):
        raise ValueError('Common terminal capture radius must be finite in [1,8] m')
    return {SETTING: float(value)}


def source_contract(source_path, helper_path=None):
    source_path = Path(source_path)
    helper_path = Path(__file__) if helper_path is None else Path(helper_path)
    if sha(source_path) != PRODUCTION_SHA256:
        raise ValueError('Terminal capture requires the unchanged original common production source')
    tree = ast.parse(source_path.read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OuterPosController')
    methods = {n.name: n for n in outer.body if isinstance(n, ast.FunctionDef)}
    original_values = {}
    for name in (ATTRIBUTE, 'terminal_hover_velocity_damping', 'terminal_hover_integral_decay_radius'):
        assignments = [n for n in ast.walk(methods['__init__']) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and ast.unparse(t) == 'self.' + name for t in n.targets)]
        if len(assignments) != 1:
            raise ValueError('Original terminal setting must have one constructor assignment')
        original_values[name] = ast.literal_eval(assignments[0].value)
    if original_values != {ATTRIBUTE: 1.0, 'terminal_hover_velocity_damping': 1.6,
                          'terminal_hover_integral_decay_radius': .20}:
        raise ValueError('Original terminal radius/damping/integral radius changed')
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MMCUAVROS2Controller')
    node_init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    if any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute)
            and ast.unparse(t) in ('self.outer.' + ATTRIBUTE, 'self.outer.terminal_hover_integral_decay_radius')
            for t in n.targets) for n in ast.walk(node_init)):
        raise ValueError('Original node initialization must preserve constructor capture/integral radii')
    helper = ast.parse(helper_path.read_text())
    install = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'install_terminal_capture')
    writes = [t.attr for n in ast.walk(install) if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'outer']
    if writes != [ATTRIBUTE] or any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in ('setattr', 'delattr') for n in ast.walk(install)):
        raise ValueError('Terminal helper may write exactly the existing capture-radius attribute')
    return {'all_passed': True, 'original_production_source_sha256': sha(source_path),
        'original_outer_method_AST_sha256': {name: ast_sha(methods[name]) for name in (
            '__init__', '_is_stationary_horizontal_target', '_is_near_stationary_horizontal_hold',
            '_terminal_hover_capture_blend', '_apply_terminal_hover_damping', 'compute')},
        'original_node_initialization_AST_sha256': ast_sha(node_init),
        'original_node_preserves_constructor_capture_and_integral_radii': True,
        'terminal_radius_application_AST_sha256': ast_sha(install),
        'helper_file': str(helper_path.resolve()), 'helper_sha256': sha(helper_path),
        'configuration_attribute_mapping': {SETTING: [ATTRIBUTE]},
        'original_terminal_settings': original_values, 'only_outer_attribute_writes': writes,
        'terminal_radius_change_is_configuration_only': True,
        'original_outer_methods_classes_and_source_files_modified': False,
        'scope': 'Existing extra velocity damping only when horizontal target is stationary and not near hold; independent 0.20 m integral washout radius and dynamic tracking unchanged'}


def scalars(attributes):
    return {name: value for name, value in attributes.items()
            if value is None or isinstance(value, (bool, int, float, str))}


def install_terminal_capture(outer, settings):
    declared = validate_terminal_capture(settings)
    if not declared:
        return None
    contract = source_contract(inspect.getsourcefile(type(outer)))
    before = dict(vars(outer))
    if before.get(ATTRIBUTE) != ORIGINAL_RADIUS_M:
        raise ValueError('Install terminal capture on the unchanged original radius-1 outer instance')
    methods = {name: value for name, value in vars(type(outer)).items() if callable(value)}
    outer.terminal_hover_capture_radius = declared[SETTING]
    after = dict(vars(outer))
    changed = sorted(name for name in before if before[name] != after[name])
    preserved = all(after[name] is value for name, value in before.items() if name != ATTRIBUTE)
    class_preserved = all(vars(type(outer)).get(name) is value for name, value in methods.items())
    clone = copy.deepcopy(outer)
    method_copy_ok = all(getattr(clone, name).__self__ is clone
        and getattr(clone, name).__func__ is value.__func__ for name, value in after.items()
        if inspect.ismethod(value) and value.__self__ is outer)
    copy_ok = clone is not outer and method_copy_ok and all(getattr(clone, name) == value for name, value in scalars(after).items())
    if set(before) != set(after) or not preserved or not class_preserved or not copy_ok:
        raise ValueError('Terminal capture changed undeclared attributes/state/class or clone settings')
    return {'schema': 'mmc-task-shared-terminal-capture-provenance-v1',
        'all_passed': True, 'configuration': declared, 'source_contract': contract,
        'before': scalars(before), 'after': scalars(after),
        'changed_existing_outer_attributes': changed,
        'existing_outer_attribute_names': sorted(before),
        'non_scalar_existing_outer_attribute_names': sorted(set(before) - set(scalars(before))),
        'all_other_existing_outer_attributes_and_state_preserved_by_identity': preserved,
        'all_original_outer_class_methods_preserved': class_preserved,
        'no_outer_attribute_added_or_removed': set(before) == set(after),
        'independent_deepcopy_preserves_all_scalar_outer_settings_and_state': copy_ok,
        'existing_instance_method_bindings_preserved_on_independent_deepcopy': method_copy_ok,
        'production_preview_path_or_copy_mechanism_changed': False,
        'original_outer_methods_or_state_algorithms_modified': False,
        'physical_or_authority_limits_modified': False}


def verify_terminal_capture_provenance(proof, settings, contract):
    declared = validate_terminal_capture(settings)
    if not declared:
        return proof is None
    if not isinstance(proof, dict):
        return False
    before, after = proof.get('before', {}), proof.get('after', {})
    changed = sorted(name for name in before if before[name] != after.get(name))
    expected = [ATTRIBUTE] if declared[SETTING] != ORIGINAL_RADIUS_M else []
    return (proof.get('schema') == 'mmc-task-shared-terminal-capture-provenance-v1'
        and proof.get('all_passed') is True and proof.get('configuration') == declared
        and set(before) == set(after)
        and before.get(ATTRIBUTE) == ORIGINAL_RADIUS_M and after.get(ATTRIBUTE) == declared[SETTING]
        and before.get('terminal_hover_velocity_damping') == after.get('terminal_hover_velocity_damping') == 1.6
        and before.get('terminal_hover_integral_decay_radius') == after.get('terminal_hover_integral_decay_radius') == .20
        and changed == expected == proof.get('changed_existing_outer_attributes')
        and set(proof.get('existing_outer_attribute_names', []))
            == set(before) | set(proof.get('non_scalar_existing_outer_attribute_names', []))
        and all(proof.get(name) is True for name in ('all_other_existing_outer_attributes_and_state_preserved_by_identity',
            'all_original_outer_class_methods_preserved', 'no_outer_attribute_added_or_removed',
            'independent_deepcopy_preserves_all_scalar_outer_settings_and_state',
            'existing_instance_method_bindings_preserved_on_independent_deepcopy'))
        and all(proof.get(name) is False for name in ('production_preview_path_or_copy_mechanism_changed',
            'original_outer_methods_or_state_algorithms_modified', 'physical_or_authority_limits_modified'))
        and all(proof.get('source_contract', {}).get(name) == value for name, value in contract.items() if name != 'helper_file'))
