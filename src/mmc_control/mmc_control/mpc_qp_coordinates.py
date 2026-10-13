"""Exact absolute-input coordinates for the existing condensed MPC QP.

The upstream model, affine response, full objective (including regularization),
inequality rows and physical acceptance budget stay in their original units.
Only the solver's decision coordinates change. Returned decisions and objective
values are converted back before upstream observation or acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse


ACTIVE_SOLVERS = []


def _pattern(matrix):
    return (matrix.shape, matrix.indptr.tobytes(), matrix.indices.tobytes())


def _finite_vector(value, size, name, *, allow_infinity=False):
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have {size} elements")
    valid = not np.isnan(vector).any() if allow_infinity else np.isfinite(vector).all()
    if not valid:
        raise ValueError(f"{name} contains invalid numerical values")
    return vector.copy()


class AbsoluteInputOSQP:
    """OSQP-compatible wrapper using U with Delta_U = D U - b.

    ``previous_input_getter`` must supply the previous *actually published*
    input triplet, ordered [tau_z, moving-mass x, moving-mass y]. The wrapper
    never reads simulator truth, disturbance estimates, or an unexecuted request.
    ``factory_original`` must be captured before installing any OSQP factory
    override, to avoid recursive construction.
    """

    def __init__(self, factory_original, previous_input_getter, *,
                 control_time_getter=None, constraint_tolerance=1e-4):
        if not callable(factory_original) or not callable(previous_input_getter):
            raise ValueError("Original solver factory and published-input getter must be callable")
        self._native = factory_original()
        self.previous_input_getter = previous_input_getter
        self.control_time_getter = control_time_getter
        self.constraint_tolerance = float(constraint_tolerance)
        if not math.isfinite(self.constraint_tolerance) or self.constraint_tolerance <= 0.0:
            raise ValueError("Physical acceptance tolerance must be finite and positive")
        self.records = []
        self._setup_generation = 0
        self._ready = False
        self._last_original_decision = None
        self._last_dual = None
        self._last_warm_original = None
        self._last_warm_absolute = None
        self.last_conversion_dt = 0.0
        self.last_native_solve_dt = 0.0
        self.last_original_constraint_violation = math.inf
        self.last_native_absolute_input = None
        self.last_original_increment = None
        ACTIVE_SOLVERS.append(self)

    def __getattr__(self, name):
        return getattr(self._native, name)

    def _published_input(self):
        return _finite_vector(self.previous_input_getter(), 3, "Previous published input")

    def _encode(self, original):
        increments = np.asarray(original, dtype=float).reshape(self.Np, 3).copy()
        increments[0] += self._previous_input
        return np.cumsum(increments, axis=0).ravel()

    def _decode(self, absolute):
        absolute = np.asarray(absolute, dtype=float).reshape(self.Np, 3)
        with np.errstate(invalid="ignore", over="ignore"):
            increments = absolute.copy()
            increments[1:] -= absolute[:-1]
            increments[0] -= self._previous_input
        return increments.ravel()

    def _transform(self):
        start = time.perf_counter()
        self._previous_input = self._published_input()
        upper = self._original_P.toarray()
        self._full_original_P = upper + upper.T
        self._full_original_P[self._diagonal] = upper[self._diagonal]
        original = self._full_original_P
        transformed = original.copy()
        transformed[:, :-3] -= original[:, 3:]
        transformed[:-3, :] -= original[3:, :]
        transformed[:-3, :-3] += original[3:, 3:]
        self._absolute_P.data[:] = transformed[self._p_rows, self._p_columns]
        offset_gradient = self._original_q - original[:, :3] @ self._previous_input
        absolute_gradient = offset_gradient.copy()
        absolute_gradient[:-3] -= offset_gradient[3:]
        self._absolute_q = absolute_gradient
        dense_constraints = self._original_A.toarray()
        absolute_constraints = dense_constraints.copy()
        absolute_constraints[:, :-3] -= dense_constraints[:, 3:]
        self._absolute_A.data[:] = absolute_constraints[self._a_rows, self._a_columns]
        bound_offset = dense_constraints[:, :3] @ self._previous_input
        self._absolute_l = self._original_l + bound_offset
        self._absolute_u = self._original_u + bound_offset
        self._objective_constant = float(
            0.5 * self._previous_input @ original[:3, :3] @ self._previous_input
            - self._original_q[:3] @ self._previous_input)
        if _pattern(self._absolute_P) != self._absolute_p_pattern:
            raise ValueError("Absolute-input Hessian CSC structure changed")
        if _pattern(self._absolute_A) != self._absolute_a_pattern:
            raise ValueError("Absolute-input constraint CSC structure changed")
        self.last_conversion_dt = time.perf_counter() - start

    def setup(self, P, q, A, l, u, **settings):
        self._original_P = sparse.triu(sparse.csc_matrix(P), format="csc")
        self._original_P.sort_indices()
        self.n = self._original_P.shape[0]
        if self.n <= 0 or self.n % 3 or self._original_P.shape != (self.n, self.n):
            raise ValueError("MPC decision dimension must be a positive multiple of three")
        self.Np = self.n // 3
        self._original_A = sparse.csc_matrix(A).copy()
        self._original_A.sort_indices()
        if self._original_A.shape[1] != self.n:
            raise ValueError("Constraint matrix width must match the decision dimension")
        self.m = self._original_A.shape[0]
        self._original_q = _finite_vector(q, self.n, "q")
        self._original_l = _finite_vector(l, self.m, "l", allow_infinity=True)
        self._original_u = _finite_vector(u, self.m, "u", allow_infinity=True)
        if not np.isfinite(self._original_P.data).all() or not np.isfinite(self._original_A.data).all():
            raise ValueError("QP matrix coefficients must be finite")
        self._original_p_pattern = _pattern(self._original_P)
        self._original_a_pattern = _pattern(self._original_A)
        self._diagonal = np.diag_indices(self.n)
        self._p_rows = np.concatenate([np.arange(column + 1) for column in range(self.n)])
        self._p_columns = np.concatenate([np.full(column + 1, column) for column in range(self.n)])
        self._absolute_P = sparse.csc_matrix(
            (np.zeros(self._p_rows.size), (self._p_rows, self._p_columns)),
            shape=(self.n, self.n))
        self._absolute_P.sort_indices()
        # Use the fixed union of possible coefficients, including zeros.
        # A*D often has an unsorted CSC result; sort before native setup AND
        # before retaining the index map used by subsequent Ax updates.
        original_structure = self._original_A.copy()
        original_structure.data.fill(1.0)
        difference_structure = (sparse.eye(3, format="csc") if self.n == 3 else sparse.diags(
            (np.ones(self.n), np.ones(self.n - 3)), (0, -3),
            shape=(self.n, self.n), format="csc"))
        self._absolute_A = (original_structure @ difference_structure).tocsc()
        self._absolute_A.sort_indices()
        self._a_rows = self._absolute_A.indices.copy()
        self._a_columns = np.repeat(np.arange(self.n), np.diff(self._absolute_A.indptr))
        self._absolute_p_pattern = _pattern(self._absolute_P)
        self._absolute_a_pattern = _pattern(self._absolute_A)
        self._setup_generation += 1
        self._last_original_decision = self._last_dual = None
        self._last_warm_original = self._last_warm_absolute = None
        self.setup_settings = dict(settings)
        self._transform()
        result = self._native.setup(P=self._absolute_P, q=self._absolute_q,
                                    A=self._absolute_A, l=self._absolute_l,
                                    u=self._absolute_u, **settings)
        self._ready = True
        if self.setup_settings.get("warm_starting", self.setup_settings.get("warm_start", True)):
            self.warm_start(x=np.zeros(self.n), y=np.zeros(self.m))
        return result

    def update(self, **kwargs):
        if not self._ready:
            raise RuntimeError("Solver setup must precede update")
        allowed = {"Px", "Px_idx", "Ax", "Ax_idx", "q", "l", "u"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported QP updates: {sorted(unknown)}")
        for name, matrix in (("Px", self._original_P), ("Ax", self._original_A)):
            if name in kwargs:
                index = kwargs.get(name + "_idx")
                if index is None:
                    matrix.data[:] = _finite_vector(kwargs[name], matrix.data.size, name)
                else:
                    index = np.asarray(index, dtype=int).reshape(-1)
                    if np.any(index < 0) or np.any(index >= matrix.data.size):
                        raise ValueError(f"Invalid {name} indices")
                    matrix.data[index] = _finite_vector(kwargs[name], index.size, name)
            elif name + "_idx" in kwargs:
                raise ValueError(f"{name}_idx requires {name} values")
        for name, size in (("q", self.n), ("l", self.m), ("u", self.m)):
            if name in kwargs:
                setattr(self, "_original_" + name,
                        _finite_vector(kwargs[name], size, name, allow_infinity=name != "q"))
        if _pattern(self._original_P) != self._original_p_pattern:
            raise ValueError("Original Hessian CSC structure changed")
        if _pattern(self._original_A) != self._original_a_pattern:
            raise ValueError("Original constraint CSC structure changed")
        self._transform()
        result = self._native.update(Px=self._absolute_P.data, q=self._absolute_q,
                                     Ax=self._absolute_A.data, l=self._absolute_l,
                                     u=self._absolute_u)
        # Retain the original controller's unshifted Delta_U warm-start law,
        # encoded using the current actually published input memory.
        if (self._last_original_decision is not None
                and self.setup_settings.get("warm_starting", self.setup_settings.get("warm_start", True))):
            self.warm_start(x=self._last_original_decision, y=self._last_dual)
        return result

    def warm_start(self, x=None, y=None):
        if not self._ready:
            raise RuntimeError("Solver setup must precede warm start")
        options = {}
        conversion_start = time.perf_counter()
        if x is not None:
            original = _finite_vector(x, self.n, "Original warm-start increments")
            absolute = self._encode(original)
            self._last_warm_original = original.copy()
            self._last_warm_absolute = absolute.copy()
            options["x"] = absolute
        if y is not None:
            options["y"] = _finite_vector(y, self.m, "Original warm-start duals")
        self.last_conversion_dt += time.perf_counter() - conversion_start
        return self._native.warm_start(**options)

    def update_settings(self, **settings):
        self.setup_settings.update(settings)
        return self._native.update_settings(**settings)

    def solve(self, *args, **kwargs):
        if not self._ready:
            raise RuntimeError("Solver setup must precede solve")
        start = time.perf_counter()
        try:
            native_result = self._native.solve(*args, **kwargs)
        except BaseException as error:
            self.last_native_solve_dt = time.perf_counter() - start
            self._save_record(None, None, -1, math.nan, math.nan, math.inf,
                              False, repr(error), 0.0)
            raise
        self.last_native_solve_dt = time.perf_counter() - start
        conversion_start = time.perf_counter()
        native_absolute = (None if native_result.x is None else
                           np.asarray(native_result.x, dtype=float).copy())
        original = None if native_absolute is None else self._decode(native_absolute)
        finite = original is not None and np.isfinite(original).all()
        violation = math.inf
        objective = math.nan
        if finite:
            constrained = self._original_A @ original
            if self.m:
                violation = float(max(0.0, np.max(self._original_l - constrained),
                                      np.max(constrained - self._original_u)))
            else:
                violation = 0.0
            with np.errstate(invalid="ignore", over="ignore"):
                objective = float(0.5 * original @ self._full_original_P @ original
                                  + self._original_q @ original)
        status = int(native_result.info.status_val)
        accepted = bool(status in (1, 2) and finite and violation <= self.constraint_tolerance)
        self.last_original_constraint_violation = violation
        self.last_native_absolute_input = native_absolute
        self.last_original_increment = original
        if accepted:
            self._last_original_decision = original.copy()
            self._last_dual = np.asarray(native_result.y, dtype=float).copy()
        else:
            self._last_original_decision = self._last_dual = None
        result = copy.copy(native_result)
        result.info = copy.copy(native_result.info)
        result.x = original
        result.info.obj_val = objective
        result.info.native_obj_val_absolute_U = float(native_result.info.obj_val)
        # A dual-infeasibility certificate is a direction, so its affine
        # offset is zero. Inequality-row primal certificates and y stay fixed.
        certificate = getattr(native_result, "dual_inf_cert", None)
        if certificate is not None:
            certificate = np.asarray(certificate, dtype=float).reshape(self.Np, 3)
            converted = certificate.copy()
            with np.errstate(invalid="ignore"):
                converted[1:] -= certificate[:-1]
            result.dual_inf_cert = converted.ravel()
        conversion_dt = time.perf_counter() - conversion_start
        self.last_conversion_dt += conversion_dt
        self._save_record(native_absolute, original, status,
                          float(native_result.info.obj_val), objective, violation,
                          accepted, "", conversion_dt,
                          int(native_result.info.iter))
        return result

    def _save_record(self, native, original, status, native_objective, objective,
                     violation, accepted, error, decode_dt, iterations=-1):
        now = math.nan if self.control_time_getter is None else float(self.control_time_getter())
        empty = lambda: np.full(self.n, np.nan)
        self.records.append({
            "solve_index": len(self.records), "setup_generation": self._setup_generation,
            "n": self.n, "m": self.m, "controller_time_s": now,
            "previous_published_input": self._previous_input.copy(),
            "native_absolute_input": empty() if native is None else native.copy(),
            "original_increment_solution": empty() if original is None else original.copy(),
            "warm_start_original_increment": (empty() if self._last_warm_original is None
                                               else self._last_warm_original.copy()),
            "warm_start_native_absolute": (empty() if self._last_warm_absolute is None
                                             else self._last_warm_absolute.copy()),
            "solver_status": status, "solver_iterations": iterations,
            "native_objective": native_objective, "original_objective": objective,
            "substitution_constant": self._objective_constant,
            "original_constraint_violation": violation,
            "physical_solution_accepted": accepted,
            "coordinate_conversion_wall_s": self.last_conversion_dt,
            "decode_wall_s": decode_dt, "native_solve_wall_s": self.last_native_solve_dt,
            "error": error,
        })

    def dump_records(self, out, *, prefix="qp_internal_coordinates"):
        """Write all native and original decisions, including rejected ones."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        groups = {}
        for record in self.records:
            groups.setdefault((record["setup_generation"], record["n"], record["m"]), []).append(record)
        archives = []
        for position, (signature, records) in enumerate(groups.items()):
            suffix = "" if len(groups) == 1 else f"_g{position:02d}"
            path = out / (prefix + suffix + ".npz")
            temporary = path.with_name(path.name + ".part")
            if path.exists() or temporary.exists():
                raise FileExistsError("Coordinate records must not overwrite existing evidence")
            arrays = {key: np.asarray([record[key] for record in records])
                      for key in records[0] if key not in ("n", "m", "error")}
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archives.append({"file": path.name, "sha256": digest,
                             "setup_generation": signature[0], "Np": signature[1] // 3,
                             "constraint_rows": signature[2], "records": len(records),
                             "solve_indices": [record["solve_index"] for record in records]})
        metadata = {
            "schema": "mmc-absolute-input-qp-internal-coordinates-v1",
            "coordinate_relation": "Delta_U = D U - [previous actually published input,0,...]",
            "objective_relation": "Original objective = native absolute-U objective + substitution constant; returned obj_val is independently evaluated in original increments",
            "full_original_regularization_preserved": True,
            "dual_rows_unmodified": True,
            "returned_decision_order": "Original time-major Delta_U triplets",
            "native_decision_order": "Absolute predicted input U triplets",
            "inequality_rows_and_bounds": "Exact substitution of the original optimization problem; original physical-row residuals evaluated before acceptance",
            "constraint_acceptance_tolerance": self.constraint_tolerance,
            "solver_settings_forwarded_unchanged": self.setup_settings,
            "warm_start": "Previous original increments encoded with current actual published input; explicit warm_start accepts original increments",
            "failed_candidates_retained": True,
            "solve_count": len(self.records),
            "physical_accepted_count": sum(record["physical_solution_accepted"] for record in self.records),
            "errors": [{"solve_index": record["solve_index"], "error": record["error"]}
                       for record in self.records if record["error"]],
            "archives": archives,
        }
        metadata_path = out / (prefix + "_metadata.json")
        if metadata_path.exists():
            raise FileExistsError("Coordinate metadata must not overwrite existing evidence")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata


def dump_records(out, solvers=None):
    """Dump every active wrapper after the inherited observer has finished."""
    active = [solver for solver in (ACTIVE_SOLVERS if solvers is None else solvers)
              if solver._ready]
    results = []
    for index, solver in enumerate(active):
        prefix = "qp_internal_coordinates" if len(active) == 1 else f"qp_internal_coordinates_s{index:02d}"
        results.append(solver.dump_records(out, prefix=prefix))
    return results
