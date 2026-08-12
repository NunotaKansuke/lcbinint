#!/usr/bin/env python3
"""Compare one warmed finite-source integration epoch at a time.

This is the cache-warm kernel benchmark requested after the block-level
review.  The first of two identical lcbinint source positions constructs the
LensModel and its caustic cache; only the second position's native C++ timing
is recorded.  The lcbinint kernel receives source ``(x, y)`` directly in the
internal lens frame, so source trajectory reconstruction is not part of the
timed epoch.  LensModel construction, caustic-cache construction, and
Python/pybind call overhead are outside the lcbinint timing.  VBM is warmed on
the same source position before its direct call is timed.

Nbin discovery is independent of VBMicrolensing: lcbinint increases Nbin until
three increasing grid values self-converge within the requested tolerance.
VBMicrolensing is evaluated once at that tolerance and is used only for the
final agreement flag and the separate timing measurement.  The historical
``--search-missing`` option is retained for command-line compatibility but no
longer controls the resolution search.

This intentionally measures a pure per-epoch kernel, not the total cost of a
LightCurve call.  ``bench_grid_vs_vbm_dark.py`` remains the historical
single-epoch cold-LensModel benchmark; this file is the corrected companion.

The two kernels are explicit here: lcbinint uses only the preplanned
Cartesian or polar inverse-ray path, while VBM uses ``BinaryMag`` for a
uniform source and ``BinaryMagDark`` for linear limb darkening.  This
benchmark never calls VBM's ``BinaryMag2`` or ``BinaryLightCurve`` automatic
dispatcher, because those APIs can take a point-source shortcut.  For
preplanned lcbinint epochs, the point-source magnification already obtained
while constructing image seeds is reused as the walk hint; the duplicate
point-source solve is not included in the timed kernel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np


def _load_build_lcbinint(build_dir):
    """Load the extension and Python package from one explicit build tree."""

    package = Path(build_dir) / "lcbinint"
    root = package / "__init__.py"
    extensions = sorted(package.glob("_lcbinint*.so"))
    if not root.is_file() or not extensions:
        raise FileNotFoundError(
            f"build package is incomplete: expected {root} and _lcbinint*.so"
        )
    sys.modules.pop("lcbinint._lcbinint", None)
    sys.modules.pop("lcbinint", None)
    root_spec = importlib.util.spec_from_file_location(
        "lcbinint",
        root,
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(root_spec)
    sys.modules["lcbinint"] = module
    extension_spec = importlib.util.spec_from_file_location(
        "lcbinint._lcbinint", extensions[0]
    )
    extension = importlib.util.module_from_spec(extension_spec)
    sys.modules["lcbinint._lcbinint"] = extension
    module._lcbinint = extension
    extension_spec.loader.exec_module(extension)
    root_spec.loader.exec_module(module)
    if not hasattr(module.LightCurve()._native, "_evaluate_preplanned_xy"):
        raise RuntimeError("selected build lacks _evaluate_preplanned_xy")
    return module, extensions[0]


BUILD_DIR = Path(__file__).resolve().parents[3] / "build"
lcbinint, BUILD_EXTENSION = _load_build_lcbinint(BUILD_DIR)

# Import the existing selection, calibration, and VBM helpers only after the
# explicit build package has been installed in sys.modules.
import bench_grid_vs_vbm_dark as base  # noqa: E402
from lcbinint import warmup  # noqa: E402


# Keep the comparison deliberately external and explicit: the two requested
# tolerances are run independently, with no looser 1e-2 condition mixed in.
TARGETS = (1.0e-3, 1.0e-4)
REFERENCE_INDICES = base.REFERENCE_INDICES
MAX_SOURCE_BINS = base.MAX_SOURCE_BINS
BLOCK_EPOCHS = base.BLOCK_EPOCHS
SELF_CONFIRMATION_POINTS = 3


def _evaluate(curve, source_x, params, method, resolutions, source_y=None):
    """Evaluate preplanned epochs from direct internal-frame source (x, y)."""

    source_x = np.asarray(source_x, dtype=float)
    if source_y is None:
        source_y = np.full(source_x.shape, float(params["u0"]), dtype=float)
    return curve._native._evaluate_preplanned_xy(
        source_x,
        np.asarray(source_y, dtype=float),
        params,
        [int(method)] * len(source_x),
        [int(value) for value in resolutions],
    )


def _forced_vbm_one(vbm, row, x, profile_c):
    """Use VBM's direct finite-source API, never its automatic dispatcher."""

    if profile_c:
        # BinaryMagDark performs the limb-darkened annular contour integral.
        return float(
            vbm.BinaryMagDark(
                float(row["s"]),
                float(row["q"]),
                -float(x),
                float(row["y"]),
                float(row["rho"]),
                base.VBM_ABSOLUTE_FLOOR,
            )
        )
    # BinaryMag performs the uniform-source boundary contour integral.
    return float(
        vbm.BinaryMag(
            float(row["s"]),
            float(row["q"]),
            -float(x),
            float(row["y"]),
            float(row["rho"]),
            base.VBM_ABSOLUTE_FLOOR,
        )
    )


def _pure_grid_worker(
    profile_c,
    target,
    times,
    params,
    method,
    resolutions,
    repeats,
    start_index,
    connection,
):
    """Warm a LensModel per target and report the second epoch only."""

    try:
        curve = base._curve(profile_c, target)
        for local_index, x in enumerate(times):
            index = int(start_index) + local_index
            bins = resolutions[local_index]
            if bins is None:
                connection.send({"kind": "unavailable", "index": index})
                continue
            samples = []
            values = []
            converged = []
            for _ in range(repeats):
                result = _evaluate(
                    curve,
                    [x, x],
                    params,
                    method,
                    [int(bins), int(bins)],
                    source_y=[float(params["u0"]), float(params["u0"])],
                )
                actual_methods = result.get("method", [])
                if len(actual_methods) < 2 or any(
                    int(value) != int(method) for value in actual_methods[:2]
                ):
                    raise RuntimeError(
                        "preplanned kernel changed method; expected explicit "
                        f"inverse-ray method {int(method)}"
                    )
                native_seconds = np.asarray(result.get("seconds", []), dtype=float)
                if native_seconds.size < 2 or not np.isfinite(native_seconds[1]):
                    raise RuntimeError("pure-kernel timing did not return epoch 2")
                samples.append(float(native_seconds[1]))
                values.append(float(result["magnification"][1]))
                converged.append(bool(result["converged"][1]))
            connection.send({
                "kind": "point",
                "index": index,
                "seconds": float(np.median(np.asarray(samples))),
                "magnification": float(values[-1]),
                "converged": bool(converged[-1]),
            })
        connection.send({"kind": "done"})
    except BaseException as error:  # noqa: BLE001
        try:
            connection.send({
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
            })
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _time_pure_grid(
    profile_c,
    target,
    times,
    params,
    method,
    resolutions,
    repeats,
    point_timeout,
):
    all_times = np.asarray(times, dtype=float)
    count = int(all_times.size)
    seconds = [None] * count
    magnification = [None] * count
    converged = [False] * count
    statuses = ["unavailable" if value is None else "error"
                for value in resolutions]
    context = mp.get_context("fork")
    start_index = 0

    while start_index < count:
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_pure_grid_worker,
            args=(
                profile_c,
                target,
                all_times[start_index:],
                params,
                method,
                list(resolutions[start_index:]),
                repeats,
                start_index,
                send,
            ),
        )
        process.start()
        send.close()
        point_index = start_index
        finished = False
        while point_index < count:
            deadline = (
                None
                if point_timeout is None or point_timeout <= 0.0
                else time.perf_counter() + float(point_timeout)
            )
            message = None
            while message is None:
                if deadline is None:
                    wait = 0.25
                else:
                    wait = max(0.0, deadline - time.perf_counter())
                    if wait == 0.0:
                        break
                    wait = min(wait, 0.25)
                if receive.poll(wait):
                    message = receive.recv()
                    break
                if not process.is_alive():
                    break
            if message is None:
                statuses[point_index] = "timeout"
                process.terminate()
                process.join(5.0)
                start_index = point_index + 1
                break
            kind = message.get("kind")
            if kind == "point":
                index = int(message["index"])
                seconds[index] = float(message["seconds"])
                magnification[index] = float(message["magnification"])
                converged[index] = bool(message["converged"])
                statuses[index] = "completed"
                point_index = index + 1
                start_index = point_index
                continue
            if kind == "unavailable":
                start_index = int(message["index"]) + 1
                point_index = start_index
                continue
            if kind == "error":
                statuses[point_index] = "error"
                process.terminate()
                process.join(5.0)
                start_index = point_index + 1
                break
            if kind == "done":
                finished = True
                start_index = count
                break
        if process.is_alive():
            process.join(1.0)
        receive.close()
        if finished:
            break

    return {
        "seconds": seconds,
        "magnification": magnification,
        "converged": converged,
        "statuses": statuses,
    }


def _pure_vbm_worker(
    row, times, indices, profile_c, reltol, repeats, connection
):
    """Warm each VBM object once, then time the direct integration call."""

    try:
        vbm = base._new_vbm(profile_c, reltol)
        for local_index, x in enumerate(times):
            index = int(indices[local_index])
            value = _forced_vbm_one(vbm, row, x, profile_c)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                value = _forced_vbm_one(vbm, row, x, profile_c)
                samples.append(time.perf_counter() - started)
            connection.send({
                "kind": "point",
                "index": index,
                "value": float(value),
                "seconds": float(np.median(np.asarray(samples))),
            })
        connection.send({"kind": "done"})
    except BaseException as error:  # noqa: BLE001
        try:
            connection.send({
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
            })
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _time_pure_vbm(row, times, profile_c, target, reference, repeats, point_timeout):
    count = int(reference.size)
    values = [None] * count
    seconds = [None] * count
    statuses = ["unrequested"] * count
    context = mp.get_context("fork")
    remaining = list(range(count))

    while remaining:
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_pure_vbm_worker,
            args=(
                row,
                np.asarray(times)[remaining],
                remaining,
                profile_c,
                target,
                repeats,
                send,
            ),
        )
        process.start()
        send.close()
        position = 0
        finished = False
        original_remaining = list(remaining)
        while position < len(original_remaining):
            point_index = original_remaining[position]
            deadline = (
                None
                if point_timeout is None or point_timeout <= 0.0
                else time.perf_counter() + float(point_timeout)
            )
            message = None
            while message is None:
                if deadline is None:
                    wait = 0.25
                else:
                    wait = max(0.0, deadline - time.perf_counter())
                    if wait == 0.0:
                        break
                    wait = min(wait, 0.25)
                if receive.poll(wait):
                    message = receive.recv()
                    break
                if not process.is_alive():
                    break
            if message is None:
                statuses[point_index] = "timeout"
                process.terminate()
                process.join(5.0)
                remaining = original_remaining[position + 1:]
                break
            kind = message.get("kind")
            if kind == "point":
                index = int(message["index"])
                values[index] = float(message["value"])
                seconds[index] = float(message["seconds"])
                statuses[index] = "completed"
                position += 1
                continue
            if kind == "error":
                statuses[point_index] = "error"
                process.terminate()
                process.join(5.0)
                remaining = original_remaining[position + 1:]
                break
            if kind == "done":
                finished = True
                remaining = []
                break
        # The worker sends ``done`` after its final point.  The loop above
        # normally exits immediately after receiving that final point, so it
        # may never consume the trailing message.  All requested points are
        # nevertheless complete; do not restart the same VBM batch forever.
        if not finished and position >= len(original_remaining):
            finished = True
            remaining = []
        if process.is_alive():
            process.join(1.0)
        receive.close()
        if finished:
            break

    errors = [
        None if values[index] is None else float(
            abs(values[index] - reference[index]) /
            max(abs(reference[index]), 1.0)
        )
        for index in range(count)
    ]
    return {
        "seconds": seconds,
        "values": values,
        "errors": errors,
        "statuses": statuses,
    }


def _vbm_reference_worker(row, times, indices, profile_c, reltol, connection):
    """Evaluate each VBM reference epoch exactly once at ``reltol``."""

    try:
        vbm = base._new_vbm(profile_c, reltol)
        for local_index, x in enumerate(times):
            value = _forced_vbm_one(vbm, row, float(x), profile_c)
            connection.send({
                "kind": "point",
                "index": int(indices[local_index]),
                "value": float(value),
            })
        connection.send({"kind": "done"})
    except BaseException as error:  # noqa: BLE001
        try:
            connection.send({
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
            })
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _vbm_reference_once(row, times, profile_c, reltol, point_timeout):
    """Obtain one independent VBM value per reference epoch.

    This is intentionally separate from the repeated VBM timing pass.  The
    returned values are not a high-precision shared corpus reference; they are
    exactly the requested VBM calculation at the target ``RelTol``.
    """

    count = int(np.asarray(times).size)
    values = [None] * count
    statuses = ["unrequested"] * count
    context = mp.get_context("fork")
    remaining = list(range(count))
    while remaining:
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_vbm_reference_worker,
            args=(
                row,
                np.asarray(times)[remaining],
                remaining,
                profile_c,
                reltol,
                send,
            ),
        )
        process.start()
        send.close()
        position = 0
        original_remaining = list(remaining)
        finished = False
        while position < len(original_remaining):
            point_index = original_remaining[position]
            deadline = (
                None
                if point_timeout is None or point_timeout <= 0.0
                else time.perf_counter() + float(point_timeout)
            )
            message = None
            while message is None:
                wait = 0.25
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - time.perf_counter()))
                    if wait == 0.0:
                        break
                if receive.poll(wait):
                    message = receive.recv()
                    break
                if not process.is_alive():
                    break
            if message is None:
                statuses[point_index] = "timeout"
                process.terminate()
                process.join(5.0)
                remaining = original_remaining[position + 1:]
                break
            kind = message.get("kind")
            if kind == "point":
                index = int(message["index"])
                values[index] = float(message["value"])
                statuses[index] = "completed"
                position += 1
                continue
            if kind == "error":
                statuses[point_index] = "error"
                process.terminate()
                process.join(5.0)
                remaining = original_remaining[position + 1:]
                break
            if kind == "done":
                finished = True
                remaining = []
                break
        if not finished and position >= len(original_remaining):
            finished = True
            remaining = []
        if process.is_alive():
            process.join(1.0)
        receive.close()
        if finished:
            break
    return {"values": values, "statuses": statuses, "reltol": float(reltol)}


def _self_convergence_run(samples, target):
    """Return the first three-grid self-converged run.

    ``samples`` maps an integer Nbin to a measured magnification.  The VBM
    value is deliberately absent: three consecutive increasing grid values
    must lie within the requested relative spread.  Returning the first grid
    in the run implements the requested minimum-Nbin rule.
    """

    ordered = sorted(samples)
    points = int(SELF_CONFIRMATION_POINTS)
    for end in range(points - 1, len(ordered)):
        run = ordered[end - points + 1:end + 1]
        values = np.asarray([samples[bins] for bins in run], dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        scale = max(float(np.max(np.abs(values))), 1.0)
        spread = float(np.max(values) - np.min(values)) / scale
        if spread <= float(target):
            return tuple(run)
    return None


def _required_resolutions_self(
    curve, profile_c, target, times, params, method, predicted, cap,
    point_timeout,
):
    """Find Nbin from lcbinint self-convergence only.

    The point-source resolution hint only determines where the measured ladder
    starts.  It is not an accuracy certificate.  Candidate grids are then
    increased until three increasing Nbin values agree with one another within
    ``target``.  No VBM value or native support certificate is used here.
    """

    indices = np.arange(int(np.asarray(times).size), dtype=int)
    states = {
        int(index): {
            "samples": {},
            "next": warmup._candidate_batch(
                int(predicted[int(index)]), int(cap)
            ),
            "run": None,
            "terminal_status": None,
        }
        for index in indices
    }
    required = {int(index): None for index in indices}

    for _ in range(warmup.GRID_SEARCH_MAX_ROUNDS):
        selected = []
        resolutions = []
        for index in sorted(states):
            for bins in states[index]["next"]:
                selected.append(index)
                resolutions.append(bins)
        if not selected:
            break
        selected = np.asarray(selected, dtype=int)
        measured = _time_pure_grid(
            profile_c,
            target,
            np.asarray(times)[selected],
            params,
            method,
            resolutions,
            repeats=1,
            point_timeout=point_timeout,
        )
        values = [
            np.nan if value is None else float(value)
            for value in measured["magnification"]
        ]
        statuses = measured["statuses"]
        for offset, index_value in enumerate(selected):
            index = int(index_value)
            bins = int(resolutions[offset])
            if statuses[offset] != "completed" or not np.isfinite(values[offset]):
                states[index]["terminal_status"] = statuses[offset]
                states[index]["next"] = ()
                continue
            states[index]["samples"][bins] = float(values[offset])

        for index in sorted(states):
            if (
                required[index] is not None
                or states[index]["terminal_status"] is not None
                or not states[index]["next"]
            ):
                continue
            run = _self_convergence_run(
                states[index]["samples"], float(target)
            )
            if run is not None:
                states[index]["run"] = run
                required[index] = int(run[0])
                states[index]["next"] = ()
            else:
                states[index]["next"] = warmup._next_candidate_batch(
                    states[index]["samples"], int(cap)
                )

    sample_records = {}
    for index, state in states.items():
        ordered = sorted(state["samples"])
        values = [float(state["samples"][bins]) for bins in ordered]
        self_errors = [None]
        for previous, current in zip(values, values[1:]):
            scale = max(abs(current), 1.0)
            self_errors.append(abs(current - previous) / scale)
        sample_records[str(index)] = {
            "nbin": ordered,
            "magnification": values,
            "successive_relative_change": self_errors,
            "confirmation_run": (
                None if state["run"] is None else list(state["run"])
            ),
            "status": "self_converged" if state["run"] is not None else (
                "self_unresolved" if state["terminal_status"] is None
                else f"self_{state['terminal_status']}"
            ),
        }
    return {
        "required": required,
        "samples": sample_records,
        "source": "self_convergence_search",
        "confirmation_points": SELF_CONFIRMATION_POINTS,
        "point_timeout": point_timeout,
    }


def _required_search(curve, row, times_ref, params, reference, target, method,
                     search_missing, profile_c, point_timeout):
    # ``reference`` and ``search_missing`` remain in the signature for
    # compatibility with older diagnostic callers, but neither is used for
    # the Nbin decision.  The only reference-free hint is point-source
    # magnification, followed by the self-convergence ladder above.
    indices = np.arange(int(np.asarray(times_ref).size), dtype=int)
    point = _evaluate(
        curve, times_ref, params, warmup.POINT_SOURCE, [0] * indices.size
    )
    point_values = np.asarray(point["magnification"], dtype=float)
    cap = int(MAX_SOURCE_BINS)
    predicted = {
        int(index): warmup._predicted_resolution(
            curve,
            method,
            point_values[index],
            0.0,
            target,
            cap,
        )
        for index in indices
    }
    search = _required_resolutions_self(
        curve,
        profile_c,
        target,
        times_ref,
        params,
        method,
        predicted,
        cap,
        point_timeout,
    )
    required = search["required"]
    return {
        **search,
        "predicted": [int(predicted[int(index)]) for index in indices],
        "required": [
            None if required[int(index)] is None
            else int(required[int(index)])
            for index in indices
        ],
    }


def _relative_errors(values, reference):
    values = list(values)
    reference = list(reference)
    errors = []
    for value, ref in zip(values, reference):
        if value is None or not np.isfinite(value) or not np.isfinite(ref):
            errors.append(None)
            continue
        errors.append(float(abs(float(value) - float(ref)) /
                        max(abs(float(ref)), 1.0)))
    return errors


def measure(
    row, target, repeats, search_missing, point_timeout, search_point_timeout,
):
    profile_c = float(row["limb_darkening_c"])
    times_full = base._times(row)
    times_ref = times_full[list(REFERENCE_INDICES)]
    params = base._params(row)
    vbm_reference = _vbm_reference_once(
        row, times_ref, profile_c, target, point_timeout
    )
    reference = np.asarray([
        np.nan if value is None else float(value)
        for value in vbm_reference["values"]
    ], dtype=float)
    curve = base._curve(profile_c, target)

    grid = {}
    for name, method in (
        ("cartesian", warmup.CARTESIAN),
        ("polar", warmup.POLAR),
    ):
        search = _required_search(
            curve,
            row,
            times_ref,
            params,
            reference,
            target,
            method,
            search_missing,
            profile_c,
            search_point_timeout,
        )
        required = search["required"]
        timed = _time_pure_grid(
            profile_c,
            target,
            times_ref,
            params,
            method,
            required,
            repeats,
            point_timeout,
        )
        values = timed["magnification"]
        errors = _relative_errors(values, reference)
        grid[name] = {
            **search,
            "kernel_mode": f"forced_inverse_ray_{name}",
            "seconds": timed["seconds"],
            "errors": errors,
            "vbm_reference_errors": errors,
            "converged": timed["converged"],
            "statuses": timed["statuses"],
        }

    vbm_timed = _time_pure_vbm(
        row,
        times_ref,
        profile_c,
        target,
        reference,
        repeats,
        point_timeout,
    )

    chosen_grid = []
    chosen_nbin = []
    chosen_seconds = []
    ratios = []
    ratio_status = []
    chosen_vbm_errors = []
    vbm_mismatch = []
    for index in range(reference.size):
        candidates = []
        for name in ("cartesian", "polar"):
            candidate = grid[name]
            if (
                candidate["required"][index] is not None
                and candidate["seconds"][index] is not None
                and candidate["statuses"][index] == "completed"
            ):
                candidates.append((
                    float(candidate["seconds"][index]),
                    name,
                    int(candidate["required"][index]),
                ))
        if not candidates:
            chosen_grid.append(None)
            chosen_nbin.append(None)
            chosen_seconds.append(None)
            ratios.append(None)
            ratio_status.append("grid_unresolved")
            chosen_vbm_errors.append(None)
            vbm_mismatch.append(None)
            continue
        best = min(candidates, key=lambda value: value[0])
        chosen_seconds.append(best[0])
        chosen_grid.append(best[1])
        chosen_nbin.append(best[2])
        chosen_error = grid[best[1]]["errors"][index]
        chosen_vbm_errors.append(chosen_error)
        vbm_mismatch.append(
            None if chosen_error is None else bool(chosen_error > target)
        )
        if vbm_timed["seconds"][index] is None:
            ratios.append(None)
            ratio_status.append("vbm_unresolved")
        else:
            ratios.append(float(vbm_timed["seconds"][index] / best[0]))
            ratio_status.append("measured")

    return {
        "status": "completed",
        "timing_mode": "pure_kernel_cache_warm_direct_xy",
        "coordinate_mode": "direct_internal_source_xy",
        "point_source_hint_mode": "reuse_preplanned_point_magnification",
        "comparison_mode": "forced_inverse_ray_vs_forced_vbm_contour",
        "lcbinint_kernel_mode": "forced_inverse_ray_cartesian_or_polar",
        "vbm_kernel_mode": (
            "BinaryMagDark_annular_contour" if profile_c
            else "BinaryMag_uniform_contour"
        ),
        "build_extension": str(BUILD_EXTENSION),
        "case_id": int(row["case_id"]),
        "s": float(row["s"]),
        "q": float(row["q"]),
        "rho": float(row["rho"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "d_over_rho": float(row["intended_distance_factor"]),
        "point_magnification": float(row["magnification"]),
        "profile": row["profile"],
        "limb_darkening_c": profile_c,
        "target": float(target),
        "reference_mode": "single_vbm_call_at_target_reltol",
        "reference": reference.tolist(),
        "reference_status": vbm_reference["statuses"],
        "grid": grid,
        "vbm": {
            "selected_seconds": vbm_timed["seconds"],
            "selected_reltol": [float(target)] * int(reference.size),
            "selected_status": vbm_timed["statuses"],
            "timing_values": vbm_timed["values"],
            "timing_reference_errors": vbm_timed["errors"],
            "reference_values": reference.tolist(),
            "reference_status": vbm_reference["statuses"],
        },
        "chosen_grid": chosen_grid,
        "chosen_nbin": chosen_nbin,
        "chosen_seconds": chosen_seconds,
        "chosen_vbm_errors": chosen_vbm_errors,
        "vbm_mismatch": vbm_mismatch,
        "ratios_vbm_over_lcbinint": ratios,
        "ratio_status": ratio_status,
        "vbm_status": vbm_timed["statuses"],
        "vbm_lower_bound_seconds": [None] * int(reference.size),
        "resolution_mode": "lcbinint_self_convergence_three_points",
        "resolution_cap": int(MAX_SOURCE_BINS),
    }


def _timed_measure(
    row, target, repeats, search_missing, point_timeout,
    search_point_timeout, job_timeout,
):
    if job_timeout is None or job_timeout <= 0.0:
        return measure(
            row, target, repeats, search_missing, point_timeout,
            search_point_timeout,
        )
    context = mp.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def worker():
        try:
            send_result = measure(
                row,
                target,
                repeats,
                search_missing,
                point_timeout,
                search_point_timeout,
            )
            send.send(send_result)
        except BaseException as error:  # noqa: BLE001
            try:
                send.send({
                    "status": "error",
                    "case_id": int(row["case_id"]),
                    "profile": row["profile"],
                    "target": float(target),
                    "error": f"{type(error).__name__}: {error}",
                    "ratios_vbm_over_lcbinint": [],
                    "chosen_nbin": [],
                })
            except (BrokenPipeError, EOFError):
                pass
        finally:
            send.close()

    process = context.Process(target=worker)
    process.start()
    send.close()
    process.join(float(job_timeout))
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        receive.close()
        return {
            "status": "timeout",
            "case_id": int(row["case_id"]),
            "profile": row["profile"],
            "target": float(target),
            "d_over_rho": float(row["intended_distance_factor"]),
            "timeout_seconds": float(job_timeout),
            "ratios_vbm_over_lcbinint": [],
            "chosen_nbin": [],
        }
    if receive.poll(1.0):
        result = receive.recv()
        receive.close()
        return result
    receive.close()
    return {
        "status": "error",
        "case_id": int(row["case_id"]),
        "profile": row["profile"],
        "target": float(target),
        "d_over_rho": float(row["intended_distance_factor"]),
        "error": f"worker exited with code {process.exitcode}",
        "ratios_vbm_over_lcbinint": [],
        "chosen_nbin": [],
    }


def main():
    global MAX_SOURCE_BINS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("tests/diagnostics/results/recal2026/speed_discovery"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--case-ids", type=int, nargs="+")
    parser.add_argument("--case-id-min", type=int)
    parser.add_argument("--case-id-max", type=int)
    parser.add_argument("--factors", type=float, nargs="+",
                        default=list(base.DEFAULT_FACTORS))
    parser.add_argument("--profiles", nargs="+", choices=("uniform", "linear"),
                        default=[])
    parser.add_argument("--rho-min", type=float)
    parser.add_argument("--rho-max", type=float)
    parser.add_argument("--q-min", type=float)
    parser.add_argument("--q-max", type=float)
    parser.add_argument("--d-max", type=float)
    parser.add_argument("--magnification-min", type=float)
    parser.add_argument("--magnification-max", type=float)
    parser.add_argument("--route-filter", choices=("all", "grid"), default="all")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--point-timeout", type=float, default=300.0)
    parser.add_argument(
        "--search-point-timeout", type=float, default=60.0,
        help="per-candidate timeout during the self-convergence ladder",
    )
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--search-missing", action="store_true",
        help="deprecated compatibility flag; self-convergence search is always run",
    )
    parser.add_argument(
        "--max-source-bins", type=int, default=MAX_SOURCE_BINS,
        help="maximum source-bin count used by the self-convergence search",
    )
    parser.add_argument("--targets", type=float, nargs="+",
                        default=list(TARGETS))
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    args = parser.parse_args()

    # The build loader runs at module import with the repository build path.
    # Refuse a different path rather than silently mixing Python and C++ builds.
    if args.build_dir.resolve() != BUILD_DIR.resolve():
        raise SystemExit(
            "this run was loaded from the repository build; invoke a fresh "
            "process with the requested --build-dir"
        )
    if args.max_source_bins < 4:
        raise SystemExit("--max-source-bins must be at least 4")
    MAX_SOURCE_BINS = int(args.max_source_bins)
    base.MAX_SOURCE_BINS = MAX_SOURCE_BINS
    rows = base._select_rows(
        base._load_rows(args.input), args.case_count, args.factors, args.seed
    )
    if args.case_id_min is not None:
        rows = [row for row in rows if int(row["case_id"]) >= args.case_id_min]
    if args.case_id_max is not None:
        rows = [row for row in rows if int(row["case_id"]) < args.case_id_max]
    if args.case_ids is not None:
        wanted_case_ids = {int(value) for value in args.case_ids}
        rows = [row for row in rows if int(row["case_id"]) in wanted_case_ids]
    rows = base._filter_rows(
        rows,
        profiles=set(args.profiles) if args.profiles else None,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        q_min=args.q_min,
        q_max=args.q_max,
        d_max=args.d_max,
        magnification_min=args.magnification_min,
        magnification_max=args.magnification_max,
    )
    if not rows:
        raise SystemExit("no rows selected")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows:
        for target in args.targets:
            if args.route_filter == "grid" and not base._pure_grid_route(row, target):
                continue
            selected.append((row, target))
    print(
        f"selected {len(rows)} blocks, {len(selected)} jobs; "
        f"{len(rows) * len(REFERENCE_INDICES)} reference epochs per target",
        flush=True,
    )
    results = []
    started = time.perf_counter()
    for index, (row, target) in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] case={row['case_id']} "
            f"d/rho={row['intended_distance_factor']} "
            f"profile={row['profile']} target={target:g}",
            flush=True,
        )
        result = _timed_measure(
            row, target, args.repeats, args.search_missing,
            args.point_timeout, args.search_point_timeout, args.job_timeout,
        )
        results.append(result)
        if result.get("status") != "completed":
            print(f"  {result.get('status')}", flush=True)

    payload = {
        "input": str(args.input),
        "case_count": args.case_count,
        "case_ids": args.case_ids,
        "case_id_min": args.case_id_min,
        "case_id_max": args.case_id_max,
        "factors": list(args.factors),
        "seed": args.seed,
        "repeats": args.repeats,
        "search_missing": args.search_missing,
        "max_source_bins": MAX_SOURCE_BINS,
        "resolution_mode": "lcbinint_self_convergence_three_points",
        "self_confirmation_points": SELF_CONFIRMATION_POINTS,
        "reference_mode": "single_vbm_call_at_target_reltol",
        "point_timeout": args.point_timeout,
        "search_point_timeout": args.search_point_timeout,
        "job_timeout": args.job_timeout,
        "targets": list(args.targets),
        "profiles": list(args.profiles),
        "route_filter": args.route_filter,
        "timing_mode": "pure_kernel_cache_warm_direct_xy",
        "coordinate_mode": "direct_internal_source_xy",
        "point_source_hint_mode": "reuse_preplanned_point_magnification",
        "comparison_mode": "forced_inverse_ray_vs_forced_vbm_contour",
        "lcbinint_kernel_mode": "forced_inverse_ray_cartesian_or_polar",
        "vbm_kernel_mode": "BinaryMag_or_BinaryMagDark_direct",
        "build_extension": str(BUILD_EXTENSION),
        "filters": {
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "q_min": args.q_min,
            "q_max": args.q_max,
            "d_max": args.d_max,
            "magnification_min": args.magnification_min,
            "magnification_max": args.magnification_max,
        },
        "reference_indices": list(REFERENCE_INDICES),
        "results": results,
        "summary": base.summarise(results, args.targets),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
