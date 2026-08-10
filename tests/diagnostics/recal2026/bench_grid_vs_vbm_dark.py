#!/usr/bin/env python3
"""Fair finite-source grid timing against VBM's explicit integrators.

This is deliberately different from ``sweep_speed.py``.  The latter measures
the production-shaped VBM ``BinaryLightCurve`` entry point, which may take a
point-source or multipole shortcut.  Here lcbinint is evaluated through its
preplanned Cartesian/polar grid and VBM is evaluated through ``BinaryMag`` for
uniform sources or ``BinaryMagDark`` for linear limb darkening.

By default, the already-measured speed corpus supplies the smallest qualifying
fixed Nbin for each grid and target.  This avoids repeating the expensive
search when the question is only the fair integrator timing.  The warm-up
search remains available as an explicit fallback for rows whose old ladder did
not reach the requested target.

The source positions and certified reference values come from the existing
recal2026 speed corpus.  This keeps the reference arbitration already done for
the high-magnification VBM path, while allowing a smaller stratified subset to
be re-timed with the fair integrator-only API.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import lcbinint
from lcbinint import warmup


TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)
# These are the d/rho strata actually present in the existing speed corpus.
# Sampling them independently matters: the corpus deliberately distributes
# the strata across cases, so one case subset cannot contain every factor.
DEFAULT_FACTORS = (0.0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.1, 1.35, 1.7, 2.0, 3.0, 5.0, 10.0, 30.0)
REFERENCE_INDICES = (0, 7, 15, 23)
BLOCK_EPOCHS = 24
BLOCK_SPAN_IN_RADII = 0.4
VBM_ABSOLUTE_FLOOR = 1.0e-12
MAX_SOURCE_BINS = 400


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _pure_grid_route(row, target):
    """Whether the recorded production auto route is exactly finite-source grid."""

    entries = [
        entry for entry in row.get("engines", ())
        if entry.get("engine") == "lcbinint_auto"
        and abs(float(entry.get("knob", -1.0)) - float(target)) < 1.0e-15
    ]
    if not entries:
        return False
    methods = set(entries[0].get("methods", ()))
    return (
        any(name.startswith("inverse_ray_") for name in methods)
        and not methods.intersection({
            "point_source", "hexadecapole", "source_plane_quadrature",
        })
    )


def _load_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("block-*.json")):
        payload = json.loads(path.read_text())
        rows.extend(payload.get("rows", []))
    return rows


def _select_rows(rows, case_count, factors, seed):
    wanted = {round(float(value), 12) for value in factors}
    by_factor_case = {factor: {} for factor in wanted}
    for row in rows:
        factor = row.get("intended_distance_factor")
        if not _finite(factor):
            continue
        factor = round(float(factor), 12)
        if factor not in wanted:
            continue
        references = row.get("references", {})
        if not all(str(index) in references for index in REFERENCE_INDICES):
            continue
        by_factor_case[factor].setdefault(int(row["case_id"]), []).append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for factor in sorted(by_factor_case):
        case_ids = sorted(by_factor_case[factor])
        if case_count < len(case_ids):
            indices = rng.choice(len(case_ids), case_count, replace=False)
            case_ids = [case_ids[int(index)] for index in indices]
        for case_id in case_ids:
            selected.extend(by_factor_case[factor][case_id])
    return selected


def _usable(row, target):
    floor = row.get("reference_floor")
    if not _finite(floor) or float(floor) > 0.1 * target:
        return False
    return all(
        row["references"][str(index)].get("status") == "ok"
        and _finite(row["references"][str(index)].get("value"))
        for index in REFERENCE_INDICES
    )


def _filter_rows(
    rows,
    profiles=None,
    rho_min=None,
    rho_max=None,
    q_min=None,
    q_max=None,
    d_max=None,
    magnification_min=None,
    magnification_max=None,
):
    selected = []
    for row in rows:
        if profiles and row.get("profile") not in profiles:
            continue
        rho = float(row["rho"])
        q = float(row["q"])
        distance = float(row["intended_distance_factor"])
        magnification = float(row["magnification"])
        if rho_min is not None and rho < rho_min:
            continue
        if rho_max is not None and rho >= rho_max:
            continue
        if q_min is not None and q < q_min:
            continue
        if q_max is not None and q >= q_max:
            continue
        if d_max is not None and distance >= d_max:
            continue
        if (
            magnification_min is not None
            and magnification < magnification_min
        ):
            continue
        if (
            magnification_max is not None
            and magnification >= magnification_max
        ):
            continue
        selected.append(row)
    return selected


def _params(row):
    return {
        "t0": 0.0,
        "tE": 1.0,
        "u0": float(row["y"]),
        "alpha": 0.0,
        "s": float(row["s"]),
        "q": float(row["q"]),
        "rho": float(row["rho"]),
    }


def _times(row):
    rho = float(row["rho"])
    return np.linspace(
        float(row["x"]) - 0.5 * BLOCK_SPAN_IN_RADII * rho,
        float(row["x"]) + 0.5 * BLOCK_SPAN_IN_RADII * rho,
        BLOCK_EPOCHS,
    )


def _reference(row):
    return np.asarray(
        [row["references"][str(index)]["value"] for index in REFERENCE_INDICES],
        dtype=float,
    )


def _curve(profile_c, target):
    limb = (
        lcbinint.LimbDarkening.linear(profile_c)
        if profile_c
        else lcbinint.LimbDarkening.none()
    )
    return lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="vbm",
            nbin="auto",
            reltol=target,
            max_source_bins=MAX_SOURCE_BINS,
        ),
        limb_darkening=limb,
    )


def _evaluate(curve, times, params, method, resolutions):
    count = int(np.asarray(times).size)
    return curve._native._evaluate_preplanned(
        np.asarray(times, dtype=float),
        params,
        [int(method)] * count,
        [int(value) for value in resolutions],
    )


def _existing_required(row, target, method):
    engine = "lcbinint_cartesian" if method == warmup.CARTESIAN else "lcbinint_polar"
    candidates = []
    for entry in row.get("engines", []):
        if entry.get("engine") != engine or "seconds_per_epoch" not in entry:
            continue
        error = entry.get("error", {}).get("worst")
        if _finite(error) and float(error) <= float(target):
            candidates.append(int(entry["knob"]))
    return min(candidates) if candidates else None


def _existing_ladder(row, method):
    engine = "lcbinint_cartesian" if method == warmup.CARTESIAN else "lcbinint_polar"
    return sorted({
        int(entry["knob"])
        for entry in row.get("engines", [])
        if entry.get("engine") == engine
        and "seconds_per_epoch" in entry
    })


def _required_grid(
    curve, row, times, params, reference, target, method, search_missing,
    use_existing=True,
):
    """Find per-epoch Nbin using the production warm-up search."""

    count = int(reference.size)
    ladder = _existing_ladder(row, method)
    if use_existing and ladder:
        existing = _existing_required(row, target, method)
        return {
            "source": "speed_discovery_ladder_revalidated",
            "predicted": [existing or ladder[0]] * count,
            "required": [None] * count,
            "candidate_ladder": ladder,
        }
    if not search_missing:
        return {
            "source": "unreached_in_speed_discovery",
            "predicted": [None] * count,
            "required": [None] * count,
        }
    indices = np.arange(count, dtype=int)
    point = _evaluate(curve, times, params, warmup.POINT_SOURCE, [0] * count)
    point_values = np.asarray(point["magnification"], dtype=float)
    predicted = {
        int(index): warmup._predicted_resolution(
            curve,
            method,
            point_values[index],
            0.0,
            target,
            MAX_SOURCE_BINS,
        )
        for index in indices
    }
    budget = target * np.maximum(np.abs(reference), 1.0)
    required = warmup._required_resolutions(
        curve,
        times,
        params,
        indices,
        method,
        reference,
        budget,
        predicted,
        MAX_SOURCE_BINS,
    )
    return {
        "source": "warmup_search",
        "predicted": [int(predicted[int(index)]) for index in indices],
        "required": [
            None if required[int(index)] is None else int(required[int(index)])
            for index in indices
        ],
    }


def _revalidate_ladder(curve, times, params, reference, target, method, search):
    """Use old ladder values as candidates, but certify them in this build."""

    required = [None] * int(reference.size)
    pending = np.arange(reference.size, dtype=int)
    for bins in search.get("candidate_ladder", ()):
        if pending.size == 0:
            break
        measured = _evaluate(
            curve,
            np.asarray(times)[pending],
            params,
            method,
            [int(bins)] * pending.size,
        )
        values = np.asarray(measured["magnification"], dtype=float)
        support = np.asarray(measured["converged"], dtype=bool)
        errors = np.abs(values - reference[pending]) / np.maximum(
            np.abs(reference[pending]), 1.0
        )
        passes = support & np.isfinite(values) & (errors <= target)
        passed = pending[passes]
        for index in passed:
            required[int(index)] = int(bins)
        pending = pending[~passes]
    return {
        **search,
        "required": required,
        "candidate_ladder": list(search.get("candidate_ladder", ())),
    }


def _grid_candidate_worker(
    profile_c, target, times, params, method, resolutions, repeats,
    start_index, connection,
):
    """Evaluate one grid method and report each epoch independently."""

    try:
        curve = _curve(profile_c, target)
        for local_index, x in enumerate(times):
            index = int(start_index) + local_index
            bins = resolutions[local_index]
            if bins is None:
                connection.send({"kind": "unavailable", "index": index})
                continue
            one_time = np.asarray([x], dtype=float)
            one_resolution = [int(bins)]
            warm_result = _evaluate(
                curve, one_time, params, method, one_resolution
            )
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                result = _evaluate(
                    curve, one_time, params, method, one_resolution
                )
                wall = time.perf_counter() - started
                native_seconds = np.asarray(
                    result.get("seconds", []), dtype=float
                )
                samples.append(
                    float(native_seconds[0])
                    if native_seconds.size == 1
                    and np.isfinite(native_seconds[0])
                    else float(wall)
                )
            connection.send({
                "kind": "point",
                "index": index,
                "seconds": float(np.median(np.asarray(samples))),
                "magnification": float(warm_result["magnification"][0]),
                "converged": bool(warm_result["converged"][0]),
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


def _time_grid(
    profile_c, target, times, params, method, resolutions, repeats,
    point_timeout,
):
    all_times = np.asarray(times, dtype=float)
    count = int(all_times.size)
    seconds = [None] * count
    magnification = [None] * count
    converged = [False] * count
    statuses = ["unavailable" if value is None else "error"
                for value in resolutions]
    lower_bounds = [None] * count
    start_index = 0
    context = mp.get_context("fork")

    while start_index < count:
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_grid_candidate_worker,
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
                lower_bounds[point_index] = float(
                    point_timeout if point_timeout and point_timeout > 0.0
                    else time.perf_counter()
                )
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
        "lower_bound_seconds": lower_bounds,
    }


def _vbm_candidates(target):
    values = []
    value = float(target)
    while value >= 1.0e-7 - 1.0e-15:
        values.append(value)
        value /= 10.0
    return tuple(values)


def _vbm_one(vbm, row, x, profile_c):
    # BinaryMagDark's final argument is the absolute accuracy, not a1.
    # a1 is configured on the object above; passing c positionally here is a
    # silent and serious API error.
    if profile_c:
        return float(
            vbm.BinaryMagDark(
                float(row["s"]),
                float(row["q"]),
                -float(x),
                float(row["y"]),
                float(row["rho"]),
                VBM_ABSOLUTE_FLOOR,
            )
        )
    return float(
        vbm.BinaryMag(
            float(row["s"]),
            float(row["q"]),
            -float(x),
            float(row["y"]),
            float(row["rho"]),
            VBM_ABSOLUTE_FLOOR,
        )
    )


def _vbm_candidate_worker(
    row, times, indices, profile_c, reltol, repeats, connection
):
    """Evaluate one VBM tolerance and report one reference epoch at a time."""

    try:
        vbm = _new_vbm(profile_c, reltol)
        for local_index, x in enumerate(times):
            index = int(indices[local_index])
            # Warm the object and its per-lens state before timing.
            value = _vbm_one(vbm, row, x, profile_c)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                value = _vbm_one(vbm, row, x, profile_c)
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


def _time_vbm_candidate(
    row, times, indices, profile_c, reltol, reference, repeats, point_timeout
):
    """Run one VBM tolerance with a per-reference-epoch watchdog."""

    count = int(reference.size)
    values = [None] * count
    seconds = [None] * count
    statuses = ["unrequested"] * count
    errors = [None] * count
    context = mp.get_context("fork")
    remaining = list(int(index) for index in indices)

    while remaining:
        receive, send = context.Pipe(duplex=False)
        remaining_times = np.asarray(times)[remaining]
        process = context.Process(
            target=_vbm_candidate_worker,
            args=(
                row,
                remaining_times,
                remaining,
                profile_c,
                reltol,
                repeats,
                send,
            ),
        )
        process.start()
        send.close()
        position = 0
        finished = False
        while position < len(remaining):
            point_index = remaining[position]
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
                remaining = remaining[position + 1:]
                break
            kind = message.get("kind")
            if kind == "point":
                index = int(message["index"])
                value = float(message["value"])
                values[index] = value
                seconds[index] = float(message["seconds"])
                errors[index] = float(
                    abs(value - reference[index])
                    / max(abs(reference[index]), 1.0)
                )
                statuses[index] = "completed"
                position += 1
                continue
            if kind == "error":
                statuses[point_index] = "error"
                process.terminate()
                process.join(5.0)
                remaining = remaining[position + 1:]
                break
            if kind == "done":
                finished = True
                remaining = []
                break
        if position >= len(remaining):
            remaining = []
            finished = True
        if process.is_alive():
            process.join(1.0)
        receive.close()
        if finished:
            break

    return {
        "reltol": float(reltol),
        "seconds": seconds,
        "errors": errors,
        "statuses": statuses,
        "lower_bound_seconds": [
            None if status != "timeout" else float(point_timeout)
            for status in statuses
        ],
        "max_error": max(
            (value for value in errors if value is not None),
            default=None,
        ),
    }


def _time_vbm(row, times, profile_c, target, reference, repeats, point_timeout):
    entries = []
    selected_seconds = []
    selected_reltol = []
    selected_status = ["unresolved"] * int(reference.size)
    selected_lower_bounds = [None] * int(reference.size)
    pending = set(range(int(reference.size)))
    for reltol in _vbm_candidates(target):
        if not pending:
            break
        entry = _time_vbm_candidate(
            row,
            times,
            sorted(pending),
            profile_c,
            reltol,
            reference,
            repeats,
            point_timeout,
        )
        entries.append(entry)
        for index in list(pending):
            status = entry["statuses"][index]
            if status == "timeout":
                selected_status[index] = "timeout"
                selected_lower_bounds[index] = float(point_timeout)
                pending.remove(index)
            elif (
                status == "completed"
                and entry["errors"][index] is not None
                and entry["errors"][index] <= target
            ):
                selected_status[index] = "completed"
                selected_lower_bounds[index] = None
                pending.remove(index)
        # The arrays are filled below from the selected entries.  This loop
        # only decides which points need a tighter fallback.
    selected_seconds = [None] * int(reference.size)
    selected_reltol = [None] * int(reference.size)
    for entry in entries:
        for index in range(int(reference.size)):
            if selected_status[index] != "completed":
                continue
            if selected_seconds[index] is not None:
                continue
            if (
                entry["statuses"][index] == "completed"
                and entry["errors"][index] is not None
                and entry["errors"][index] <= target
            ):
                selected_seconds[index] = float(entry["seconds"][index])
                selected_reltol[index] = float(entry["reltol"])
    return {
        "candidates": entries,
        "selected_seconds": selected_seconds,
        "selected_reltol": selected_reltol,
        "selected_status": selected_status,
        "selected_lower_bound_seconds": selected_lower_bounds,
    }


def _new_vbm(profile_c, reltol):
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = VBM_ABSOLUTE_FLOOR
    vbm.RelTol = float(reltol)
    vbm.a1 = float(profile_c)
    vbm.a2 = 0.0
    vbm.SetLDprofile(vbm.LDlinear)
    return vbm


def measure(row, target, repeats, search_missing, point_timeout):
    profile_c = float(row["limb_darkening_c"])
    times_full = _times(row)
    times_ref = times_full[list(REFERENCE_INDICES)]
    params = _params(row)
    reference = _reference(row)
    curve = _curve(profile_c, target)
    grid = {}
    for name, method in (
        ("cartesian", warmup.CARTESIAN),
        ("polar", warmup.POLAR),
    ):
        search = _required_grid(
            curve, row, times_ref, params, reference, target, method,
            search_missing,
        )
        if search.get("candidate_ladder"):
            search = _revalidate_ladder(
                curve, times_ref, params, reference, target, method, search
            )
        if any(value is None for value in search["required"]) and search_missing:
            search = _required_grid(
                curve, row, times_ref, params, reference, target, method,
                search_missing, use_existing=False,
            )
        selected = search["required"]
        seconds = None
        errors = None
        converged = None
        timed = _time_grid(
            profile_c,
            target,
            times_ref,
            params,
            method,
            selected,
            repeats,
            point_timeout,
        )
        if timed is not None:
            seconds = timed["seconds"]
            values = timed["magnification"]
            converged = timed["converged"]
            errors = [
                None if value is None else float(
                    abs(float(value) - reference[index])
                    / max(abs(reference[index]), 1.0)
                )
                for index, value in enumerate(values)
            ]
        grid[name] = {
            **search,
            "seconds": seconds,
            "errors": errors,
            "converged": converged,
            "statuses": None if timed is None else timed["statuses"],
            "lower_bound_seconds": (
                None if timed is None else timed["lower_bound_seconds"]
            ),
        }
    vbm = _time_vbm(
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
    vbm_seconds = vbm["selected_seconds"]
    vbm_status = vbm["selected_status"]
    ratios = []
    ratio_status = []
    for index in range(reference.size):
        candidates = []
        for name in ("cartesian", "polar"):
            bins = grid[name]["required"][index]
            seconds = grid[name]["seconds"]
            errors = grid[name]["errors"]
            if (
                bins is not None
                and seconds is not None
                and errors is not None
                and grid[name]["statuses"] is not None
                and grid[name]["statuses"][index] == "completed"
                and seconds[index] is not None
                and errors[index] is not None
                and errors[index] <= target
            ):
                candidates.append((float(seconds[index]), name, int(bins)))
        if not candidates:
            chosen_grid.append(None)
            chosen_nbin.append(None)
            chosen_seconds.append(None)
            ratios.append(None)
            if any(
                grid[name]["statuses"][index] == "timeout"
                for name in ("cartesian", "polar")
                if grid[name]["statuses"] is not None
            ):
                if vbm_status[index] == "timeout":
                    ratio_status.append("both_timeout")
                elif vbm_status[index] == "completed":
                    ratio_status.append("vbm_win_grid_timeout")
                else:
                    ratio_status.append("grid_timeout")
            else:
                ratio_status.append("grid_unresolved")
            continue
        best = min(candidates, key=lambda value: value[0])
        chosen_seconds.append(best[0])
        chosen_grid.append(best[1])
        chosen_nbin.append(best[2])
        if vbm_seconds[index] is None:
            ratios.append(None)
            if vbm_status[index] == "timeout":
                ratio_status.append("grid_win_vbm_timeout")
            else:
                ratio_status.append("vbm_unresolved")
        else:
            ratios.append(float(vbm_seconds[index] / best[0]))
            ratio_status.append("measured")
    return {
        "status": "completed",
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
        "reference": reference.tolist(),
        "grid": grid,
        "vbm": vbm,
        "chosen_grid": chosen_grid,
        "chosen_nbin": chosen_nbin,
        "chosen_seconds": chosen_seconds,
        "ratios_vbm_over_lcbinint": ratios,
        "ratio_status": ratio_status,
        "vbm_status": vbm_status,
        "vbm_lower_bound_seconds": vbm["selected_lower_bound_seconds"],
        "reference_floor": float(row["reference_floor"]),
    }


def _timed_measure(
    row, target, repeats, search_missing, point_timeout, job_timeout
):
    """Run one job in a killable child process.

    VBM's C++ integrator can spend an unbounded amount of time on a handful of
    tangency/high-magnification calls.  A Python signal alarm cannot reliably
    interrupt such a native loop, so the benchmark isolates each job instead.
    A timed-out job is recorded as censored and is never counted as a win or a
    loss.
    """
    if job_timeout is None or job_timeout <= 0.0:
        return measure(
            row, target, repeats, search_missing, point_timeout
        )
    context = mp.get_context("fork")
    queue = context.Queue()

    def worker():
        try:
            queue.put(measure(
                row, target, repeats, search_missing, point_timeout
            ))
        except BaseException as error:  # noqa: BLE001
            queue.put({
                "status": "error",
                "case_id": int(row["case_id"]),
                "profile": row["profile"],
                "target": float(target),
                "d_over_rho": float(row["intended_distance_factor"]),
                "error": f"{type(error).__name__}: {error}",
                "ratios_vbm_over_lcbinint": [],
                "chosen_nbin": [],
            })

    process = context.Process(target=worker)
    process.start()
    process.join(float(job_timeout))
    if process.is_alive():
        process.terminate()
        process.join(5.0)
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
    if not queue.empty():
        return queue.get()
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


def summarise(results, targets):
    summary = {}
    for profile in sorted({row["profile"] for row in results}):
        summary[profile] = {}
        for target in targets:
            rows = [
                row for row in results
                if row["profile"] == profile
                and abs(row["target"] - target) < 1.0e-15
            ]
            ratios = [
                value for row in rows
                for value in row["ratios_vbm_over_lcbinint"]
                if value is not None and math.isfinite(value)
            ]
            statuses = [
                status for row in rows
                for status in row.get("ratio_status", [])
            ]
            bins = [
                value for row in rows
                for value in row["chosen_nbin"]
                if value is not None
            ]
            grid_counts = {
                name: sum(
                    value == name
                    for row in rows
                    for value in row.get("chosen_grid", [])
                )
                for name in ("cartesian", "polar")
            }
            if ratios:
                array = np.asarray(ratios, dtype=float)
                ratio_summary = {
                    "count": int(array.size),
                    "win_rate": float(np.mean(array > 1.0)),
                    "median": float(np.median(array)),
                    "p10": float(np.percentile(array, 10)),
                    "p90": float(np.percentile(array, 90)),
                }
            else:
                ratio_summary = {"count": 0}
            if bins:
                array = np.asarray(bins, dtype=float)
                bin_summary = {
                    "count": int(array.size),
                    "median": float(np.median(array)),
                    "p10": float(np.percentile(array, 10)),
                    "p90": float(np.percentile(array, 90)),
                }
            else:
                bin_summary = {"count": 0}
            status_counts = {
                status: statuses.count(status)
                for status in sorted(set(statuses))
            }
            measured_wins = sum(
                value > 1.0 for row in rows
                for value in row["ratios_vbm_over_lcbinint"]
                if value is not None and math.isfinite(value)
            )
            classified_grid_wins = (
                measured_wins
                + status_counts.get("grid_win_vbm_timeout", 0)
            )
            classified_vbm_wins = (
                len(ratios) - measured_wins
                + status_counts.get("vbm_win_grid_timeout", 0)
            )
            classified = classified_grid_wins + classified_vbm_wins
            total_points = len(statuses)
            summary[profile][str(target)] = {
                "rows": len(rows),
                "grid_counts": grid_counts,
                "nbin": bin_summary,
                "ratio_vbm_over_lcbinint": ratio_summary,
                "comparison": {
                    "total_points": total_points,
                    "status_counts": status_counts,
                    "classified_points": classified,
                    "grid_wins": classified_grid_wins,
                    "vbm_wins": classified_vbm_wins,
                    "grid_win_rate_among_classified": (
                        None if not classified
                        else classified_grid_wins / classified
                    ),
                    "grid_win_rate_with_both_timeout_unresolved": (
                        None if not total_points
                        else classified_grid_wins / total_points
                    ),
                },
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/diagnostics/results/recal2026/speed_discovery"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/diagnostics/results/recal2026/grid_vs_vbm_dark"),
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=8,
        help="number of geometries sampled per d/rho stratum (both profiles)",
    )
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=list(DEFAULT_FACTORS),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("uniform", "linear"),
        default=[],
    )
    parser.add_argument("--rho-min", type=float)
    parser.add_argument("--rho-max", type=float)
    parser.add_argument("--q-min", type=float)
    parser.add_argument("--q-max", type=float)
    parser.add_argument("--d-max", type=float)
    parser.add_argument("--magnification-min", type=float)
    parser.add_argument("--magnification-max", type=float)
    parser.add_argument(
        "--route-filter",
        choices=("all", "grid"),
        default="all",
        help="restrict jobs to the recorded pure production grid route",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=0.0,
        help="emergency seconds per job; non-positive disables this guard",
    )
    parser.add_argument(
        "--point-timeout",
        type=float,
        default=300.0,
        help="shared seconds per reference epoch for grid and VBM",
    )
    parser.add_argument(
        "--search-missing",
        action="store_true",
        help="run the full warm-up search when the old Nbin ladder did not qualify",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=list(TARGETS),
    )
    arguments = parser.parse_args()
    rows = _select_rows(
        _load_rows(arguments.input),
        arguments.case_count,
        arguments.factors,
        arguments.seed,
    )
    rows = _filter_rows(
        rows,
        profiles=set(arguments.profiles) if arguments.profiles else None,
        rho_min=arguments.rho_min,
        rho_max=arguments.rho_max,
        q_min=arguments.q_min,
        q_max=arguments.q_max,
        d_max=arguments.d_max,
        magnification_min=arguments.magnification_min,
        magnification_max=arguments.magnification_max,
    )
    if not rows:
        raise SystemExit("no rows selected")
    arguments.output.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows:
        for target in arguments.targets:
            if arguments.route_filter == "grid" and not _pure_grid_route(
                row, target
            ):
                continue
            if not _usable(row, target):
                continue
            selected.append((row, target))
    print(
        f"selected {len(rows)} blocks, {len(selected)} block/profile/target jobs; "
        f"{len(rows) * len(REFERENCE_INDICES)} reference epochs per target",
        flush=True,
    )
    results = []
    started = time.perf_counter()
    partial_path = arguments.output / "results.partial.json"
    for index, (row, target) in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] case={row['case_id']} "
            f"d/rho={row['intended_distance_factor']} "
            f"profile={row['profile']} target={target:g}",
            flush=True,
        )
        result = _timed_measure(
            row,
            target,
            arguments.repeats,
            arguments.search_missing,
            arguments.point_timeout,
            arguments.job_timeout,
        )
        results.append(result)
        if result.get("status") != "completed":
            print(
                f"  {result.get('status')} case={row['case_id']} "
                f"d/rho={row['intended_distance_factor']} "
                f"profile={row['profile']} target={target:g}",
                flush=True,
            )
        partial_path.write_text(json.dumps({
            "input": str(arguments.input),
            "case_count": arguments.case_count,
            "factors": list(arguments.factors),
            "seed": arguments.seed,
            "repeats": arguments.repeats,
            "search_missing": arguments.search_missing,
            "point_timeout": arguments.point_timeout,
            "job_timeout": arguments.job_timeout,
            "targets": list(arguments.targets),
            "profiles": list(arguments.profiles),
            "route_filter": arguments.route_filter,
            "filters": {
                "rho_min": arguments.rho_min,
                "rho_max": arguments.rho_max,
                "q_min": arguments.q_min,
                "q_max": arguments.q_max,
                "d_max": arguments.d_max,
                "magnification_min": arguments.magnification_min,
                "magnification_max": arguments.magnification_max,
            },
            "reference_indices": list(REFERENCE_INDICES),
            "results": results,
        }, indent=2))
    payload = {
        "input": str(arguments.input),
        "case_count": arguments.case_count,
        "factors": list(arguments.factors),
        "seed": arguments.seed,
        "repeats": arguments.repeats,
        "search_missing": arguments.search_missing,
        "point_timeout": arguments.point_timeout,
        "job_timeout": arguments.job_timeout,
        "targets": list(arguments.targets),
        "profiles": list(arguments.profiles),
        "route_filter": arguments.route_filter,
        "filters": {
            "rho_min": arguments.rho_min,
            "rho_max": arguments.rho_max,
            "q_min": arguments.q_min,
            "q_max": arguments.q_max,
            "d_max": arguments.d_max,
            "magnification_min": arguments.magnification_min,
            "magnification_max": arguments.magnification_max,
        },
        "reference_indices": list(REFERENCE_INDICES),
        "results": results,
        "summary": summarise(results, arguments.targets),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (arguments.output / "results.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved {arguments.output / 'results.json'}")


if __name__ == "__main__":
    main()
