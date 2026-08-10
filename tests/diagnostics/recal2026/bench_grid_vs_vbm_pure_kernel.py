#!/usr/bin/env python3
"""Compare one warmed finite-source integration epoch at a time.

This is the cache-warm kernel benchmark requested after the block-level
review.  The first of two identical lcbinint epochs constructs the LensModel
and its caustic cache; only the second epoch's native C++ timing is recorded.
Thus LensModel construction, caustic-cache construction, and Python/pybind
call overhead are outside the lcbinint timing.  VBM is warmed on the same
epoch before its direct call is timed.

The Nbin used here is read from the already-certified speed-discovery corpus:
it is the smallest stored Cartesian/polar knob whose recorded worst error
meets the requested target.  This benchmark does not repeat Nbin discovery.
``--search-missing`` is an explicit fallback for corpus rows without such a
stored value.

This intentionally measures a pure per-epoch kernel, not the total cost of a
LightCurve call.  ``bench_grid_vs_vbm_dark.py`` remains the historical
single-epoch cold-LensModel benchmark; this file is the corrected companion.
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
    if not hasattr(module.LightCurve()._native, "_evaluate_preplanned"):
        raise RuntimeError("selected build lacks _evaluate_preplanned")
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


def _evaluate(curve, times, params, method, resolutions):
    return curve._native._evaluate_preplanned(
        np.asarray(times, dtype=float),
        params,
        [int(method)] * len(times),
        [int(value) for value in resolutions],
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
            value = base._vbm_one(vbm, row, x, profile_c)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                value = base._vbm_one(vbm, row, x, profile_c)
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


def _required_search(curve, row, times_ref, params, reference, target, method,
                     search_missing):
    # The previous speed corpus already contains the certified minimum knob
    # for each method and target.  Re-running the ladder here would turn a
    # pure timing benchmark back into an Nbin-search benchmark, and can be
    # especially expensive at high magnification.
    stored = base._existing_required(row, target, method)
    ladder = base._existing_ladder(row, method)
    if stored is not None:
        count = int(reference.size)
        return {
            "source": "speed_discovery_saved_minimum",
            "predicted": [int(stored)] * count,
            "required": [int(stored)] * count,
            "candidate_ladder": ladder,
        }
    if search_missing:
        return base._required_grid(
            curve,
            row,
            times_ref,
            params,
            reference,
            target,
            method,
            search_missing,
            use_existing=False,
        )
    count = int(reference.size)
    return {
        "source": "unreached_in_speed_discovery",
        "predicted": [None] * count,
        "required": [None] * count,
        "candidate_ladder": ladder,
    }


def measure(row, target, repeats, search_missing, point_timeout):
    profile_c = float(row["limb_darkening_c"])
    times_full = base._times(row)
    times_ref = times_full[list(REFERENCE_INDICES)]
    params = base._params(row)
    reference = base._reference(row)
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
        errors = [
            None if value is None else float(
                abs(float(value) - reference[index]) /
                max(abs(reference[index]), 1.0)
            )
            for index, value in enumerate(values)
        ]
        grid[name] = {
            **search,
            "seconds": timed["seconds"],
            "errors": errors,
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
    vbm_pass = [
        status == "completed"
        and error is not None
        and error <= target
        for status, error in zip(vbm_timed["statuses"], vbm_timed["errors"])
    ]

    chosen_grid = []
    chosen_nbin = []
    chosen_seconds = []
    ratios = []
    ratio_status = []
    for index in range(reference.size):
        candidates = []
        for name in ("cartesian", "polar"):
            candidate = grid[name]
            if (
                candidate["required"][index] is not None
                and candidate["seconds"][index] is not None
                and candidate["statuses"][index] == "completed"
                and candidate["converged"][index]
                and candidate["errors"][index] is not None
                and candidate["errors"][index] <= target
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
            continue
        best = min(candidates, key=lambda value: value[0])
        chosen_seconds.append(best[0])
        chosen_grid.append(best[1])
        chosen_nbin.append(best[2])
        if not vbm_pass[index] or vbm_timed["seconds"][index] is None:
            ratios.append(None)
            ratio_status.append("vbm_unresolved")
        else:
            ratios.append(float(vbm_timed["seconds"][index] / best[0]))
            ratio_status.append("measured")

    return {
        "status": "completed",
        "timing_mode": "pure_kernel_cache_warm",
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
        "reference": reference.tolist(),
        "grid": grid,
        "vbm": {
            "selected_seconds": vbm_timed["seconds"],
            "selected_reltol": [float(target)] * int(reference.size),
            "selected_status": vbm_timed["statuses"],
            "errors": vbm_timed["errors"],
            "pass": vbm_pass,
        },
        "chosen_grid": chosen_grid,
        "chosen_nbin": chosen_nbin,
        "chosen_seconds": chosen_seconds,
        "ratios_vbm_over_lcbinint": ratios,
        "ratio_status": ratio_status,
        "vbm_status": vbm_timed["statuses"],
        "vbm_lower_bound_seconds": [None] * int(reference.size),
        "reference_floor": float(row["reference_floor"]),
    }


def _timed_measure(row, target, repeats, search_missing, point_timeout,
                   job_timeout):
    if job_timeout is None or job_timeout <= 0.0:
        return measure(row, target, repeats, search_missing, point_timeout)
    context = mp.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def worker():
        try:
            send_result = measure(
                row, target, repeats, search_missing, point_timeout
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("tests/diagnostics/results/recal2026/speed_discovery"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=8)
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
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--search-missing", action="store_true")
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
    rows = base._select_rows(
        base._load_rows(args.input), args.case_count, args.factors, args.seed
    )
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
            if not base._usable(row, target):
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
            args.point_timeout, args.job_timeout,
        )
        results.append(result)
        if result.get("status") != "completed":
            print(f"  {result.get('status')}", flush=True)

    payload = {
        "input": str(args.input),
        "case_count": args.case_count,
        "factors": list(args.factors),
        "seed": args.seed,
        "repeats": args.repeats,
        "search_missing": args.search_missing,
        "point_timeout": args.point_timeout,
        "job_timeout": args.job_timeout,
        "targets": list(args.targets),
        "profiles": list(args.profiles),
        "route_filter": args.route_filter,
        "timing_mode": "pure_kernel_cache_warm",
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
