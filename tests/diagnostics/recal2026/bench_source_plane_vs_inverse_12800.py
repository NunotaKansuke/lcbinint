#!/usr/bin/env python3
"""Add a forced source-plane comparison to the controlled 12,800-epoch test.

The input is the final controlled pure-kernel result.  Its selected
Cartesian/polar inverse-ray times are reused as the baseline, so this script
does not rerun the inverse-ray benchmark.  Source-plane quadrature is forced
through the preplanned native API; the production automatic dispatcher is not
used.

For each epoch, source-plane panels are tested on the same self-convergence
ladder used by the native grazing-quadrature route: 48 -> 96, followed by
192 only if the 48/96 pair does not satisfy the requested tolerance.  Search
time is excluded from the final source-plane timing.  A source-plane kernel
call is stopped when it exceeds ``timeout_factor`` times the saved inverse-ray
kernel time for that epoch.  A timeout is reported as such and is never
counted as a source-plane win or loss.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bench_grid_vs_vbm_pure_kernel as pure  # noqa: E402


DEFAULT_INPUT = Path(
    "/tmp/lcbinint_optimize_irs_polar_rerun_20260813/merged/results.json"
)
DEFAULT_OUTPUT = Path("/tmp/lcbinint_source_plane_vs_inverse_12800_20260814")
REFERENCE_INDICES = tuple(pure.REFERENCE_INDICES)
SOURCE_PLANE = pure.warmup.SOURCE_PLANE
SOURCE_RESOLUTIONS = (48, 96, 192)
SELF_CONFIRMATION_FLOOR = 1.0


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _row(result):
    return {
        "case_id": result["case_id"],
        "s": result["s"],
        "q": result["q"],
        "rho": result["rho"],
        "x": result["x"],
        "y": result["y"],
    }


def _relative_change(fine, coarse):
    if not (_finite(fine) and _finite(coarse)):
        return None
    return float(abs(float(fine) - float(coarse)) /
                 max(abs(float(fine)), SELF_CONFIRMATION_FLOOR))


def _timeout_budget(inverse_seconds, timeout_factor):
    return float(timeout_factor) * float(inverse_seconds)


def _evaluate_source_plane(curve, source_x, source_y, params, resolution):
    """Evaluate duplicate XY epochs and return the cache-warm second epoch."""

    result = pure._evaluate(
        curve,
        [float(source_x), float(source_x)],
        params,
        SOURCE_PLANE,
        [int(resolution), int(resolution)],
        source_y=[float(source_y), float(source_y)],
    )
    methods = result.get("method", ())
    if len(methods) < 2 or any(int(value) != SOURCE_PLANE for value in methods[:2]):
        raise RuntimeError(
            "preplanned source-plane call changed method: "
            f"expected {SOURCE_PLANE}, got {methods}"
        )
    seconds = np.asarray(result.get("seconds", ()), dtype=float)
    values = np.asarray(result.get("magnification", ()), dtype=float)
    if seconds.size < 2 or values.size < 2:
        raise RuntimeError("source-plane diagnostic returned no second epoch")
    if not np.isfinite(seconds[1]) or not np.isfinite(values[1]):
        raise RuntimeError("source-plane diagnostic returned a non-finite value")
    return float(values[1]), float(seconds[1])


def _source_plane_worker(
    profile_c,
    target,
    source_x,
    source_y,
    params,
    indices,
    resolutions,
    repeats,
    mode,
    connection,
):
    """Run a batch while exposing a watchdog boundary for every kernel call."""

    try:
        curve = pure.base._curve(float(profile_c), float(target))
        for local_index, index_value in enumerate(indices):
            index = int(index_value)
            resolution = int(resolutions[local_index])

            connection.send({"kind": "start", "index": index, "phase": "probe"})
            value, probe_seconds = _evaluate_source_plane(
                curve,
                source_x[local_index],
                source_y[local_index],
                params,
                resolution,
            )
            if mode == "probe":
                connection.send({
                    "kind": "probe",
                    "index": index,
                    "resolution": resolution,
                    "value": value,
                    "seconds": probe_seconds,
                })
                continue

            samples = []
            for repeat in range(int(repeats)):
                connection.send({
                    "kind": "start",
                    "index": index,
                    "phase": "timing",
                    "repeat": repeat,
                })
                timed_value, timed_seconds = _evaluate_source_plane(
                    curve,
                    source_x[local_index],
                    source_y[local_index],
                    params,
                    resolution,
                )
                if not np.isfinite(timed_value) or not np.isfinite(timed_seconds):
                    raise RuntimeError("non-finite source-plane timing result")
                samples.append(timed_seconds)
            connection.send({
                "kind": "timed",
                "index": index,
                "resolution": resolution,
                "value": value,
                "seconds": float(np.median(np.asarray(samples, dtype=float))),
                "samples": [float(sample) for sample in samples],
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


def _run_source_batch(
    profile_c,
    target,
    source_x,
    source_y,
    params,
    indices,
    resolutions,
    inverse_seconds,
    timeout_factor,
    repeats=1,
    mode="probe",
):
    """Run a source-plane batch with an epoch-specific relative watchdog."""

    indices = [int(index) for index in indices]
    if not indices:
        return {}
    source_x = np.asarray(source_x, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    resolutions = [int(value) for value in resolutions]
    if len(indices) != len(resolutions):
        raise ValueError("indices and resolutions must have equal length")

    outcomes = {}
    pending = list(range(len(indices)))
    context = mp.get_context("fork")

    while pending:
        pending_indices = [indices[position] for position in pending]
        pending_x = [source_x[index] for index in pending_indices]
        pending_y = [source_y[index] for index in pending_indices]
        pending_resolutions = [resolutions[position] for position in pending]
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_source_plane_worker,
            args=(
                profile_c,
                target,
                pending_x,
                pending_y,
                params,
                pending_indices,
                pending_resolutions,
                repeats,
                mode,
                send,
            ),
        )
        process.start()
        send.close()

        pending_position = 0
        # Start the watchdog before the child sends its first progress
        # message.  A very expensive native call may otherwise prevent the
        # initial ``start`` message from reaching the parent at all.
        current_index = indices[pending[0]]
        current_deadline = (
            time.perf_counter()
            + _timeout_budget(
                inverse_seconds[current_index],
                timeout_factor,
            )
        )
        finished = False
        while pending_position < len(pending):
            message = None
            while message is None:
                if current_deadline is None:
                    wait = 0.25
                else:
                    wait = max(0.0, current_deadline - time.perf_counter())
                    if wait == 0.0:
                        break
                    wait = min(wait, 0.25)
                if receive.poll(wait):
                    message = receive.recv()
                    break
                if not process.is_alive():
                    break

            if message is None:
                if current_index is not None:
                    outcomes[current_index] = {
                        "status": "timeout",
                        "timeout_seconds": float(
                            _timeout_budget(
                                inverse_seconds[current_index],
                                timeout_factor,
                            )
                        ),
                    }
                    timed_out_position = pending_position
                    pending = pending[timed_out_position + 1:]
                else:
                    for position in pending[pending_position:]:
                        index = indices[position]
                        outcomes[index] = {"status": "error"}
                    pending = []
                process.terminate()
                process.join(0.2)
                if process.is_alive():
                    process.kill()
                    process.join(1.0)
                break

            kind = message.get("kind")
            if kind == "start":
                current_index = int(message["index"])
                budget = _timeout_budget(
                    inverse_seconds[current_index],
                    timeout_factor,
                )
                current_deadline = time.perf_counter() + budget
                continue

            if kind in ("probe", "timed"):
                index = int(message["index"])
                payload = dict(message)
                payload.pop("kind", None)
                payload["status"] = "completed"
                outcomes[index] = payload
                current_index = None
                current_deadline = None
                pending_position += 1
                continue

            if kind == "error":
                if current_index is not None:
                    outcomes[current_index] = {
                        "status": "error",
                        "error": message.get("error"),
                    }
                    pending = pending[pending_position + 1:]
                process.terminate()
                process.join(0.2)
                if process.is_alive():
                    process.kill()
                    process.join(1.0)
                break

            if kind == "done":
                finished = True
                pending = []
                break

        if pending_position >= len(pending):
            finished = True
            pending = []
        if process.is_alive():
            process.join(1.0)
        receive.close()
        if finished:
            break

    return outcomes


def _measure_epoch_source_plane(
    result,
    timeout_factor,
    repeats,
):
    """Self-converge and time source-plane for the four reference epochs."""

    row = _row(result)
    params = pure.base._params(row)
    times = pure.base._times(row)[list(REFERENCE_INDICES)]
    source_y = np.full(times.shape, float(row["y"]), dtype=float)
    inverse_seconds = [
        None if value is None else float(value)
        for value in result.get("chosen_seconds", ())
    ]
    count = len(REFERENCE_INDICES)
    output = {
        "status": ["inverse_unresolved"] * count,
        "resolution": [None] * count,
        "seconds": [None] * count,
        "magnification": [None] * count,
        "ratio_inverse_over_source_plane": [None] * count,
        "timeout_seconds": [None] * count,
        "self_convergence_error": [None] * count,
        "samples": [[] for _ in range(count)],
    }

    active = [
        index for index, value in enumerate(inverse_seconds)
        if _finite(value) and float(value) > 0.0
    ]
    for index in active:
        output["status"][index] = "searching"

    candidate_values = {index: {} for index in active}
    candidate_seconds = {index: {} for index in active}

    for resolution in SOURCE_RESOLUTIONS[:2]:
        pending = [
            index for index in active
            if output["status"][index] == "searching"
        ]
        if not pending:
            break
        probes = _run_source_batch(
            float(result["limb_darkening_c"]),
            float(result["target"]),
            times,
            source_y,
            params,
            pending,
            [resolution] * len(pending),
            inverse_seconds,
            timeout_factor,
            mode="probe",
        )
        for index in pending:
            item = probes.get(index, {"status": "error"})
            if item.get("status") != "completed":
                output["status"][index] = item.get("status", "error")
                output["timeout_seconds"][index] = item.get("timeout_seconds")
                continue
            value = float(item["value"])
            candidate_values[index][resolution] = value
            candidate_seconds[index][resolution] = float(item["seconds"])
            output["samples"][index].append({
                "resolution": resolution,
                "magnification": value,
                "kernel_seconds": float(item["seconds"]),
            })

    pending_192 = []
    for index in active:
        if output["status"][index] != "searching":
            continue
        change = _relative_change(
            candidate_values[index].get(96),
            candidate_values[index].get(48),
        )
        if change is not None and change <= float(result["target"]):
            output["resolution"][index] = 96
            output["self_convergence_error"][index] = change
            continue
        pending_192.append(index)

    if pending_192:
        probes = _run_source_batch(
            float(result["limb_darkening_c"]),
            float(result["target"]),
            times,
            source_y,
            params,
            pending_192,
            [192] * len(pending_192),
            inverse_seconds,
            timeout_factor,
            mode="probe",
        )
        for index in pending_192:
            item = probes.get(index, {"status": "error"})
            if item.get("status") != "completed":
                output["status"][index] = item.get("status", "error")
                output["timeout_seconds"][index] = item.get("timeout_seconds")
                continue
            value = float(item["value"])
            candidate_values[index][192] = value
            candidate_seconds[index][192] = float(item["seconds"])
            output["samples"][index].append({
                "resolution": 192,
                "magnification": value,
                "kernel_seconds": float(item["seconds"]),
            })
            change = _relative_change(value, candidate_values[index].get(96))
            if change is not None and change <= float(result["target"]):
                output["resolution"][index] = 192
                output["self_convergence_error"][index] = change
            else:
                output["status"][index] = "self_unresolved"

    selected = [
        index for index in active
        if output["resolution"][index] is not None
        and output["status"][index] == "searching"
    ]
    if selected:
        final = _run_source_batch(
            float(result["limb_darkening_c"]),
            float(result["target"]),
            times,
            source_y,
            params,
            selected,
            [output["resolution"][index] for index in selected],
            inverse_seconds,
            timeout_factor,
            repeats=repeats,
            mode="timed",
        )
        for index in selected:
            item = final.get(index, {"status": "error"})
            if item.get("status") != "completed":
                output["status"][index] = item.get("status", "error")
                output["timeout_seconds"][index] = item.get("timeout_seconds")
                continue
            seconds = float(item["seconds"])
            output["status"][index] = "completed"
            output["seconds"][index] = seconds
            output["magnification"][index] = float(item["value"])
            output["ratio_inverse_over_source_plane"][index] = (
                inverse_seconds[index] / seconds
            )

    return output


def _percentiles(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "source_plane_wins": int(np.sum(values > 1.0)),
        "inverse_ray_wins": int(np.sum(values <= 1.0)),
    }


def _summary(results):
    by_condition = {}
    all_ratios = []
    statuses = Counter()
    for result in results:
        key = (result["profile"], f"{float(result['target']):g}")
        entry = by_condition.setdefault(key, {"ratios": [], "statuses": Counter()})
        for status, ratio in zip(
            result["source_plane"]["status"],
            result["source_plane"]["ratio_inverse_over_source_plane"],
        ):
            entry["statuses"][status] += 1
            statuses[status] += 1
            if _finite(ratio) and float(ratio) > 0.0:
                entry["ratios"].append(float(ratio))
                all_ratios.append(float(ratio))
    formatted = {}
    for key, entry in sorted(by_condition.items()):
        formatted[f"{key[0]}|{key[1]}"] = {
            "profile": key[0],
            "target": float(key[1]),
            "statuses": dict(entry["statuses"]),
            "ratio_inverse_over_source_plane": _percentiles(entry["ratios"]),
        }
    return {
        "nominal_epochs": len(results) * len(REFERENCE_INDICES),
        "statuses": dict(statuses),
        "ratio_inverse_over_source_plane": _percentiles(all_ratios),
        "by_condition": formatted,
    }


def _report(payload):
    summary = payload["summary"]
    lines = [
        "# Forced source-plane comparison for the controlled 12,800-epoch test",
        "",
        "This adds forced `source_plane_quadrature` to the existing controlled",
        "pure-kernel benchmark. The inverse-ray baseline is the saved selected",
        "Cartesian/polar cache-warm native time from the same epoch.",
        "",
        f"- Nominal epochs: {summary['nominal_epochs']}",
        f"- Timeout rule: source-plane kernel call > "
        f"{payload['timeout_factor']:g} x inverse-ray kernel time",
        "- Source-plane self-convergence: 48 -> 96, then 96 -> 192 if needed",
        "- Ratio: `R_source = t_inverse_ray / t_source_plane`; `R_source > 1` "
        "means source-plane is faster",
        "- Search and warm-up time are excluded from the reported source-plane "
        "kernel time",
        "",
        "## Overall",
        "",
        "| source-plane status | count |",
        "|---|---:|",
    ]
    for status, count in sorted(summary["statuses"].items()):
        lines.append(f"| `{status}` | {count} |")
    ratio = summary["ratio_inverse_over_source_plane"]
    lines += [
        "",
        "| measured ratio count | p10 | p25 | p50 | p75 | p90 | source-plane wins | inverse-ray wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {ratio.get('count', 0)} | {ratio.get('p10', float('nan')):.3g} | "
        f"{ratio.get('p25', float('nan')):.3g} | {ratio.get('p50', float('nan')):.3g} | "
        f"{ratio.get('p75', float('nan')):.3g} | {ratio.get('p90', float('nan')):.3g} | "
        f"{ratio.get('source_plane_wins', 0)} | {ratio.get('inverse_ray_wins', 0)} |",
        "",
        "## By profile and requested tolerance",
        "",
        "| profile | epsilon_rel | measured | timeout | unresolved | p50 R_source | p10 | p90 | source-plane wins | inverse-ray wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_condition"].values():
        stats = item["ratio_inverse_over_source_plane"]
        statuses = item["statuses"]
        lines.append(
            f"| {item['profile']} | `{item['target']:g}` | {stats.get('count', 0)} | "
            f"{statuses.get('timeout', 0)} | "
            f"{statuses.get('self_unresolved', 0)} | "
            f"{stats.get('p50', float('nan')):.3g} | "
            f"{stats.get('p10', float('nan')):.3g} | "
            f"{stats.get('p90', float('nan')):.3g} | "
            f"{stats.get('source_plane_wins', 0)} | "
            f"{stats.get('inverse_ray_wins', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-factor", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--limit-jobs", type=int, default=0,
        help="process only the first N jobs; 0 runs all 3,200 jobs",
    )
    args = parser.parse_args()
    if args.timeout_factor <= 0.0:
        raise SystemExit("--timeout-factor must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    payload = json.loads(args.input.read_text())
    jobs = list(payload["results"])
    if args.limit_jobs > 0:
        jobs = jobs[:args.limit_jobs]
    print(
        f"processing {len(jobs)} jobs / {len(jobs) * len(REFERENCE_INDICES)} epochs; "
        f"timeout={args.timeout_factor:g}x baseline",
        flush=True,
    )

    results = []
    started = time.perf_counter()
    for number, job in enumerate(jobs, 1):
        print(
            f"[{number}/{len(jobs)}] case={job['case_id']} "
            f"profile={job['profile']} epsilon={float(job['target']):g}",
            flush=True,
        )
        source_plane = _measure_epoch_source_plane(
            job,
            float(args.timeout_factor),
            int(args.repeats),
        )
        results.append({
            "case_id": job["case_id"],
            "profile": job["profile"],
            "limb_darkening_c": job["limb_darkening_c"],
            "target": job["target"],
            "s": job["s"],
            "q": job["q"],
            "rho": job["rho"],
            "x": job["x"],
            "y": job["y"],
            "d_over_rho": job.get("actual_d_over_rho", job.get("d_over_rho")),
            "inverse_ray_grid": job.get("chosen_grid"),
            "inverse_ray_nbin": job.get("chosen_nbin"),
            "inverse_ray_seconds": job.get("chosen_seconds"),
            "source_plane": source_plane,
        })

    output = {
        "input": str(args.input),
        "build_extension": str(pure.BUILD_EXTENSION),
        "timing_mode": "forced_source_plane_vs_saved_inverse_ray_native_kernel",
        "coordinate_mode": "direct_internal_source_xy",
        "comparison_mode": "forced_source_plane_vs_selected_inverse_ray",
        "source_plane_method": "source_plane_quadrature",
        "source_plane_resolutions": list(SOURCE_RESOLUTIONS),
        "timeout_factor": float(args.timeout_factor),
        "repeats": int(args.repeats),
        "reference_indices": list(REFERENCE_INDICES),
        "results": results,
        "summary": _summary(results),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(output, indent=2))
    (args.output / "REPORT_source_plane_vs_inverse_12800.md").write_text(
        _report(output)
    )
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
