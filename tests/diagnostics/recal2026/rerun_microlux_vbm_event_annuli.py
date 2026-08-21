#!/usr/bin/env python3
"""Rerun microLUX with the event-specific VBMicrolensing annulus count.

Each input row is one four-epoch batched event.  ``BinaryMagDark`` can choose
different adaptive ``nannuli`` values at those four epochs, while microLUX's
``n_annuli`` is a static JAX argument for the whole batched callable.  This
runner therefore uses the largest VBM count in the row for all four epochs.
The per-epoch VBM counts are retained in the output so the policy is explicit
and auditable.

Only microLUX is rerun.  The saved JAX/native rows and VBM reference values
are reused, and the reported timing starts after the selected microLUX
callable has been compiled and warmed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
DEFAULT_INPUT = (
    ROOT
    / "tests/diagnostics/results/recal2026/"
    / "jax_microlux_12800_final_adaptive_v6_20260819/results.json"
)
DEFAULT_SHIM = Path("/tmp/lcbinint_vbm_nannuli.so")
VBM_ABSOLUTE_FLOOR = 1.0e-12
VBM_DEFAULT_MAX_ANNULI = 100


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "bench_jax_microlux_12800",
        SCRIPT.parent / "bench_jax_microlux_12800.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vbm_library_dir():
    import VBMicrolensing

    directory = Path(VBMicrolensing.__file__).resolve().parent / "lib"
    if not (directory / "VBMicrolensingLibrary.h").is_file():
        raise RuntimeError(f"VBMicrolensing source library not found: {directory}")
    return directory


def _build_vbm_shim(path):
    """Build the diagnostic shim once, outside the measured microLUX path."""

    path = Path(path).expanduser().resolve()
    source = SCRIPT.parent / "vbm_nannuli_shim.cpp"
    library_dir = _vbm_library_dir()
    header = library_dir / "VBMicrolensingLibrary.h"
    newest_input = max(source.stat().st_mtime, header.stat().st_mtime)
    if path.is_file() and path.stat().st_mtime >= newest_input:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}"
    )
    command = [
        "g++",
        "-O3",
        "-std=c++11",
        "-fPIC",
        "-shared",
        f"-I{library_dir}",
        str(source),
        str(library_dir / "VBMicrolensingLibrary.cpp"),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


class VBMAnnuliProbe:
    """Sequential access to VBM's public final adaptive nannuli field."""

    def __init__(self, library_path):
        self.library_path = str(Path(library_path).resolve())
        library = ctypes.CDLL(self.library_path)
        self._library = library
        self._create = library.vbm_nannuli_create
        self._create.argtypes = [ctypes.c_double, ctypes.c_double]
        self._create.restype = ctypes.c_void_p
        self._destroy = library.vbm_nannuli_destroy
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None
        self._call = library.vbm_nannuli_binary_mag_dark
        self._call.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_double] * 6
            + [ctypes.POINTER(ctypes.c_int)]
        )
        self._call.restype = ctypes.c_double

    def measure(self, row, times, target, profile_c):
        handle = self._create(float(target), float(profile_c))
        if not handle:
            raise RuntimeError("VBMicrolensing nannuli object creation failed")
        values = []
        counts = []
        started = time.perf_counter()
        try:
            for active_time in times:
                count = ctypes.c_int(-1)
                value = self._call(
                    handle,
                    float(row["s"]),
                    float(row["q"]),
                    float(active_time),
                    float(row["y"]),
                    float(row["rho"]),
                    VBM_ABSOLUTE_FLOOR,
                    ctypes.byref(count),
                )
                values.append(float(value))
                counts.append(int(count.value))
        finally:
            self._destroy(handle)
        if not counts or any(value < 1 for value in counts):
            raise RuntimeError(f"invalid VBM nannuli values: {counts!r}")
        return {
            "values": values,
            "nannuli": counts,
            "selected_n_annuli": max(counts),
            "seconds": float(time.perf_counter() - started),
        }


def _finite(value, bench):
    return bench._finite(value)


def _finite_max(values, bench):
    if values is None:
        return None
    finite = [float(value) for value in values if _finite(value, bench)]
    return max(finite) if finite else None


def _row_key(row):
    """Stable identity for one corpus event across parent/child processes."""

    return "|".join(
        (
            str(int(row["case_id"])),
            str(row["profile"]),
            f"{float(row['target']):.17g}",
            f"{float(row['x']):.17g}",
            f"{float(row['y']):.17g}",
        )
    )


def _pass_status(errors, references, uncertainties, target, bench):
    checked = [
        index
        for index, reference in enumerate(references)
        if _finite(reference, bench)
    ]
    certified = bool(
        checked
        and uncertainties is not None
        and len(uncertainties) == len(references)
        and all(
            _finite(value, bench)
            and float(value)
            <= bench.REFERENCE_UNCERTAINTY_FRACTION * float(target)
            for value in uncertainties
        )
    )
    passed = bool(
        certified
        and all(
            _finite(errors[index], bench)
            and float(errors[index]) <= float(target)
            for index in checked
        )
    )
    return certified, passed


def _error_record(row, profile, target, times, stage, error):
    return {
        "status": "timeout" if error.__class__.__name__ == "_DeadlineExceeded" else "error",
        "error": f"{type(error).__name__}: {error}",
        "error_stage": stage,
        "case_id": int(row["case_id"]),
        "profile": profile,
        "target": target,
        "x": float(row["x"]),
        "y": float(row["y"]),
        "batch_epochs": len(times),
    }


def _row_result(row, executor, probe, bench, args, event_record=None):
    profile = str(row["profile"])
    target = float(row["target"])
    profile_c = float(row.get("limb_darkening_c", 0.0))
    times = bench._times(row)
    references, uncertainties, reference_source = bench._reference_bundle(row)
    if uncertainties is None:
        uncertainties = row.get("reference_uncertainty")
    if reference_source is None and uncertainties is not None:
        reference_source = "saved_corpus_vbm_fine_reltol_1e-7"

    vbm_probe = None
    stage = "vbm_nannuli"
    try:
        if event_record is not None:
            vbm_probe = event_record
            event_n_annuli = int(vbm_probe["selected_n_annuli"])
        elif profile == "linear":
            with bench._deadline(args.vbm_timeout):
                vbm_probe = probe.measure(row, times, target, profile_c)
            event_n_annuli = int(vbm_probe["selected_n_annuli"])
        else:
            event_n_annuli = None

        stage = "select"
        selection = executor.select(
            profile,
            profile_c,
            target,
            row,
            times,
            references,
            uncertainties,
            args.forward_timeout,
            n_annuli_override=event_n_annuli,
        )
        selected_n_annuli = selection["selected_n_annuli"]
        stage = "timing"
        timing = executor.timed(
            profile,
            profile_c,
            target,
            selected_n_annuli,
            row,
            times,
            args.repeats,
            args.forward_timeout,
            args.derivative_timeout,
            False,
        )
    except Exception as error:  # noqa: BLE001
        return _error_record(row, profile, target, times, stage, error)

    values = timing["forward_values"]
    derivatives = timing["dA_dt_values"]
    errors = [
        bench._relative_error(value, reference)
        for value, reference in zip(values, references)
    ]
    certified, passed = _pass_status(
        errors, references, uncertainties, target, bench
    )
    epoch_count = len(times)
    forward_block = timing["forward_block_seconds"]
    derivative_block = timing["dA_dt_block_seconds"]
    selected_calibration = selection.get("selected_entry") or {}
    warnings = list(selected_calibration.get("warnings", ()))
    warnings.extend(timing.get("warnings", ()))
    return {
        "status": "completed",
        "case_id": int(row["case_id"]),
        "input_status": row.get("status"),
        "profile": profile,
        "target": target,
        "limb_darkening_c": profile_c,
        "s": float(row["s"]),
        "q": float(row["q"]),
        "rho": float(row["rho"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "d_over_rho": float(row.get("d_over_rho", float("nan"))),
        "times": times.tolist(),
        "batch_epochs": int(epoch_count),
        "reference": references,
        "reference_uncertainty": uncertainties,
        "reference_available": all(
            _finite(value, bench) for value in references
        ),
        "reference_certified_for_target": certified,
        "reference_source": reference_source,
        "reference_floor": row.get("accuracy_reference_floor"),
        "vbm_nannuli_per_epoch": (
            None if vbm_probe is None else vbm_probe["nannuli"]
        ),
        "vbm_event_n_annuli": (
            None
            if vbm_probe is None
            else int(vbm_probe["selected_n_annuli"])
        ),
        "vbm_probe_values": (
            None if vbm_probe is None else vbm_probe["values"]
        ),
        "vbm_probe_seconds": (
            None if vbm_probe is None else float(vbm_probe["seconds"])
        ),
        "microlux_tol": target,
        "microlux_retol": target,
        "microlux_strategy": list(bench._microlux_strategy(target)),
        "microlux_accuracy_status": selection["status"],
        "microlux_selection_mode": selected_calibration.get(
            "selection_mode", selection["status"]
        ),
        "microlux_fixed_n_annuli": None,
        "microlux_n_annuli": selected_n_annuli,
        "microlux_selected_n_annuli": selected_n_annuli,
        "microlux_warmup_seconds": float(selection["warmup_seconds"]),
        "microlux_values": values.tolist(),
        "microlux_dA_dt": (
            None if derivatives is None else derivatives.tolist()
        ),
        "microlux_relative_errors": errors,
        "microlux_max_relative_error": _finite_max(errors, bench),
        # A derivative reference is not part of the saved VBM corpus.  Do
        # not compare dA/dt to magnification values merely to fill a field.
        "microlux_dA_dt_relative_errors": None,
        "microlux_dA_dt_max_relative_error": None,
        "microlux_passes_reference": passed,
        "microlux_forward_block_seconds": float(forward_block),
        "microlux_forward_seconds_per_epoch": float(
            forward_block / epoch_count
        ),
        "microlux_dA_dt_block_seconds": (
            None if derivative_block is None else float(derivative_block)
        ),
        "microlux_dA_dt_seconds_per_epoch": (
            None
            if derivative_block is None
            else float(derivative_block / epoch_count)
        ),
        "microlux_forward_samples_seconds": [
            float(value) for value in timing["forward_samples_seconds"]
        ],
        "microlux_dA_dt_samples_seconds": (
            None
            if timing["dA_dt_samples_seconds"] is None
            else [
                float(value)
                for value in timing["dA_dt_samples_seconds"]
            ]
        ),
        "microlux_forward_first_after_warmup_seconds": float(
            timing["forward_first_after_warmup_seconds"]
        ),
        "microlux_dA_dt_first_after_warmup_seconds": float(
            timing["dA_dt_first_after_warmup_seconds"]
        ),
        "microlux_dA_dt_timeout": bool(timing["dA_dt_timeout"]),
        "microlux_dA_dt_timeout_message": timing[
            "dA_dt_timeout_message"
        ],
        "microlux_warnings": warnings,
        "microlux_budget_exhausted": bool(
            selected_calibration.get("budget_exhausted", False)
            or timing.get("budget_exhausted", False)
        ),
    }


def _stats(values, bench):
    finite = [float(value) for value in values if _finite(value, bench)]
    if not finite:
        return {"count": 0}
    finite.sort()
    return {
        "count": len(finite),
        "median_seconds": float(finite[len(finite) // 2]),
        "minimum_seconds": float(min(finite)),
        "maximum_seconds": float(max(finite)),
    }


def _summary(results, bench):
    grouped = {}
    for result in results:
        key = f"{result['profile']}:target={float(result['target']):g}"
        grouped.setdefault(key, []).append(result)
    summary = {}
    for key, rows in sorted(grouped.items()):
        completed = [row for row in rows if row.get("status") == "completed"]
        selected_counts = Counter(
            str(row.get("vbm_event_n_annuli"))
            for row in completed
            if row.get("vbm_event_n_annuli") is not None
        )
        epoch_counts = Counter(
            str(value)
            for row in completed
            for value in (row.get("vbm_nannuli_per_epoch") or ())
        )
        summary[key] = {
            "jobs": len(rows),
            "epochs": sum(int(row.get("batch_epochs", 0)) for row in rows),
            "status_counts": {
                status: sum(row.get("status") == status for row in rows)
                for status in sorted({row.get("status") for row in rows})
            },
            "vbm_event_n_annuli_counts": dict(
                sorted(selected_counts.items(), key=lambda item: int(item[0]))
            ),
            "vbm_epoch_nannuli_counts": dict(
                sorted(epoch_counts.items(), key=lambda item: int(item[0]))
            ),
            "vbm_event_n_annuli": _stats(
                [row.get("vbm_event_n_annuli") for row in completed],
                bench,
            ),
            "accuracy_certified_count": sum(
                bool(row.get("reference_certified_for_target"))
                for row in completed
            ),
            "accuracy_pass_count": sum(
                bool(row.get("microlux_passes_reference"))
                for row in completed
            ),
            "accuracy_fail_count": sum(
                row.get("reference_certified_for_target") is True
                and row.get("microlux_passes_reference") is False
                for row in completed
            ),
            "forward": _stats(
                [row.get("microlux_forward_block_seconds") for row in completed],
                bench,
            ),
            "dA_dt": _stats(
                [row.get("microlux_dA_dt_block_seconds") for row in completed],
                bench,
            ),
            "dA_dt_timeout_count": sum(
                bool(row.get("microlux_dA_dt_timeout"))
                for row in completed
            ),
            "budget_exhausted_count": sum(
                bool(row.get("microlux_budget_exhausted"))
                for row in completed
            ),
            "max_relative_error": _stats(
                [row.get("microlux_max_relative_error") for row in completed],
                bench,
            ),
        }
    return summary


def _split_lane_target(value):
    return f"{float(value):.0e}".replace("+", "p").replace("-", "m")


def _merge_payloads(
    payloads,
    args,
    profiles,
    targets,
    bench,
    shim,
    extra_results=(),
    group_metadata=None,
    precompute_seconds=0.0,
):
    if not payloads:
        raise RuntimeError("no lane payloads were produced")
    results = list(extra_results) + [
        result
        for payload in payloads
        for result in payload.get("results", ())
    ]
    merged = dict(payloads[0])
    merged["configuration"] = dict(payloads[0]["configuration"])
    merged["configuration"].update({
        "profiles": list(profiles),
        "targets": list(targets),
        "split_lanes": True,
        "parallel_workers": args.parallel_workers,
        "grouped_by_vbm_event_n_annuli": True,
        "group_count": len(group_metadata or ()),
    })
    merged["results"] = results
    merged["compile_records"] = {
        "microlux": [
            record
            for payload in payloads
            for record in payload.get("compile_records", {}).get(
                "microlux", ()
            )
        ]
    }
    merged["summary"] = _summary(results, bench)
    merged["elapsed_seconds"] = sum(
        float(payload.get("elapsed_seconds", 0.0)) for payload in payloads
    ) + float(precompute_seconds)
    merged["vbm_nannuli_shim"] = str(shim)
    if group_metadata is not None:
        merged["event_groups"] = group_metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(merged["summary"], indent=2), flush=True)
    print(f"saved {args.output}", flush=True)


def _event_group_id(profile, target, n_annuli):
    annuli = "none" if n_annuli is None else str(int(n_annuli))
    return f"{profile}-target-{_split_lane_target(target)}-n-{annuli}"


def _run_event_groups(args):
    """Precompute VBM counts, then isolate one static n_annuli per process.

    XLA retains compiled executables for the lifetime of a Python process.
    Compiling many annulus counts in one process eventually exhausts LLVM
    memory, while compiling the same count in many case chunks duplicates the
    cost.  A group process contains exactly one linear annulus count, so its
    cache is both reusable for all matching events and bounded in lifetime.
    """

    bench = _load_benchmark_module()
    shim = _build_vbm_shim(args.vbm_shim)
    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    input_payload, source_rows = bench._load_rows(args.input)
    rows = bench._select_rows(
        source_rows,
        args.profiles,
        args.targets,
        args.case_id_min,
        args.case_id_max,
        args.max_jobs,
    )
    if not rows:
        raise SystemExit("no rows selected")

    probe = VBMAnnuliProbe(shim)
    grouped = {}
    manifest_rows = {}
    probe_errors = []
    precompute_started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        key = _row_key(row)
        profile = str(row["profile"])
        target = float(row["target"])
        if profile == "linear":
            try:
                with bench._deadline(args.vbm_timeout):
                    record = probe.measure(
                        row,
                        bench._times(row),
                        target,
                        float(row.get("limb_darkening_c", 0.0)),
                    )
            except Exception as error:  # noqa: BLE001
                probe_errors.append(
                    _error_record(
                        row,
                        profile,
                        target,
                        bench._times(row),
                        "vbm_nannuli",
                        error,
                    )
                )
                continue
            manifest_rows[key] = record
            n_annuli = int(record["selected_n_annuli"])
        else:
            n_annuli = None
        group_id = _event_group_id(profile, target, n_annuli)
        group = grouped.setdefault(
            group_id,
            {
                "profile": profile,
                "target": target,
                "n_annuli": n_annuli,
                "row_keys": [],
            },
        )
        group["row_keys"].append(key)
        if index == 1 or index % 200 == 0:
            print(
                f"VBM event annuli precompute [{index}/{len(rows)}]",
                flush=True,
            )
    precompute_seconds = time.perf_counter() - precompute_started
    if not grouped:
        raise SystemExit("VBM annulus precomputation produced no runnable groups")

    group_metadata = [
        {
            "group_id": group_id,
            "profile": group["profile"],
            "target": group["target"],
            "n_annuli": group["n_annuli"],
            "jobs": len(group["row_keys"]),
        }
        for group_id, group in sorted(grouped.items())
    ]
    print(
        f"VBM precompute complete: {len(rows)} rows, "
        f"{len(grouped)} static groups, {precompute_seconds:.3f}s",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="microlux-vbm-event-groups-") as temp_dir:
        temp = Path(temp_dir)
        manifest = {
            "input": str(args.input),
            "rows": manifest_rows,
            "groups": grouped,
        }
        manifest_path = temp / "event_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        tasks = []
        for group_id, group in sorted(grouped.items()):
            part = temp / f"{group_id}.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(args.input),
                "--output",
                str(part),
                "--profiles",
                str(group["profile"]),
                "--targets",
                str(group["target"]),
                "--vbm-shim",
                str(shim),
                "--event-manifest",
                str(manifest_path),
                "--event-group-id",
                group_id,
                "--repeats",
                str(args.repeats),
                "--forward-timeout",
                str(args.forward_timeout),
                "--derivative-timeout",
                str(args.derivative_timeout),
            ]
            tasks.append((group_id, command, part))

        def run_group(group_index, task):
            group_id, command, part = task
            group = grouped[group_id]
            print(
                f"run group [{group_index}/{len(tasks)}] {group_id}: "
                f"{len(group['row_keys'])} jobs",
                flush=True,
            )
            subprocess.run(command, check=True)
            return json.loads(part.read_text())

        if args.parallel_workers == 1:
            payloads = [
                run_group(group_index, task)
                for group_index, task in enumerate(tasks, 1)
            ]
        else:
            worker_count = min(args.parallel_workers, len(tasks))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                payloads = list(
                    pool.map(
                        run_group,
                        range(1, len(tasks) + 1),
                        tasks,
                    )
                )

    _merge_payloads(
        payloads,
        args,
        profiles,
        targets,
        bench,
        shim,
        extra_results=probe_errors,
        group_metadata=group_metadata,
        precompute_seconds=precompute_seconds,
    )


def _run_split_lanes(args):
    bench = _load_benchmark_module()
    shim = _build_vbm_shim(args.vbm_shim)
    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    _, source_rows = bench._load_rows(args.input)

    def case_chunks(profile, target):
        selected = bench._select_rows(
            source_rows,
            (profile,),
            (target,),
            args.case_id_min,
            args.case_id_max,
            None,
        )
        case_ids = sorted({int(row["case_id"]) for row in selected})
        count = min(args.parallel_workers, len(case_ids))
        chunks = []
        for index in range(count):
            start = (index * len(case_ids)) // count
            stop = ((index + 1) * len(case_ids)) // count
            if start != stop:
                chunks.append((case_ids[start], case_ids[stop - 1] + 1))
        return tuple(chunks)

    with tempfile.TemporaryDirectory(prefix="microlux-vbm-annuli-") as temp_dir:
        temp = Path(temp_dir)
        tasks = []
        for profile in profiles:
            for target in targets:
                for chunk_index, (case_min, case_max) in enumerate(
                    case_chunks(profile, target)
                ):
                    part = temp / (
                        f"{profile}-target-{_split_lane_target(target)}-"
                        f"chunk-{chunk_index:02d}.json"
                    )
                    command = [
                        sys.executable,
                        str(SCRIPT),
                        "--input",
                        str(args.input),
                        "--output",
                        str(part),
                        "--profiles",
                        profile,
                        "--targets",
                        str(target),
                        "--vbm-shim",
                        str(shim),
                        "--repeats",
                        str(args.repeats),
                        "--vbm-timeout",
                        str(args.vbm_timeout),
                        "--forward-timeout",
                        str(args.forward_timeout),
                        "--derivative-timeout",
                        str(args.derivative_timeout),
                        "--case-id-min",
                        str(case_min),
                        "--case-id-max",
                        str(case_max),
                    ]
                    tasks.append((command, part))

        def run_task(task):
            subprocess.run(task[0], check=True)
            return task

        if args.parallel_workers == 1:
            completed_tasks = [run_task(task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
                completed_tasks = list(pool.map(run_task, tasks))
        payloads = [json.loads(task[1].read_text()) for task in completed_tasks]
    _merge_payloads(payloads, args, profiles, targets, bench, shim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("uniform", "linear"))
    parser.add_argument(
        "--targets", nargs="+", type=float, default=(1.0e-3, 1.0e-4)
    )
    parser.add_argument("--case-id-min", type=int)
    parser.add_argument("--case-id-max", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--vbm-shim", type=Path, default=DEFAULT_SHIM)
    parser.add_argument("--vbm-timeout", type=float, default=300.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--forward-timeout", type=float, default=300.0)
    parser.add_argument("--derivative-timeout", type=float, default=300.0)
    parser.add_argument("--split-lanes", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--event-manifest", type=Path)
    parser.add_argument("--event-group-id")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.vbm_timeout <= 0:
        raise SystemExit("--vbm-timeout must be positive")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be positive")

    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    if args.split_lanes:
        _run_event_groups(args)
        return

    bench = _load_benchmark_module()
    shim = _build_vbm_shim(args.vbm_shim)
    input_payload, source_rows = bench._load_rows(args.input)
    rows = bench._select_rows(
        source_rows,
        args.profiles,
        args.targets,
        args.case_id_min,
        args.case_id_max,
        args.max_jobs,
    )
    event_manifest = None
    event_records = {}
    if args.event_manifest is not None:
        if args.event_group_id is None:
            raise SystemExit("--event-group-id is required with --event-manifest")
        event_manifest = json.loads(args.event_manifest.read_text())
        group = event_manifest.get("groups", {}).get(args.event_group_id)
        if group is None:
            raise SystemExit(
                f"event group not found: {args.event_group_id}"
            )
        allowed = set(group.get("row_keys", ()))
        rows = [row for row in rows if _row_key(row) in allowed]
        event_records = event_manifest.get("rows", {})
    if not rows:
        raise SystemExit("no rows selected")

    executor = bench.MicroLuxBatchExecutor(
        max_annuli=VBM_DEFAULT_MAX_ANNULI,
    )
    probe = None if event_manifest is not None else VBMAnnuliProbe(shim)
    print(
        f"selected {len(rows)} jobs, "
        f"{len(rows) * len(bench.REFERENCE_INDICES)} batched epochs; "
        "microLUX linear n_annuli=max(VBM nannuli) per event, "
        f"repeats={args.repeats}",
        flush=True,
    )
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        result = _row_result(
            row,
            executor,
            probe,
            bench,
            args,
            event_record=event_records.get(_row_key(row)),
        )
        results.append(result)
        if index == 1 or index % 25 == 0 or result["status"] != "completed":
            if result["status"] == "completed":
                detail = (
                    f"case={result['case_id']} "
                    f"vbm_n={result['vbm_event_n_annuli']} "
                    f"forward={result['microlux_forward_block_seconds']:.6g}s "
                    f"dA/dt={result['microlux_dA_dt_block_seconds']} "
                    f"pass={result['microlux_passes_reference']}"
                )
            else:
                detail = result.get("error", "")
            print(f"[{index}/{len(rows)}] {result['status']} {detail}", flush=True)

    try:
        import microlux

        microlux_path = str(Path(microlux.__file__).resolve())
        microlux_commit = bench._checkout(microlux)
    except Exception:  # noqa: BLE001
        microlux_path = ""
        microlux_commit = ""

    payload = {
        "input": str(args.input),
        "timing_mode": "compiled_warm_microLUX_vbm_event_annuli_only",
        "derivative_mode": "source_trajectory_dA_dt_forward_mode_jvp",
        "reference_policy": (
            "reuse the saved VBM corpus reference and uncertainty; no JAX or "
            "native call is rerun"
        ),
        "configuration": {
            "profiles": list(profiles),
            "targets": list(targets),
            "reference_indices": list(bench.REFERENCE_INDICES),
            "batch_epochs": len(bench.REFERENCE_INDICES),
            "vbm_absolute_floor": VBM_ABSOLUTE_FLOOR,
            "vbm_reltol_policy": "RelTol=target per event",
            "vbm_nannuli_policy": (
                "run VBM BinaryMagDark at all four event epochs and use the "
                "maximum final nannuli as one static microLUX n_annuli"
            ),
            "microlux_n_annuli_policy": "event-specific max VBM nannuli",
            "microlux_tol_policy": "tol=target",
            "microlux_retol_policy": "retol=target",
            "microlux_strategy_policy": (
                "library default strategy at 1e-3; "
                "(60,60,120,240,480) for tighter targets"
            ),
            "vbm_timeout_seconds": args.vbm_timeout,
            "forward_timeout_seconds": args.forward_timeout,
            "derivative_timeout_seconds": args.derivative_timeout,
            "repeats": args.repeats,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "split_lanes": False,
            "parallel_workers": args.parallel_workers,
            "event_group_id": args.event_group_id,
            "event_grouped_static_n_annuli": event_manifest is not None,
        },
        "input_metadata": {
            "source_payload_keys": sorted(input_payload),
            "reference_indices": list(bench.REFERENCE_INDICES),
            "reference_epoch_count": len(bench.REFERENCE_INDICES),
            "nominal_epochs": len(rows) * len(bench.REFERENCE_INDICES),
        },
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "vbm_nannuli_shim": str(shim),
            "microlux_path": microlux_path,
            "microlux_commit": microlux_commit,
            "event_manifest": (
                None
                if args.event_manifest is None
                else str(args.event_manifest)
            ),
        },
        "results": results,
        "compile_records": {
            "microlux": list(executor.compile_records.values())
        },
        "summary": _summary(results, bench),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
