#!/usr/bin/env python3
"""Rebuild the synthetic-light-curve result after the JAX route fix.

The existing result contains the paper-facing native, VBM, and microLUX
measurements.  This script deliberately reuses those lanes and reruns only
the corrected JAX warm-up workers, so the warm denominator and every derived
microLUX/JAX ratio are based on the same corrected implementation.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = Path(__file__).with_name("benchmark_synthetic_lightcurve_jax_microlux.py")
SOURCE_DIR = ROOT / "tests/diagnostics/results/recal2026/synthetic_lightcurve_jax_vjp_20260820"
SOURCE_JSON = SOURCE_DIR / "results.json"
OUTPUT_DIR = ROOT / "tests/diagnostics/results/recal2026/synthetic_lightcurve_jax_warmupfix_20260820"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "benchmark_synthetic_lightcurve_jax_microlux", HARNESS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_one(harness, output_dir, cache, source_record, case):
    profile_c = float(source_record["profile_c"])
    tag = f"corrected_{source_record['case']}_{source_record['profile']}"
    worker = harness._run_worker_subprocess(
        output_dir,
        case,
        profile_c,
        source_record["times"],
        source_record.get("vbm_nannuli_per_epoch"),
        cache,
        "warmup",
        False,
        tag=tag,
        measure_micro=False,
    )
    return source_record["case"], source_record["profile"], worker


def main():
    if not SOURCE_JSON.is_file():
        raise FileNotFoundError(SOURCE_JSON)

    harness = _load_harness()
    # Load the in-tree build once in the parent as an early ABI/API check.  The
    # actual JAX processes load it again through the harness worker entrypoint.
    harness._load_benchmark_backend()

    source_payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    source_records = source_payload["records"]
    if len(source_records) != 12:
        raise RuntimeError(f"expected 12 source records, found {len(source_records)}")

    case_by_name = {case["name"]: case for case in harness.CASES}
    missing = sorted({record["case"] for record in source_records} - set(case_by_name))
    if missing:
        raise RuntimeError(f"source records contain unknown cases: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT_DIR / "jax_cache"
    jobs = [
        (record, case_by_name[record["case"]])
        for record in source_records
    ]
    workers = {}
    max_workers = min(4, len(jobs))
    print(f"rerunning corrected JAX warm-up lanes: {len(jobs)} jobs, {max_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, harness, OUTPUT_DIR, cache, record, case):
            (record["case"], record["profile"])
            for record, case in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            case_name, profile, worker = future.result()
            workers[key] = worker
            status = worker.get("status")
            if status != "completed":
                print(f"FAILED {case_name}/{profile}: {worker.get('error')}", flush=True)
            else:
                lane = worker["jax"]
                print(
                    f"completed {case_name}/{profile}: "
                    f"A={lane['value']['steady_seconds']:.9g}s "
                    f"TJ={lane['gradient']['steady_seconds']:.9g}s "
                    f"VJP={lane['vjp']['steady_seconds']:.9g}s",
                    flush=True,
                )

    failed = [key for key, worker in workers.items() if worker.get("status") != "completed"]
    if failed:
        raise RuntimeError(f"corrected warm-up workers failed: {failed}")
    if len(workers) != len(jobs):
        raise RuntimeError(f"missing worker results: {len(workers)} / {len(jobs)}")

    payload = copy.deepcopy(source_payload)
    payload["benchmark"] = "synthetic_lightcurve_jax_gradient_microlux_warmupfix"
    payload["timing_policy"] = (
        "JAX no-warmup values and native/microLUX values are reused from the "
        f"source result {SOURCE_JSON}; only the JAX warm-up lane was rerun after "
        "the route-selection fix. The corrected warm-up uses the JAX automatic "
        "route/bin proposal plus native self-converged reference certification, "
        "then fixed-plan value/Jacobian/VJP compilation; steady values are medians "
        "of repeated calls."
    )
    payload["correction"] = {
        "route_selection_fix": (
            "JAX warm-up obtains its initial method split and nbin proposal from "
            "the JAX automatic dispatcher; the native self-converged result is "
            "used for certification, not for selecting a JAX route."
        ),
        "jax_warmup_rows_rerun": len(jobs),
        "microlux_rows_rerun": 0,
        "native_rows_rerun": 0,
        "microLUX_reused_unchanged": True,
        "source_result": str(SOURCE_JSON),
        "old_warmup_values_reused": False,
    }

    for record in payload["records"]:
        key = (record["case"], record["profile"])
        worker = workers[key]
        corrected_lane = worker["jax"]
        values = corrected_lane["value"]["values"]
        corrected_record_lane = {
            **corrected_lane,
            "accuracy_vs_vbm": harness._accuracy(record["vbm_values"], values),
        }
        record["jax_warmup"] = corrected_record_lane
        # Keep the historical compatibility alias synchronized with the final
        # warm lane; reports use the explicit jax_warmup key.
        record["jax"] = copy.deepcopy(corrected_record_lane)
        record.setdefault("workers", {})["jax_warmup"] = worker

    result_path = OUTPUT_DIR / "results.json"
    result_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    harness._write_report_v2(OUTPUT_DIR / "REPORT.md", payload)
    harness._write_plots(OUTPUT_DIR, payload["records"])
    print(f"report: {OUTPUT_DIR / 'REPORT.md'}")
    print(f"json: {result_path}")


if __name__ == "__main__":
    main()
