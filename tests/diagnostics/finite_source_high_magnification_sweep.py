#!/usr/bin/env python3
"""Long-timeout convergence sweep for difficult/high-magnification samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from finite_source_calibration_sweep import LensCase, _timed_lc


DEFAULT_BINS = (64, 100, 160, 256, 400)


def is_difficult(row: dict[str, Any], minimum_magnification: float) -> bool:
    point_magnification = row.get("lc_auto", {}).get("point_magnification")
    if point_magnification is None or abs(float(point_magnification)) >= minimum_magnification:
        return True
    if row.get("lc_auto", {}).get("timeout") or "error" in row.get("lc_auto", {}):
        return True
    return any(
        result.get("timeout")
        or result.get("skipped_after_timeout")
        or "error" in result
        for sequence in row.get("lc_fixed_sequences", {}).values()
        for result in sequence
    )


def samples(input_dir: Path, minimum_magnification: float):
    for path in sorted(input_dir.glob("case-*.json")):
        document = json.loads(path.read_text())
        if "error" in document:
            continue
        case = LensCase(**document["case"])
        for row in document.get("rows", []):
            if is_difficult(row, minimum_magnification):
                yield case, row


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(samples(args.input_dir, args.minimum_magnification))
    assigned = [
        item for index, item in enumerate(selected)
        if index % args.shard_count == args.shard_index
    ]
    completed = 0
    started = time.monotonic()
    for case, row in assigned:
        limb_tag = str(row["limb_c"]).replace(".", "p")
        destination = args.output_dir / (
            f"sample-{case.case_id:06d}-{int(row['point_id']):03d}-{limb_tag}.json"
        )
        if destination.exists() and not args.overwrite:
            completed += 1
            continue
        result: dict[str, Any] = {
            "schema_version": 1,
            "case": case.__dict__,
            "point_id": int(row["point_id"]),
            "source_x": float(row["source_x"]),
            "source_y": float(row["source_y"]),
            "limb_c": float(row["limb_c"]),
            "caustic_distance_over_rho": float(row["caustic_distance_over_rho"]),
            "original_lc_auto": row.get("lc_auto", {}),
            "sequences": {},
        }
        sample_started = time.monotonic()
        for grid in args.grids:
            sequence = []
            lower_timeout_bins: int | None = None
            for bins in args.bins:
                if lower_timeout_bins is None:
                    evaluation = _timed_lc(
                        grid,
                        bins,
                        args.caustic_bins,
                        case,
                        result["source_x"],
                        result["source_y"],
                        result["limb_c"],
                        False,
                        args.timeout,
                    )
                    if evaluation.get("timeout"):
                        lower_timeout_bins = bins
                else:
                    evaluation = {
                        "skipped_after_timeout": True,
                        "lower_timeout_bins": lower_timeout_bins,
                    }
                evaluation["bins"] = bins
                sequence.append(evaluation)
            result["sequences"][grid] = sequence
        result["sample_elapsed_seconds"] = time.monotonic() - sample_started
        _atomic_json(destination, result)
        completed += 1
        if completed % args.progress_every == 0 or completed == len(assigned):
            print(
                f"shard={args.shard_index}/{args.shard_count} "
                f"samples={completed}/{len(assigned)} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=lambda text: tuple(int(x) for x in text.split(",")), default=DEFAULT_BINS)
    parser.add_argument("--grids", type=lambda text: tuple(text.split(",")), default=("cartesian", "polar"))
    parser.add_argument("--minimum-magnification", type=float, default=1000.0)
    parser.add_argument("--caustic-bins", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if any(grid not in {"cartesian", "polar"} for grid in args.grids):
        parser.error("grids must be cartesian and/or polar")
    if not args.bins or any(value <= 0 for value in args.bins):
        parser.error("bins must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
