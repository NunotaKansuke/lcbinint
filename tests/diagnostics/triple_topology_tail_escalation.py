#!/usr/bin/env python3
"""Escalate non-converged triple chord references to orders 400 and 512."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from triple_topology_calibration import atomic_json, timed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_dir.glob("sample-*.json")):
        row = json.loads(path.read_text())
        coarse = row.get("chord_160", {})
        fine = row.get("chord_256", {})
        if "value" not in coarse or "value" not in fine:
            continue
        tolerance = 1.0e-4 + 1.0e-3 * max(abs(fine["value"]), 1.0)
        if abs(coarse["value"] - fine["value"]) > tolerance:
            rows.append(row)
    rows = [row for index, row in enumerate(rows)
            if index % args.shard_count == args.shard_index]
    for index, frozen in enumerate(rows):
        destination = args.output_dir / f"sample-{args.shard_index:02d}-{index:05d}.json"
        if destination.exists():
            continue
        output = dict(frozen)
        output["chord_400"] = timed(
            output["case"], output["sample"], "chord-400", args.timeout)
        output["chord_512"] = timed(
            output["case"], output["sample"], "chord-512", args.timeout)
        atomic_json(destination, output)
        print(f"shard={args.shard_index}/{args.shard_count} "
              f"sample={index + 1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
