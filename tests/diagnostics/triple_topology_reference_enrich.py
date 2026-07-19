#!/usr/bin/env python3
"""Add independent high-order chord references to topology calibration rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from triple_topology_calibration import atomic_json, timed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(path.read_text())
            for path in sorted(args.input_dir.glob("sample-*.json"))]
    rows = [row for row in rows
            if row["auto"].get("method") == "source_plane_quadrature"]
    rows = [row for index, row in enumerate(rows)
            if index % args.shard_count == args.shard_index]
    for index, frozen in enumerate(rows):
        destination = args.output_dir / f"sample-{args.shard_index:02d}-{index:05d}.json"
        if destination.exists():
            continue
        output = dict(frozen)
        if "value" not in output.get("chord_160", {}):
            output["chord_160"] = timed(
                output["case"], output["sample"], "chord-160", args.timeout)
        if "value" not in output.get("chord_256", {}):
            output["chord_256"] = timed(
                output["case"], output["sample"], "chord-256", args.timeout)
        atomic_json(destination, output)
        print(f"shard={args.shard_index}/{args.shard_count} "
              f"sample={index + 1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
