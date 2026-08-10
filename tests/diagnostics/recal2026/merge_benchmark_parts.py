#!/usr/bin/env python3
"""Merge independent benchmark part results with duplicate protection."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from bench_grid_vs_vbm_dark import summarise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.parts))
    if not paths:
        raise SystemExit("no part results found")
    payloads = [json.loads(path.read_text()) for path in paths]
    results = []
    keys = set()
    for payload in payloads:
        for result in payload["results"]:
            key = (
                int(result["case_id"]),
                result["profile"],
                round(float(result["d_over_rho"]), 12),
                round(float(result["target"]), 12),
            )
            if key in keys:
                raise SystemExit(f"duplicate result key: {key}")
            keys.add(key)
            results.append(result)

    first = payloads[0]
    targets = sorted({float(result["target"]) for result in results})
    factors = sorted({float(result["d_over_rho"]) for result in results})
    merged = {
        "input": first.get("input"),
        "case_count": first.get("case_count"),
        "factors": factors,
        "seed": first.get("seed"),
        "repeats": first.get("repeats"),
        "search_missing": first.get("search_missing"),
        "point_timeout": first.get("point_timeout"),
        "job_timeout": first.get("job_timeout"),
        "route_filter": first.get("route_filter"),
        "timing_mode": first.get("timing_mode"),
        "build_extension": first.get("build_extension"),
        "targets": targets,
        "profiles": sorted({result["profile"] for result in results}),
        "filters": first.get("filters", {}),
        "reference_indices": first.get("reference_indices"),
        "part_results": [str(path) for path in paths],
        "results": results,
        "summary": summarise(results, targets),
        "elapsed_seconds": sum(
            float(payload.get("elapsed_seconds", 0.0)) for payload in payloads
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(merged, indent=2))
    print(json.dumps({
        "parts": len(paths),
        "results": len(results),
        "targets": targets,
        "summary": merged["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
