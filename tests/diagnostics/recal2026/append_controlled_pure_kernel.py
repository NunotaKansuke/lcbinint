#!/usr/bin/env python3
"""Combine a base pure-kernel corpus with a disjoint generated extension.

The timing results for the base corpus can then be retained while only the
new case IDs are measured.  Both inputs use the one-row-per-block corpus
format produced by ``generate_controlled_pure_kernel.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_corpus(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = json.loads((path / "rows.json").read_text())["rows"]
    if not rows:
        raise ValueError(f"empty corpus: {path}")
    return manifest, rows


def _case_ids(rows: list[dict], label: str) -> set[int]:
    ids = {int(row["case_id"]) for row in rows}
    if not ids:
        raise ValueError(f"no case IDs in {label}")
    return ids


def _validate_rows(rows: list[dict], label: str) -> set[int]:
    ids = _case_ids(rows, label)
    keys = [
        (int(row["case_id"]), str(row["profile"]), int(row["d_bin_index"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate case/profile/d-bin rows in {label}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")

    base_manifest, base_rows = _read_corpus(args.base)
    extension_manifest, extension_rows = _read_corpus(args.extension)
    base_ids = _validate_rows(base_rows, "base corpus")
    extension_ids = _validate_rows(extension_rows, "extension corpus")
    overlap = base_ids & extension_ids
    if overlap:
        raise ValueError(f"overlapping case IDs: {sorted(overlap)[:8]}")

    rows = sorted(
        [*base_rows, *extension_rows],
        key=lambda row: (
            int(row["case_id"]), int(row["d_bin_index"]), str(row["profile"])
        ),
    )
    configs = sorted(
        [
            *base_manifest.get("configurations", ()),
            *extension_manifest.get("configurations", ()),
        ],
        key=lambda config: int(config["configuration_id"]),
    )
    config_ids = [int(config["configuration_id"]) for config in configs]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("duplicate configuration IDs")
    if config_ids != sorted(base_ids | extension_ids):
        raise ValueError("configuration IDs do not match corpus case IDs")

    manifest = dict(base_manifest)
    manifest.update({
        "generator": "append_controlled_pure_kernel.py",
        "cases": len(configs),
        "case_id_start": min(config_ids),
        "case_id_end": max(config_ids) + 1,
        "configurations": configs,
        "composition": {
            "base": {
                "path": str(args.base),
                "seed": base_manifest.get("seed"),
                "cases": base_manifest.get("cases"),
                "case_id_start": base_manifest.get("case_id_start", 0),
                "case_id_end": base_manifest.get(
                    "case_id_end", base_manifest.get("cases")
                ),
            },
            "extension": {
                "path": str(args.extension),
                "seed": extension_manifest.get("seed"),
                "cases": extension_manifest.get("cases"),
                "case_id_start": extension_manifest.get("case_id_start", 0),
                "case_id_end": extension_manifest.get(
                    "case_id_end", extension_manifest.get("cases")
                ),
            },
        },
    })

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "rows.json").write_text(json.dumps({"rows": rows}, indent=2))
    for index, row in enumerate(rows):
        (args.output / f"block-{index:05d}.json").write_text(
            json.dumps({"rows": [row]}, indent=2)
        )
    print(json.dumps({
        "output": str(args.output),
        "cases": len(configs),
        "rows": len(rows),
        "base_cases": len(base_ids),
        "extension_cases": len(extension_ids),
        "case_id_start": min(config_ids),
        "case_id_end": max(config_ids) + 1,
    }, indent=2))


if __name__ == "__main__":
    main()
