#!/usr/bin/env python3
"""Select a bounded, stratified tolerance-sweep subset from the paper corpus."""

from __future__ import annotations

import csv
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JOINED = ROOT / "joined_v2_vbm.tsv"
SNAPSHOT = ROOT / "input_snapshot.tsv"
OUTPUT = HERE / "probe_input.tsv"
SELECTION = HERE / "selection.tsv"


def read_tsv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t"))


def read_snapshot(path: Path):
    fields = [
        "case_id", "configuration_id", "profile", "d_bin_index", "epoch_index",
        "s", "q", "rho", "x", "y", "time", "u", "X", "reference_1e-6",
    ]
    with path.open() as stream:
        return [dict(zip(fields, line.split())) for line in stream
                if line.strip() and not line.startswith("#")]


def key(row):
    return (row["case_id"], row["profile"], row["d_bin_index"], row["epoch_index"])


def mag_bin(value: float) -> str:
    if value < 1.1:
        return "A_lt_1p1"
    if value < 1.5:
        return "A_1p1_to_1p5"
    if value < 5.0:
        return "A_1p5_to_5"
    return "A_ge_5"


def main():
    joined = read_tsv(JOINED)
    snapshot = read_snapshot(SNAPSHOT)
    snapshot_by_key = {key(row): row for row in snapshot}
    eligible = [
        row for row in joined
        if abs(float(row["target"]) - 1e-3) < 1e-15 and row["v2_status"] == "OK"
    ]

    rng = random.Random(20261013)
    selected: dict[tuple[str, str, str, str], tuple[dict, str]] = {}
    quotas = {"A_lt_1p1": 60, "A_1p1_to_1p5": 20, "A_1p5_to_5": 20, "A_ge_5": 20}
    for profile in ("uniform", "linear"):
        profile_rows = [row for row in eligible if row["profile"] == profile]
        for label, quota in quotas.items():
            bucket = [row for row in profile_rows if mag_bin(float(row["reference_1e-6"])) == label]
            bucket.sort(key=key)
            picked = bucket if len(bucket) <= quota else rng.sample(bucket, quota)
            for row in picked:
                selected[key(row)] = (row, f"stratified:{label}")

        worst = sorted(
            profile_rows,
            key=lambda row: float(row["relative_error_v2_over_vbm_1e-6"]),
            reverse=True,
        )[:10]
        for row in worst:
            selected.setdefault(key(row), (row, "top_current_error"))

        for case_id, epoch in ((26, 15), (35, 15), (77, 23), (95, 23), (98, 23)):
            prior = [row for row in profile_rows
                     if int(row["case_id"]) == case_id and int(row["epoch_index"]) == epoch]
            for row in prior:
                selected.setdefault(key(row), (row, "former_endpoint_outlier"))

    selected_rows = sorted(selected.values(), key=lambda item: key(item[0]))
    input_fields = [
        "case_id", "configuration_id", "profile", "d_bin_index", "epoch_index",
        "s", "q", "rho", "x", "y", "time", "u", "X", "reference_1e-6",
    ]
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(input_fields)
        for row, _reason in selected_rows:
            source = snapshot_by_key[key(row)]
            writer.writerow([source[name] for name in input_fields])

    with SELECTION.open("w", newline="") as stream:
        fields = ["case_id", "profile", "d_bin_index", "epoch_index", "A_ref",
                  "current_relative_error", "selection_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row, reason in selected_rows:
            writer.writerow({
                "case_id": row["case_id"], "profile": row["profile"],
                "d_bin_index": row["d_bin_index"], "epoch_index": row["epoch_index"],
                "A_ref": row["reference_1e-6"],
                "current_relative_error": row["relative_error_v2_over_vbm_1e-6"],
                "selection_reason": reason,
            })

    print(f"selected={len(selected_rows)} input={OUTPUT}")
    for profile in ("uniform", "linear"):
        for label in quotas:
            count = sum(1 for row, _ in selected_rows
                        if row["profile"] == profile and
                        mag_bin(float(row["reference_1e-6"])) == label)
            print(f"{profile}\t{label}\t{count}")


if __name__ == "__main__":
    main()
