#!/usr/bin/env python3
"""Reproduce the VBM 5.5 dispatch and time the saved low-magnification sample."""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import VBMicrolensing as vbml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAPSHOT = ROOT / "input_snapshot.tsv"
SELECTION = HERE / "selection.tsv"
JOINED = ROOT / "joined_v2_vbm.tsv"
SHORTCUT_AUDIT = HERE / "vbm_uniform_shortcut_audit.tsv"
TIMING_AUDIT = HERE / "vbm_low_magnification_timing.tsv"


def read_snapshot():
    fields = [
        "case_id", "configuration_id", "profile", "d_bin_index", "epoch_index",
        "s", "q", "rho", "x", "y", "time", "u", "X", "reference_1e-6",
    ]
    with SNAPSHOT.open() as stream:
        return [dict(zip(fields, line.split())) for line in stream
                if line.strip() and not line.startswith("#")]


def read_tsv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def key(row):
    return (row["case_id"], row["profile"], row["d_bin_index"], row["epoch_index"])


def new_vbm(profile):
    v = vbml.VBMicrolensing()
    v.Tol = 1e-3
    v.RelTol = 1e-3
    if profile == "linear":
        v.a1 = 0.5
        v.SetLDprofile(v.LDlinear)
    return v


def method_call(v, row):
    s, q, rho = map(float, (row["s"], row["q"], row["rho"]))
    source_x, source_y = -float(row["time"]), float(row["y"])
    if row["profile"] == "uniform":
        return v.BinaryMag2(s, q, source_x, source_y, rho)
    return v.BinaryMagDark(s, q, source_x, source_y, rho, 1e-3)


def main():
    if vbml.__version__ != "5.5":
        raise RuntimeError(f"expected installed VBMicrolensing 5.5, got {vbml.__version__}")
    snapshot = read_snapshot()
    uniform = [row for row in snapshot if row["profile"] == "uniform"]
    v = new_vbm("uniform")
    with SHORTCUT_AUDIT.open("w", newline="") as stream:
        fields = ["case_id", "d_bin_index", "epoch_index", "reference_1e-6",
                  "rho", "point_mu", "multipole_gate", "binarymag2_mu",
                  "exact_point_return"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        gate_count = 0
        exact_count = 0
        for row in uniform:
            s, q, rho = map(float, (row["s"], row["q"], row["rho"]))
            sx, sy = -float(row["time"]), float(row["y"])
            point = v.BinaryMag0(s, q, sx, sy)
            gate = (
                v.corrquad * 6.0 * (rho * rho + 1e-4 * v.Tol) < v.Tol
                and v.corrquad2 * 256.0 * (rho * rho + 1e-8) < 1.0
                and v.safedist > 4.0 * rho * rho
            )
            result = v.BinaryMag2(s, q, sx, sy, rho)
            gate_count += bool(gate)
            exact_count += result == point
            writer.writerow({
                "case_id": row["case_id"], "d_bin_index": row["d_bin_index"],
                "epoch_index": row["epoch_index"], "reference_1e-6": row["reference_1e-6"],
                "rho": row["rho"], "point_mu": point, "multipole_gate": int(gate),
                "binarymag2_mu": result, "exact_point_return": int(result == point),
            })

    selected = read_tsv(SELECTION)
    joined_by_key = {key(row): row for row in read_tsv(JOINED)
                     if abs(float(row["target"]) - 1e-3) < 1e-15}
    vbm_by_profile = {profile: new_vbm(profile) for profile in ("uniform", "linear")}
    snapshot_by_key = {key(row): row for row in snapshot}
    with TIMING_AUDIT.open("w", newline="") as stream:
        fields = ["case_id", "profile", "d_bin_index", "epoch_index", "A_ref",
                  "saved_vbm_ms", "fresh_vbm_ms_p50", "fresh_over_saved"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        per_profile = {"uniform": [], "linear": []}
        for picked in selected:
            if float(picked["A_ref"]) >= 1.1:
                continue
            row = snapshot_by_key[key(picked)]
            v = vbm_by_profile[row["profile"]]
            method_call(v, row)  # warm the per-instance path
            samples = []
            for _ in range(7):
                start = time.perf_counter_ns()
                method_call(v, row)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            fresh = statistics.median(samples)
            saved = float(joined_by_key[key(row)]["vbm_ms"])
            per_profile[row["profile"]].append((saved, fresh))
            writer.writerow({
                "case_id": row["case_id"], "profile": row["profile"],
                "d_bin_index": row["d_bin_index"], "epoch_index": row["epoch_index"],
                "A_ref": row["reference_1e-6"], "saved_vbm_ms": saved,
                "fresh_vbm_ms_p50": fresh, "fresh_over_saved": fresh / saved,
            })

    print(f"VBMicrolensing={vbml.__version__}")
    print(f"uniform_rows={len(uniform)} multipole_gate={gate_count} exact_point_return={exact_count}")
    for profile, pairs in per_profile.items():
        saved = statistics.median(pair[0] for pair in pairs)
        fresh = statistics.median(pair[1] for pair in pairs)
        print(f"{profile} low_A_rows={len(pairs)} saved_ms_p50={saved:.9g} fresh_ms_p50={fresh:.9g} fresh/saved={fresh/saved:.4f}")
    print(f"wrote {SHORTCUT_AUDIT} and {TIMING_AUDIT}")


if __name__ == "__main__":
    main()
