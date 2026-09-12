#!/usr/bin/env python3
"""Summarize opt-in D14 qf-warm per-root traces and root-role matches."""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACE_FILES = [
    ROOT / (f"case{case}_trace.log.gz" if (ROOT / f"case{case}_trace.log.gz").exists()
            else f"case{case}_trace.log")
    for case in (149, 0, 64, 9)
]


def fields(line: str) -> tuple[str, dict[str, str]]:
    parts = line.rstrip("\n").split("\t")
    out: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return parts[0], out


def dval(row: dict[str, str], key: str) -> Decimal:
    return Decimal(row[key])


def cabs(re: Decimal, im: Decimal) -> Decimal:
    return (re * re + im * im).sqrt()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(k, "") for k in
                 ("case", "config", "profile", "d_bin", "epoch", "rep", "lane"))


def parse_traces() -> list[dict]:
    calls: list[dict] = []
    row: dict[str, str] = {}
    current: dict | None = None
    last_warm: dict | None = None
    for path in TRACE_FILES:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                if line.startswith("D14TRACE_ROW\t"):
                    _, row = fields(line)
                    current = None
                    last_warm = None
                elif line.startswith("D14QF_BEGIN\t"):
                    _, f = fields(line)
                    current = None
                    if f.get("seed") == "warm":
                        current = {
                            "row": dict(row), "begin": f,
                            "steps": defaultdict(dict), "pairs": {},
                            "roots": {}, "stage": {}, "trace": path.name,
                        }
                        calls.append(current)
                        last_warm = current
                elif line.startswith("D14QF_ROOTSTEP\t") and current is not None:
                    _, f = fields(line)
                    current["steps"][int(f["it"])][int(f["index"])] = f
                elif line.startswith("D14QF_PAIR\t") and current is not None:
                    _, f = fields(line)
                    current["pairs"][int(f["it"])] = f
                elif line.startswith("D14QF_ROOT\t"):
                    _, f = fields(line)
                    if f.get("seed") == "warm" and last_warm is not None:
                        last_warm["roots"][int(f["index"])] = f
                elif line.startswith("D14QF_STAGE\t"):
                    _, f = fields(line)
                    if f.get("kind") == "warm" and last_warm is not None:
                        last_warm["stage"] = f
    return calls


def read_rootwork() -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for case in (149, 0, 64, 9):
        path = ROOT / f"case{case}_roots.tsv"
        with path.open() as stream:
            reader = csv.DictReader(stream, delimiter=" ", skipinitialspace=True)
            for row in reader:
                key = tuple(row[k] for k in
                            ("case_id", "configuration_id", "profile", "d_bin_index",
                             "epoch_index", "rep", "lane"))
                out[key].append(row)
    return out


def min_assignment(cost: list[list[float]]) -> list[int]:
    """Small exact DP assignment for at most 14 roots."""
    n = len(cost)
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for i in range(n):
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, assign) in dp.items():
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                candidate = (total + cost[i][j], assign + (j,))
                prev = nxt.get(new_mask)
                if prev is None or candidate[0] < prev[0]:
                    nxt[new_mask] = candidate
        dp = nxt
    return list(dp[(1 << n) - 1][1])


def final_roles(call: dict, rootwork: dict) -> dict[int, dict]:
    roots = call["roots"]
    key = row_key(call["row"])
    rows = rootwork.get(key, [])
    by_index = {int(r["root_index"]): r for r in rows}
    if len(roots) != 14 or len(by_index) != 14:
        return {}
    warm = [roots[i] for i in range(14)]
    final = [by_index[i] for i in range(14)]
    wr = [(dval(r, "re"), dval(r, "im")) for r in warm]
    fr = [(Decimal(r["final_v_re"]), Decimal(r["final_v_im"])) for r in final]
    cost = []
    for ar, ai in wr:
        row = []
        for br, bi in fr:
            scale = max(Decimal(1), cabs(br, bi))
            row.append(float(cabs(ar - br, ai - bi) / scale))
        cost.append(row)
    assign = min_assignment(cost)
    result: dict[int, dict] = {}
    for i, j in enumerate(assign):
        dist = cabs(wr[i][0] - fr[j][0], wr[i][1] - fr[j][1])
        wsep = min(cabs(wr[i][0] - wr[k][0], wr[i][1] - wr[k][1])
                   for k in range(14) if k != i)
        fsep = min(cabs(fr[j][0] - fr[k][0], fr[j][1] - fr[k][1])
                   for k in range(14) if k != j)
        valid = dist <= Decimal("0.25") * min(wsep, fsep)
        record = final[j]
        result[i] = {
            "final_root_index": j,
            "match_valid": bool(valid),
            "match_distance": float(dist),
            "final_role": int(record["role"]),
            "physical_real": int(record["physical_real"]),
            "role_hint": int(record["role_hint"]),
        }
    return result


def summarize(call: dict, roles: dict[int, dict]) -> tuple[dict, list[dict]]:
    sweeps = call["steps"]
    its = sorted(sweeps)
    per_root: list[dict] = []
    max_ids: list[int] = []
    max_steps: list[Decimal] = []
    max_partners: list[int] = []
    for it in its:
        records = sweeps[it]
        dominant = max(records, key=lambda i: dval(records[i], "step_abs"))
        max_ids.append(dominant)
        max_steps.append(dval(records[dominant], "step_abs"))
        max_partners.append(int(records[dominant]["nearest_index"]))

    pair_f = [call["pairs"][it] for it in sorted(call["pairs"])]
    pair_keys = [tuple(sorted((int(p["i"]), int(p["j"])))) for p in pair_f]
    pair_counts = Counter(pair_keys)
    dominant_pair, dominant_pair_count = pair_counts.most_common(1)[0] if pair_counts else ((-1, -1), 0)
    driver_pair_records = [
        p for p in pair_f
        if tuple(sorted((int(p["i"]), int(p["j"])))) == dominant_pair
    ]
    pair_m = [complex(float(dval(p, "m_re")), float(dval(p, "m_im")))
              for p in driver_pair_records]
    pair_d2 = [complex(float(dval(p, "d2_re")), float(dval(p, "d2_im")))
               for p in driver_pair_records]
    pair_sep = [float(dval(p, "sep")) for p in driver_pair_records]
    m_drift = max((abs(z - pair_m[0]) for z in pair_m), default=math.nan)
    d2_drift = max((abs(z - pair_d2[0]) for z in pair_d2), default=math.nan)
    d2_rel_drift = d2_drift / max(abs(pair_d2[0]), 1e-300) if pair_d2 else math.nan

    for i in range(14):
        rs = [sweeps[it][i] for it in its if i in sweeps[it]]
        steps = [dval(r, "step_abs") for r in rs]
        newton_steps = [dval(r, "newton_abs") for r in rs]
        step_newton_ratios = [
            dval(r, "step_abs") / dval(r, "newton_abs")
            for r in rs if dval(r, "newton_abs") != 0
        ]
        top_count = sum(1 for k in max_ids if k == i)
        role = roles.get(i, {})
        per_root.append({
            "index": i,
            "role_hint": int(rs[0]["role_hint"]) if rs else -1,
            "matched_final_role": role.get("final_role", -1),
            "matched_physical_real": role.get("physical_real", -1),
            "role_match_valid": role.get("match_valid", False),
            "dominant_sweep_count": top_count,
            # Preserve qf exponents instead of converting tiny values to
            # binary64 (which can underflow to 5e-324 or zero).
            "step_first": str(steps[0]) if steps else "nan",
            "step_median": str(statistics.median(steps)) if steps else "nan",
            "step_last": str(steps[-1]) if steps else "nan",
            "step_max": str(max(steps)) if steps else "nan",
            "step_min": str(min(steps)) if steps else "nan",
            "newton_step_last": str(newton_steps[-1]) if newton_steps else "nan",
            "aberth_to_newton_ratio_median": (
                str(statistics.median(step_newton_ratios))
                if step_newton_ratios else "nan"
            ),
            "p_abs_last": rs[-1]["p_abs"] if rs else "nan",
            "nearest_sep_last": rs[-1]["nearest_sep"] if rs else "nan",
        })

    dominant_counts = Counter(max_ids)
    driver_i, driver_j = dominant_pair
    dominant_role_hints = Counter(
        int(sweeps[it][idx]["role_hint"]) for it, idx in zip(its, max_ids))
    monotone_fraction = (sum(b <= a for a, b in zip(max_steps, max_steps[1:])) /
                         max(1, len(max_steps) - 1))
    stage = call.get("stage", {})
    row = call["row"]
    summary = {
        **row,
        "qf_warm_sweeps": len(its),
        "qf_warm_converged": stage.get("converged", ""),
        "qf_warm_worst_residual": stage.get("worst", ""),
        "qf_warm_scalar_certificate": stage.get("scalar_certificate", ""),
        "max_step_last": float(max_steps[-1]) if max_steps else math.nan,
        "max_step_first": float(max_steps[0]) if max_steps else math.nan,
        "max_step_ratio_last_first": float(max_steps[-1] / max_steps[0]) if max_steps and max_steps[0] else math.nan,
        "dominant_root_counts": dict(sorted(dominant_counts.items())),
        "dominant_root_role_hints": dict(sorted(dominant_role_hints.items())),
        "dominant_partner_pair_counts": {
            f"{a}-{b}": n for (a, b), n in Counter(
                tuple(sorted((i, j))) for i, j in zip(max_ids, max_partners)
            ).most_common()
        },
        "max_step_monotone_nonincrease_fraction": monotone_fraction,
        "driver_pair": list(dominant_pair),
        "driver_pair_fraction": dominant_pair_count / max(1, len(pair_keys)),
        "driver_pair_role_hints": [
            int(driver_pair_records[0]["role_i"]),
            int(driver_pair_records[0]["role_j"])
        ] if driver_pair_records else [-1, -1],
        "driver_pair_final_roles": [
            roles.get(driver_i, {}).get("final_role", -1),
            roles.get(driver_j, {}).get("final_role", -1),
        ],
        "driver_pair_physical_real": [
            roles.get(driver_i, {}).get("physical_real", -1),
            roles.get(driver_j, {}).get("physical_real", -1),
        ],
        "driver_pair_role_match_valid": [
            roles.get(driver_i, {}).get("match_valid", False),
            roles.get(driver_j, {}).get("match_valid", False),
        ],
        "driver_pair_sep_first": pair_sep[0] if pair_sep else math.nan,
        "driver_pair_sep_last": pair_sep[-1] if pair_sep else math.nan,
        "driver_pair_m_abs_drift": m_drift,
        "driver_pair_d2_abs_drift": d2_drift,
        "driver_pair_d2_relative_drift": d2_rel_drift,
        "driver_pair_m_first": [pair_m[0].real, pair_m[0].imag] if pair_m else [],
        "driver_pair_m_last": [pair_m[-1].real, pair_m[-1].imag] if pair_m else [],
        "driver_pair_d2_first": [pair_d2[0].real, pair_d2[0].imag] if pair_d2 else [],
        "driver_pair_d2_last": [pair_d2[-1].real, pair_d2[-1].imag] if pair_d2 else [],
    }
    return summary, per_root


def main() -> None:
    rootwork = read_rootwork()
    summaries: list[dict] = []
    roots_out: list[dict] = []
    calls = parse_traces()
    for call in calls:
        roles = final_roles(call, rootwork)
        summary, per_root = summarize(call, roles)
        summaries.append(summary)
        for r in per_root:
            roots_out.append({
                **{k: call["row"].get(k, "") for k in
                   ("case", "config", "profile", "d_bin", "epoch", "rep", "lane")},
                "index": r["index"],
                **{k: v for k, v in r.items() if k != "index"},
            })
    (ROOT / "rootwise_summary.json").write_text(
        json.dumps({"schema": 1, "warm_qf_calls": summaries}, indent=2) + "\n")
    if roots_out:
        with (ROOT / "rootwise_per_root.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(roots_out[0]),
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(roots_out)
    print(f"warm qf calls={len(summaries)}; root summaries={len(roots_out)}")


if __name__ == "__main__":
    main()
