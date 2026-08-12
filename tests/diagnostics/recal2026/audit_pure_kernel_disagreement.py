#!/usr/bin/env python3
"""Audit cross-engine differences in an existing pure-kernel benchmark.

This script is deliberately post-hoc and read-only with respect to the
benchmark inputs.  It reconstructs the selected lcbinint value from the
self-convergence samples already stored in ``results.json`` and compares the
available values with two contextual witnesses:

* the tighter-tolerance VBM value stored in the controlled corpus; and
* the highest measured Cartesian and polar samples in each search ladder.

None of these comparisons changes timing eligibility, the selected Nbin, or
the reported speed winner.  The output describes agreement patterns only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _relative_to(value, witness):
    """Relative difference using the same scale convention as the benchmark."""

    if not _finite(value) or not _finite(witness):
        return None
    return float(
        abs(float(value) - float(witness)) / max(abs(float(witness)), 1.0)
    )


def _at(values, index):
    if values is None or index >= len(values):
        return None
    value = values[index]
    return float(value) if _finite(value) else None


def _row_key(row):
    return (
        int(row["case_id"]),
        str(row["profile"]),
        f"{float(row['x']):.17g}",
        f"{float(row['y']):.17g}",
    )


def _sample_value(grid, epoch_offset, nbin):
    if grid is None or nbin is None:
        return None
    sample = grid.get("samples", {}).get(str(epoch_offset), {})
    bins = sample.get("nbin", ())
    values = sample.get("magnification", ())
    for measured_nbin, value in zip(bins, values):
        if int(measured_nbin) == int(nbin) and _finite(value):
            return float(value)
    return None


def _tail_value(grid, epoch_offset):
    if grid is None:
        return None, None
    sample = grid.get("samples", {}).get(str(epoch_offset), {})
    pairs = [
        (int(nbin), float(value))
        for nbin, value in zip(
            sample.get("nbin", ()), sample.get("magnification", ())
        )
        if _finite(value)
    ]
    if not pairs:
        return None, None
    return max(pairs, key=lambda item: item[0])


def _witness_band(lcbinint_difference, vbm_difference, target):
    if lcbinint_difference is None or vbm_difference is None:
        return "unavailable"
    lcbinint_near = lcbinint_difference <= float(target)
    vbm_near = vbm_difference <= float(target)
    if lcbinint_near and vbm_near:
        return "both_within_nominal_band"
    if lcbinint_near:
        return "lcbinint_only_within_nominal_band"
    if vbm_near:
        return "target_vbm_only_within_nominal_band"
    return "neither_within_nominal_band"


def _outside_band(value, target):
    return None if value is None else bool(value > float(target))


def _audit_epoch(result, corpus_row, epoch_offset, corpus_epoch_index):
    target = float(result["target"])
    # Grid names are strings, so retrieve them without the numeric helper.
    chosen_names = result.get("chosen_grid", ())
    chosen_grid = (
        None if epoch_offset >= len(chosen_names)
        else chosen_names[epoch_offset]
    )
    chosen_nbin = _at(result.get("chosen_nbin"), epoch_offset)
    chosen_nbin = None if chosen_nbin is None else int(chosen_nbin)
    grid = result.get("grid", {})
    lcbinint_value = _sample_value(
        grid.get(chosen_grid), epoch_offset, chosen_nbin
    )

    # ``reference`` is the legacy field name for the independent VBM call at
    # the target RelTol in this benchmark schema.
    target_vbm_value = _at(result.get("reference"), epoch_offset)
    timed_vbm_value = _at(
        result.get("vbm", {}).get("timing_values"), epoch_offset
    )

    witness_entry = corpus_row.get("references", {}).get(
        str(corpus_epoch_index), {}
    )
    tight_vbm_witness = (
        float(witness_entry["value"])
        if _finite(witness_entry.get("value"))
        else None
    )
    tight_vbm_witness_uncertainty = (
        float(witness_entry["uncertainty"])
        if _finite(witness_entry.get("uncertainty"))
        else None
    )

    cartesian_tail_nbin, cartesian_tail = _tail_value(
        grid.get("cartesian"), epoch_offset
    )
    polar_tail_nbin, polar_tail = _tail_value(grid.get("polar"), epoch_offset)

    cross_engine_difference = _relative_to(
        lcbinint_value, target_vbm_value
    )
    lcbinint_to_tight_vbm = _relative_to(
        lcbinint_value, tight_vbm_witness
    )
    target_vbm_to_tight_vbm = _relative_to(
        target_vbm_value, tight_vbm_witness
    )
    timed_vbm_to_tight_vbm = _relative_to(
        timed_vbm_value, tight_vbm_witness
    )
    vbm_repeat_difference = _relative_to(timed_vbm_value, target_vbm_value)
    grid_tail_difference = _relative_to(cartesian_tail, polar_tail)

    return {
        "case_id": int(result["case_id"]),
        "profile": str(result["profile"]),
        "target": target,
        "epoch_offset": int(epoch_offset),
        "corpus_epoch_index": int(corpus_epoch_index),
        "x": float(result["x"]),
        "y": float(result["y"]),
        "d_over_rho": float(
            result.get("actual_d_over_rho", result.get("d_over_rho"))
        ),
        "chosen_grid": chosen_grid,
        "chosen_nbin": chosen_nbin,
        "lcbinint_value": lcbinint_value,
        "target_vbm_value": target_vbm_value,
        "timed_vbm_value": timed_vbm_value,
        "tight_vbm_witness": tight_vbm_witness,
        "tight_vbm_witness_uncertainty": tight_vbm_witness_uncertainty,
        "cartesian_tail_nbin": cartesian_tail_nbin,
        "cartesian_tail_witness": cartesian_tail,
        "polar_tail_nbin": polar_tail_nbin,
        "polar_tail_witness": polar_tail,
        "cross_engine_relative_difference": cross_engine_difference,
        "cross_engine_outside_nominal_band": _outside_band(
            cross_engine_difference, target
        ),
        "lcbinint_to_tight_vbm_witness": lcbinint_to_tight_vbm,
        "target_vbm_to_tight_vbm_witness": target_vbm_to_tight_vbm,
        "timed_vbm_to_tight_vbm_witness": timed_vbm_to_tight_vbm,
        "tight_vbm_witness_band": _witness_band(
            lcbinint_to_tight_vbm, target_vbm_to_tight_vbm, target
        ),
        "vbm_repeat_relative_difference": vbm_repeat_difference,
        "vbm_repeat_outside_nominal_band": _outside_band(
            vbm_repeat_difference, target
        ),
        "cartesian_polar_tail_relative_difference": grid_tail_difference,
        "cartesian_polar_tails_outside_nominal_band": _outside_band(
            grid_tail_difference, target
        ),
        "lcbinint_to_cartesian_tail": _relative_to(
            lcbinint_value, cartesian_tail
        ),
        "lcbinint_to_polar_tail": _relative_to(lcbinint_value, polar_tail),
    }


def _summarise(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["profile"], float(record["target"]))].append(record)

    summary = {}
    for (profile, target), rows in sorted(groups.items()):
        categories = Counter(
            row["tight_vbm_witness_band"] for row in rows
        )
        summary.setdefault(profile, {})[str(target)] = {
            "epochs": len(rows),
            "cross_engine_outside_nominal_band": sum(
                row["cross_engine_outside_nominal_band"] is True
                for row in rows
            ),
            "cross_engine_compared": sum(
                row["cross_engine_outside_nominal_band"] is not None
                for row in rows
            ),
            "tight_vbm_witness_band_counts": dict(categories),
            "vbm_repeat_outside_nominal_band": sum(
                row["vbm_repeat_outside_nominal_band"] is True
                for row in rows
            ),
            "vbm_repeat_compared": sum(
                row["vbm_repeat_outside_nominal_band"] is not None
                for row in rows
            ),
            "cartesian_polar_tails_outside_nominal_band": sum(
                row["cartesian_polar_tails_outside_nominal_band"] is True
                for row in rows
            ),
            "cartesian_polar_tails_compared": sum(
                row["cartesian_polar_tails_outside_nominal_band"] is not None
                for row in rows
            ),
        }
    return summary


def audit_payload(results_payload, corpus_rows):
    """Return an audit derived exclusively from already recorded values."""

    corpus_lookup = {_row_key(row): row for row in corpus_rows}
    corpus_epoch_indices = tuple(
        int(value)
        for value in results_payload.get("reference_indices", (0, 7, 15, 23))
    )
    records = []
    checked = 0
    max_residual = 0.0

    for result in results_payload.get("results", ()):
        if result.get("status") != "completed":
            continue
        key = _row_key(result)
        if key not in corpus_lookup:
            raise ValueError(
                "result row is absent from the corpus: "
                f"case={result.get('case_id')} profile={result.get('profile')}"
            )
        corpus_row = corpus_lookup[key]
        for offset, corpus_epoch_index in enumerate(corpus_epoch_indices):
            record = _audit_epoch(
                result, corpus_row, offset, corpus_epoch_index
            )
            stored = _at(result.get("chosen_vbm_errors"), offset)
            reconstructed = record["cross_engine_relative_difference"]
            if stored is not None and reconstructed is not None:
                residual = abs(stored - reconstructed)
                max_residual = max(max_residual, residual)
                checked += 1
                tolerance = 5.0e-15 + 1.0e-12 * abs(stored)
                if residual > tolerance:
                    raise ValueError(
                        "selected-value reconstruction disagrees with the "
                        "stored cross-engine difference: "
                        f"case={result['case_id']} epoch={offset} "
                        f"residual={residual:g}"
                    )
            records.append(record)

    return {
        "mode": "posthoc_read_only_no_kernel_remeasurement",
        "interpretation": {
            "speed_selection": (
                "unchanged; lcbinint self-convergence and target-VBM RelTol "
                "remain independent"
            ),
            "cross_engine_difference": (
                "diagnostic only; it does not alter eligibility or speed winner"
            ),
            "tight_vbm_witness": (
                "contextual tighter-tolerance VBM value from the corpus"
            ),
            "grid_tail_witnesses": (
                "highest already measured Nbin in each self-convergence ladder"
            ),
        },
        "self_check": {
            "reconstructed_stored_pairs": checked,
            "max_absolute_residual": max_residual,
        },
        "summary": _summarise(records),
        "epochs": records,
    }


def _fraction(item, count_key, total_key):
    return f"{item.get(count_key, 0)}/{item.get(total_key, 0)}"


def _markdown(audit, manifest):
    levels = manifest.get("reference_relative_levels", ())
    level_text = ", ".join(f"`{float(value):g}`" for value in levels)
    lines = [
        "# Post-hoc pure-kernel cross-engine audit",
        "",
        "This audit uses only values already stored by the benchmark and its",
        "controlled corpus. No kernel is rerun, and no Nbin, timing eligibility,",
        "speed ratio, or speed winner is changed.",
        "",
        "lcbinint reaches its three-point self-convergence rule independently;",
        "VBM is called at the requested `RelTol` independently. A cross-engine",
        "difference is therefore recorded as a diagnostic, not as an accuracy",
        "verdict about either engine.",
        "",
        "The tight-VBM witness is the stored finer-tolerance corpus value",
        (f"generated at relative levels {level_text}." if level_text else
         "from the corpus's tighter-tolerance VBM convergence pair."),
        "The Cartesian and polar tail witnesses are the largest Nbin values",
        "that were already evaluated in each point's self-convergence search.",
        "All three are contextual witnesses and do not participate in the speed",
        "comparison.",
        "",
        "## Summary",
        "",
        "`outside` means that the relative difference is larger than the nominal",
        "target-sized band; it is not an eligibility filter.",
        "",
        "| profile | target | epochs | cross-engine outside | both near tight-VBM | lcbinint only near | target-VBM only near | neither near | VBM repeats outside | grid tails outside |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, targets in audit["summary"].items():
        for target, item in sorted(
            targets.items(), key=lambda pair: float(pair[0]), reverse=True
        ):
            categories = item["tight_vbm_witness_band_counts"]
            lines.append(
                f"| {profile} | `{float(target):g}` | {item['epochs']} | "
                f"{_fraction(item, 'cross_engine_outside_nominal_band', 'cross_engine_compared')} | "
                f"{categories.get('both_within_nominal_band', 0)} | "
                f"{categories.get('lcbinint_only_within_nominal_band', 0)} | "
                f"{categories.get('target_vbm_only_within_nominal_band', 0)} | "
                f"{categories.get('neither_within_nominal_band', 0)} | "
                f"{_fraction(item, 'vbm_repeat_outside_nominal_band', 'vbm_repeat_compared')} | "
                f"{_fraction(item, 'cartesian_polar_tails_outside_nominal_band', 'cartesian_polar_tails_compared')} |"
            )

    largest = sorted(
        (
            row for row in audit["epochs"]
            if row["cross_engine_relative_difference"] is not None
        ),
        key=lambda row: row["cross_engine_relative_difference"],
        reverse=True,
    )[:20]
    lines += [
        "",
        "## Largest cross-engine differences",
        "",
        "These rows are candidates for a deeper independent convergence study;",
        "their ordering does not affect the speed comparison.",
        "",
        "| case | profile | target | epoch | d/rho | grid | Nbin | cross-engine \\|Δ\\| | lcbinint to tight-VBM | target-VBM to tight-VBM | Cartesian/polar tails |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    def show(value):
        return "n/a" if value is None else f"{float(value):.6g}"

    for row in largest:
        lines.append(
            f"| {row['case_id']} | {row['profile']} | `{row['target']:g}` | "
            f"{row['epoch_offset']} | {row['d_over_rho']:.3f} | "
            f"{row['chosen_grid']} | {row['chosen_nbin']} | "
            f"{show(row['cross_engine_relative_difference'])} | "
            f"{show(row['lcbinint_to_tight_vbm_witness'])} | "
            f"{show(row['target_vbm_to_tight_vbm_witness'])} | "
            f"{show(row['cartesian_polar_tail_relative_difference'])} |"
        )
    check = audit["self_check"]
    lines += [
        "",
        "## Reconstruction check",
        "",
        f"Reconstructed {check['reconstructed_stored_pairs']} stored selected-value",
        "comparisons from the saved ladders. The maximum absolute residual was",
        f"`{check['max_absolute_residual']:.3g}`.",
    ]
    return "\n".join(lines) + "\n"


def _input_file(path, filename):
    path = Path(path)
    return path / filename if path.is_dir() else path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results_path = _input_file(args.results, "results.json")
    corpus_path = _input_file(args.corpus, "rows.json")
    results_payload = json.loads(results_path.read_text())
    corpus_payload = json.loads(corpus_path.read_text())
    manifest_path = args.corpus / "manifest.json" if args.corpus.is_dir() else None
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path is not None and manifest_path.is_file()
        else {}
    )

    audit = audit_payload(results_payload, corpus_payload["rows"])
    audit["source_results"] = str(results_path)
    audit["source_corpus"] = str(corpus_path)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    report = _markdown(audit, manifest)
    (args.output / "REPORT_cross_engine_audit.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
