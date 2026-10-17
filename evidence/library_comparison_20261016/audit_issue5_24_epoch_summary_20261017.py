from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VBM_DIR = HERE / "issue5_vbm_epoch_isolation_20261017"
OTHER_RAW = HERE / "issue5_stratified_09_remaining_apis_20261017/raw/stratified_09.npz"
LC_RAW = HERE / "issue5_stratified_09_fallback_verification_20261017/raw/stratified_09.npz"
OUT = HERE / "issue5_stratified_09_24_epoch_comparison_20261017"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = json.loads((VBM_DIR / "metadata.json").read_text())
    by_index = {}
    for row in metadata["rows"]:
        by_index.setdefault(row["index"], {})[row["library"]] = row
    with np.load(OTHER_RAW, allow_pickle=False) as other, np.load(
        LC_RAW, allow_pickle=False
    ) as lc:
        times = np.asarray(lc["times"], dtype=np.float64)
        lc_accepted = np.asarray(lc["value_accepted__lcbinint"], dtype=bool)
        selected = [
            index for index in range(times.size)
            if lc_accepted[index]
            and all(by_index.get(index, {}).get(name, {}).get("status") == "completed"
                    for name in ("vbm", "autodiff_vbm"))
        ]
        values = {
            "vbm": np.asarray([by_index[i]["vbm"]["value"] for i in selected]),
            "autodiff_vbm": np.asarray([
                by_index[i]["autodiff_vbm"]["value"] for i in selected
            ]),
            "microlux": np.asarray(other["value__microlux"])[selected],
            "microjax": np.asarray(other["value__microjax"])[selected],
            "lcbinint": np.asarray(lc["value__lcbinint"])[selected],
        }
        family_values = np.stack([
            np.median(np.stack([values["vbm"], values["autodiff_vbm"]]), axis=0),
            values["microlux"], values["microjax"], values["lcbinint"],
        ])
        consensus = np.median(family_values, axis=0)
        scale = np.maximum(np.abs(consensus), 1.0e-12)
        relative_to_consensus = {
            name: np.abs(value - consensus) / scale
            for name, value in values.items()
        }
        pairwise = {}
        for i, left in enumerate(("vbm_family", "microlux", "microjax", "lcbinint")):
            for right in ("vbm_family", "microlux", "microjax", "lcbinint")[i + 1:]:
                left_value = family_values[
                    ("vbm_family", "microlux", "microjax", "lcbinint").index(left)
                ]
                right_value = family_values[
                    ("vbm_family", "microlux", "microjax", "lcbinint").index(right)
                ]
                pairwise[f"{left}_vs_{right}"] = float(np.max(
                    np.abs(left_value - right_value)
                    / np.maximum(np.abs(consensus), 1.0e-12)
                ))
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT / "values.npz",
            indices=np.asarray(selected, dtype=np.int64),
            times=times[selected],
            **values,
            vbm_family=np.median(np.stack([values["vbm"], values["autodiff_vbm"]]), axis=0),
            descriptive_family_median=consensus,
        )
        summary = {
            "case_id": "stratified_09",
            "included_epochs": len(selected),
            "excluded_epochs": sorted(set(range(times.size)) - set(selected)),
            "consensus_is_reference": False,
            "consensus_definition": "per-epoch median of four families; official VBM 5.6.1 and autodiff-vbm are collapsed to one VBM family",
            "pairwise_max_relative_difference_to_family_median": pairwise,
            "max_relative_difference_to_descriptive_median": {
                name: float(np.max(diff)) for name, diff in relative_to_consensus.items()
            },
            "source_artifacts": {
                str(OTHER_RAW.relative_to(HERE)): sha256(OTHER_RAW),
                str(LC_RAW.relative_to(HERE)): sha256(LC_RAW),
                "issue5_vbm_epoch_isolation_20261017/metadata.json": sha256(
                    VBM_DIR / "metadata.json"
                ),
            },
            "note": "The three excluded center epochs have support-unproven lcbinint values and timed out for both VBM-family implementations. The MicroLUX center outputs also carry sampling-capacity warnings; no consensus value is treated as truth.",
        }
        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
