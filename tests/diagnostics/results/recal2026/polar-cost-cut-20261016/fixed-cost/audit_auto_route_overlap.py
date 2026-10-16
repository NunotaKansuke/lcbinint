import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
SELECTED = ROOT / "tests/diagnostics/results/recal2026/predictive-route-20261015/selected-inputs.tsv"
FULL_DIR = ROOT / (
    "tests/diagnostics/results/recal2026/polar-speed-20261014/"
    "polar-cost-cut-full-initial2-gridratio6-v1-20261016"
)
QUANTUM = Decimal("0.000000000001")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def values_by_id(path):
    values = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"{path}:{line_number}: expected id plus six values")
        row_id = int(fields[0])
        if row_id in values:
            raise ValueError(f"duplicate geometry id {row_id} in {path}")
        values[row_id] = tuple(fields[1:])
    return values


def q12(values):
    return tuple(Decimal(value).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)
                 for value in values)


def main():
    selected = values_by_id(SELECTED)
    shards = sorted(FULL_DIR.glob("*-input.tsv"))
    full = {}
    for shard in shards:
        rows = values_by_id(shard)
        overlap = full.keys() & rows.keys()
        if overlap:
            raise ValueError(f"duplicate geometry ids across shards: {sorted(overlap)[:5]}")
        full.update(rows)

    common = selected.keys() & full.keys()
    value_matches = [row_id for row_id in common
                     if q12(selected[row_id]) == q12(full[row_id])]
    payload = {
        "comparison_columns": ["s", "q", "rho", "x", "y", "c"],
        "decimal_places": 12,
        "selected_rows": len(selected),
        "full_auto_rows": len(full),
        "full_auto_input_shards": len(shards),
        "id_overlap": len(common),
        "value_overlap_at_12dp": len(value_matches),
        "selected_sha256": sha256_bytes(SELECTED.read_bytes()),
        "full_shards_sha256": sha256_bytes(b"".join(
            shard.name.encode() + b"\0" + shard.read_bytes() for shard in shards
        )),
        "matched_geometry_ids": sorted(value_matches),
    }
    output = OUT / "selector-auto-overlap.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "matched_geometry_ids"}, indent=2))


if __name__ == "__main__":
    main()
