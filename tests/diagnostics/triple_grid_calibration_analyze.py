#!/usr/bin/env python3
"""Analyze production triple-grid calibration data and search cheap rules."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ABS_TOL = 1e-4
REL_TOL = 1e-3


def finite(row: dict[str, Any]) -> bool:
    return "value" in row and math.isfinite(float(row["value"]))


def tail(sequence: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [x for x in sequence if finite(x)]
    if len(valid) < 2:
        return None
    return {"value": float(valid[-1]["value"]), "change": abs(float(valid[-1]["value"])-float(valid[-2]["value"])),
            "bins": int(valid[-1]["bins"]), "method": valid[-1].get("method")}


def reference(row: dict[str, Any]) -> dict[str, Any]:
    tails = {grid: tail(seq) for grid, seq in row["fixed_sequences"].items()}
    if any(value is None for value in tails.values()):
        return {"trusted": False, "reason": "missing tail"}
    scale = float(np.median([value["value"] for value in tails.values()]))
    tolerance = ABS_TOL + REL_TOL * max(abs(scale), 1.0)
    if any(value["change"] > 2*tolerance for value in tails.values()):
        return {"trusted": False, "reason": "unstable tail"}
    spread = abs(tails["cartesian"]["value"] - tails["polar"]["value"])
    if spread > 2*tolerance:
        return {"trusted": False, "reason": "grid disagreement"}
    return {"trusted": True, "value": scale, "tolerance": tolerance}


def required(sequence: list[dict[str, Any]], ref: dict[str, Any]) -> dict[str, Any]:
    for index, candidate in enumerate(sequence):
        suffix = sequence[index:]
        if finite(candidate) and suffix and all(
                finite(x) and abs(float(x["value"])-ref["value"]) <= ref["tolerance"] for x in suffix):
            return {"bins": int(candidate["bins"]), "elapsed_ns": int(candidate["elapsed_ns"]),
                    "method": candidate.get("method"), "censored": False}
    valid = [x for x in sequence if finite(x)]
    return {"bins": int(valid[-1]["bins"]) if valid else 0, "censored": True}


def load(directory: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records=[]; meta={"cases":0,"rows":0,"trusted":0,"untrusted":0}
    for path in sorted(directory.glob("case-*.json")):
        doc=json.loads(path.read_text())
        if "error" in doc: continue
        meta["cases"] += 1
        for row in doc["rows"]:
            meta["rows"] += 1; ref=reference(row)
            if not ref["trusted"]: meta["untrusted"] += 1; continue
            meta["trusted"] += 1
            needs={grid:required(seq,ref) for grid,seq in row["fixed_sequences"].items()}
            records.append({"case":doc["case"],"row":row,"reference":ref,"required":needs})
    return records,meta


def choose(record: dict[str, Any], a_uniform: float, a_limb: float, d_min: float) -> str:
    row=record["row"]; cutoff=a_uniform if float(row["limb_c"]) == 0 else a_limb
    return "polar" if abs(float(row["point_magnification"])) >= cutoff and float(row["caustic_distance_over_rho"]) >= d_min else "cartesian"


def at_bins(record: dict[str, Any], grid: str, bins: int) -> dict[str, Any] | None:
    return next((x for x in record["row"]["fixed_sequences"][grid]
                 if int(x["bins"]) == bins and finite(x)), None)


def search(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable=[r for r in records if all(not r["required"][g]["censored"] for g in ("cartesian","polar"))]
    best=None
    for au in (30,50,100,200,300,500,1000,math.inf):
        for al in (30,50,100,200,300,500,1000,math.inf):
            for dm in (0,.2,.5,1,2,3,5,10):
                for polar_bins in (64,80,100,128,160,200):
                    total=0; violations=0; polar=0
                    for r in usable:
                        grid=choose(r,au,al,dm); selected=r["required"][grid]
                        if grid=="polar":
                            measured=at_bins(r,"polar",polar_bins)
                            accurate=(measured is not None and measured.get("method")=="inverse_ray_polar" and
                                      abs(float(measured["value"])-r["reference"]["value"]) <= r["reference"]["tolerance"])
                            if accurate:
                                selected={"elapsed_ns":measured["elapsed_ns"]}; polar += 1
                            else:
                                violations += 1
                        total += int(selected["elapsed_ns"])
                    candidate={"uniform_point_mag_cutoff":au,"limb_point_mag_cutoff":al,
                               "distance_over_rho_min":dm,"polar_bins":polar_bins,
                               "elapsed_ns":total,"violations":violations,
                               "polar_rows":polar,"rows":len(usable)}
                    if best is None or (candidate["violations"],candidate["elapsed_ns"]) < (best["violations"],best["elapsed_ns"]):
                        best=candidate
    cart=sum(r["required"]["cartesian"].get("elapsed_ns",0) for r in usable)
    oracle=sum(min(r["required"][g].get("elapsed_ns",10**30) for g in ("cartesian","polar")) for r in usable)
    best["all_cartesian_elapsed_ns"]=cart; best["oracle_elapsed_ns"]=oracle
    return best


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);p.add_argument("--output",type=Path);args=p.parse_args()
    records,meta=load(args.directory); result={"metadata":meta,"grid_rule":search(records)}
    rendered=json.dumps(result,indent=2,allow_nan=True);print(rendered)
    if args.output: args.output.write_text(rendered+"\n")


if __name__=="__main__": main()
