#!/usr/bin/env python3
"""Derive uniform / pure-LD X gradients from independent direct FD values."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = ROOT / "projective_fd.tsv"
groups = defaultdict(dict)
with raw.open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["mode"] != "VALUE" or row["map_used"] != "on":
            continue
        if int(row["fail"]) != 0:
            continue
        key = (row["resolution"], float(row["hscale"]))
        groups[key][int(row["offset"])] = (float(row["mu0"]), float(row["muhalf"]))

result = []
rho = 0.02  # row 17 caustic-cross source radius
for (resolution, hscale), vals in sorted(groups.items()):
    if set(vals) != {-2, -1, 1, 2}:
        continue
    h = rho * hscale
    fd0 = (vals[-2][0] - 8 * vals[-1][0] + 8 * vals[1][0] - vals[2][0]) / (12 * h)
    fdh = (vals[-2][1] - 8 * vals[-1][1] + 8 * vals[1][1] - vals[2][1]) / (12 * h)
    # mu(u=.5)=(.5*F0+.5*Fh)/(5/6 D0), while
    # mu(u=1)=Fh/(2/3 D0); eliminate Fh/D0 to get
    # mu1'=(5/2) mu_half'-(3/2) mu0'.
    fd1 = 2.5 * fdh - 1.5 * fd0
    result.append({
        "resolution": resolution,
        "hscale": hscale,
        "fd_uniform_u0": fd0,
        "fd_mixed_u05": fdh,
        "fd_pure_ld_u1": fd1,
        "projective_folds": 2,
    })

with (ROOT / "independent_fd_components.json").open("w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
with (ROOT / "independent_fd_components.tsv").open("w") as f:
    f.write("resolution\thscale\tfd_uniform_u0\tfd_mixed_u05\tfd_pure_ld_u1\tprojective_folds\n")
    for r in result:
        f.write("{resolution}\t{hscale:.17g}\t{fd_uniform_u0:.17g}\t{fd_mixed_u05:.17g}\t{fd_pure_ld_u1:.17g}\t{projective_folds}\n".format(**r))
print(json.dumps(result, indent=2))
