"""Run in lcbinint parent checkout; build path, output JSONL and optional case offset."""
import csv
import json
import sys
import time
from pathlib import Path
build = Path(sys.argv[1]).resolve()
sys.meta_path[:] = [finder for finder in sys.meta_path if "editable" not in str(finder)]
sys.path.insert(0, str(build))
import lcbinint as lc
import numpy as np

assert Path(lc._native.__file__).is_relative_to(build)
offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
vary_lens = len(sys.argv) > 4 and sys.argv[4] == "--vary-lens"


def fixtures():
    binary_inputs = Path("figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv")
    with binary_inputs.open() as stream:
        uniform = [r for r in csv.DictReader(stream, delimiter="\t") if r["profile"] == "uniform"]
    for i in (25, 360, 720, 1170, 1884, 2470, 2890, 3310):
        r = uniform[i + offset]
        params = dict(t0=0., tE=1., u0=float(r["y"]), alpha=0.,
                      s=float(r["s"]), q=1/float(r["q"]), rho=float(r["rho"]))
        yield "binary", r["geom_id"], "center_of_mass", float(r["time"]), params
    triple_inputs = Path("experiments/algebraic_boundary/evidence/triple_event_precision_20261029/inputs")
    for i in (568, 575, 651, 734, 810, 889, 935, 1010):
        case = json.loads((triple_inputs / f"{i+offset:04d}.json").read_text())
        yield "triple", str(i + offset), "vbm", 0., case["params"]


def benchmark(output):
    for lens, key, coords, time_center, params in fixtures():
        for limb in (0., 0.5):
            options = lc.Options(param_type=coords, finite_source_backend="algebraic_boundary",
                                 finite_source_tol=0., finite_source_reltol=1e-4)
            for gradient in (False, True):
                curve = lc.LightCurve(lens=lens, ld_c=limb, options=options)
                for j in range(4):
                    p = dict(params, u0=params["u0"] + j * .012 * params["rho"])
                    if vary_lens:
                        p.update(s=p["s"] * (1 + .0001*j), q=p["q"] * (1 - .0001*j))
                    times = [time_center + x * p["rho"] for x in (-.08, .08)]
                    begin = time.thread_time_ns()
                    result = curve.magnification(times, p, with_gradient=gradient)
                    ms = (time.thread_time_ns() - begin) / 1e6
                    row = dict(lens=lens, id=key, ld=limb, grad=gradient, j=j, ms=ms)
                    for name, field in (("mu", "magnifications"), ("status", "status"),
                                        ("valid", "gradient_valid"), ("g", "gradients"),
                                        ("support", "support_certified"),
                                        ("vc", "value_converged"), ("seeds", "seed_support_proven"),
                                        ("discovery", "frontier_discovery_completed"),
                                        ("integrated", "support_required")):
                        row[name] = np.asarray(getattr(result, field)).tolist()
                    print(json.dumps(row), file=output)


if __name__ == "__main__":
    with open(sys.argv[2], "w") as output:
        benchmark(output)
