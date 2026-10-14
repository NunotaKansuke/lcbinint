"""M5 audit sweep (plan sec. 16): accuracy vs failure rate, no silent miss.

Runs :func:`holonomic_ref.solve_epoch` over a curated hard set plus a seeded
random sample of the design domain

    q   in [1e-8, 1]      (log-uniform)
    a   in [0.1, 10]      (log-uniform)
    rho in [1e-3, 0.3]    (log-uniform)
    |source offset| chosen to graze the caustic structure

(the random rho floor is 1e-3, not the plan's 1e-6: below that the
inverse-ray reference spends minutes per point and the finite-source value
is within the reference's own noise of the point-source limit anyway.  The
rho ~ 1e-6 / extreme-q corners are covered by the curated set, whose
inverse-ray references were checked to terminate.)

and compares every value to the production inverse-ray solver
(``lcbinint.binary_ray_shooting``), which is the plan's correctness bar
("現行 production solver 同等以上").

Three outcomes are reported **separately** (the M5 gate):

* **accuracy** -- the error distribution over points the solver returned as
  ``OK_VALIDATED`` (it claims the value is good and an independent check
  agreed);
* **failure rate** -- the fraction of points the solver flagged with a
  non-clean status (``TOPOLOGY_UNCERTAIN`` / ``LOCAL_REFERENCE_USED`` /
  ``TRANSPORT_TOLERANCE_FAILED`` / ...), i.e. it declined to certify;
* **silent misses** -- points the solver returned as ``OK`` or
  ``OK_VALIDATED`` whose value is nonetheless wrong by more than
  ``MISS_RTOL``.  This count **must be zero**.

Usage::

    python -m benchmarks.holonomic.audit            # curated + 40 random
    python -m benchmarks.holonomic.audit --n 200    # bigger random sample
    python -m benchmarks.holonomic.audit --quick    # curated only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_REF = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", "python", "lcbinint"))
if _REF not in sys.path:
    sys.path.insert(0, _REF)

import numpy as np                                           # noqa: E402

import holonomic_ref as H                                    # noqa: E402
from holonomic_ref import LensParams                          # noqa: E402
from holonomic_ref.direct_quadrature import (                   # noqa: E402
    image_plane_flux_grid, source_plane_flux)

try:
    import lcbinint                                           # noqa: E402
except Exception:                                             # pragma: no cover
    lcbinint = None

# a solver-reported value is a "silent miss" if its status is clean but it
# is wrong by more than this vs the production solver.  Looser than the
# solver's own 3e-3 promotion bar so the production solver's ~1e-3 grid
# error does not raise false alarms.
MISS_RTOL = 5.0e-3
CLEAN = {"OK", "OK_VALIDATED"}
U_VALUES = (0.0, 0.5)


# --------------------------------------------------------------- config sets
def curated():
    """Hard configs -- the plan sec. 16 emphasis regions."""
    c = [
        ("plan15", 1 / 5, 1 / 7, 1 / 8, 0.5, 1.2),
        ("resonant", 0.05, 0.02, 0.05, 0.3, 0.9),
        ("close-binary", 0.4, -0.05, 0.09, 0.8, 0.55),
        ("wide-planet", 1.4, 0.10, 0.03, 1e-3, 2.5),
        ("cusp-approach", 0.9, -0.3, 0.05, 0.25, 0.7),
        ("big-source", 0.1, 0.1, 0.3, 0.5, 0.8),
        ("zeta~0", 3e-4, 0.0, 0.1, 0.4, 1.1),
        ("zeta=0-exact", 0.0, 0.0, 0.1, 0.4, 1.1),
        ("on-axis-off", 0.5, 0.0, 0.05, 0.4, 1.1),
        ("R=a-graze", 1.2, 0.0, 0.05, 0.5, 1.2),
        ("R=sqrt(m0)", 0.0, 0.8165, 0.03, 0.5, 1.5),
        ("tiny-rho", 0.1, 0.02, 1e-3, 0.3, 1.0),
        ("caustic-cross", 0.12, 0.0, 0.02, 0.5, 1.0),
        ("extreme-q-lo", 0.7, 0.05, 0.02, 1e-6, 1.5),
        ("extreme-q-planet", 1.0, 0.01, 0.015, 3e-5, 1.6),
        ("very-wide", 3.0, 0.2, 0.05, 0.4, 5.0),
        ("very-close", 0.1, 0.05, 0.06, 0.6, 0.25),
        ("fold-tangent", 0.55, 0.28, 0.04, 0.7, 1.0),
    ]
    return [(n, dict(xs=xs, ys=ys, rho=rho, q=q, a=a, barycentric=False))
            for (n, xs, ys, rho, q, a) in c]


def random_sample(n, seed=20260907):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = 10.0 ** rng.uniform(-8, 0)
        a = 10.0 ** rng.uniform(-1, 1)
        rho = 10.0 ** rng.uniform(-3, math.log10(0.3))
        # source within ~1.3 Einstein radii of the primary, biased to the
        # axis and the companion (where the caustics live)
        r = abs(rng.normal(0.0, 0.5)) + rng.uniform(0.0, 0.3)
        ang = rng.uniform(0.0, 2 * math.pi)
        xs, ys = r * math.cos(ang), r * math.sin(ang)
        if rng.random() < 0.25:
            ys *= 0.02                                        # near-axis
        out.append((f"rand{i:03d}",
                    dict(xs=xs, ys=ys, rho=rho, q=q, a=a, barycentric=False)))
    return out


# ------------------------------------------------------------------- oracle
def _covers_a_lens(cfg):
    """True when the source disk encloses a point lens centre -- inverse-ray
    shooting then adaptively refines forever near the enclosed point-mass
    singularity and never returns, so it cannot be the oracle.

    Strict geometric coverage (``d < rho``): a disk whose *edge* merely grazes
    a lens (e.g. the resonant caustic config, ``d0/rho ~ 1.08``) is fine for
    ``binary_ray_shooting`` and must NOT be diverted to the source-plane
    quadrature, which is itself unreliable near a caustic
    (``checkpoint_M5.md`` sec. 4)."""
    xs, ys, rho, a = cfg["xs"], cfg["ys"], cfg["rho"], cfg["a"]
    d0 = math.hypot(xs, ys)                          # primary lens at origin
    d1 = math.hypot(xs - a, ys)                      # companion at +a
    return d0 < rho or d1 < rho


def _grid_reference(cfg, u):
    """Image-plane dense-grid ``mu`` -- bounded, convergent both at a caustic
    and with a lens inside the disk (unlike the source-plane quadrature)."""
    p = LensParams(**cfg)
    f0, fh = image_plane_flux_grid(p, n_r=1600, n_theta=24000)
    rho2 = cfg["rho"] ** 2
    return float(((1.0 - u) * f0 + u * fh) / (math.pi * rho2 * (1.0 - u / 3.0)))


def _source_plane_reference(cfg, u):
    p = LensParams(**cfg)
    f0, fh = source_plane_flux(p, n_r=128, n_theta=768)
    rho2 = cfg["rho"] ** 2
    return float(((1.0 - u) * f0 + u * fh) / (math.pi * rho2 * (1.0 - u / 3.0)))


def ray_reference(cfg, u):
    """Production inverse-ray ``mu`` -- or the image-plane grid where the
    inverse-ray solver would hang (source centre on a lens)."""
    if _covers_a_lens(cfg):
        try:
            return _grid_reference(cfg, u)
        except Exception as exc:
            return f"ERR:{exc!r}"
    if lcbinint is None:
        try:
            return _source_plane_reference(cfg, u)
        except Exception as exc:
            return f"ERR:{exc!r}"
    q, a = cfg["q"], cfg["a"]
    m1a = (q / (1.0 + q)) * a
    ld = lcbinint.LimbDarkening.linear(u) if u else None
    try:
        return float(lcbinint.binary_ray_shooting(
            cfg["xs"] - m1a, cfg["ys"], s=a, q=q, rho=cfg["rho"],
            limb_darkening=ld))
    except Exception as exc:
        return f"ERR:{exc!r}"


# --------------------------------------------------------------------- run
def run(configs, *, verbose=True):
    rows = []
    for name, cfg in configs:
        p = LensParams(**cfg)
        for u in U_VALUES:
            t0 = time.perf_counter()
            try:
                er = H.solve_epoch(p, u)
                mu, status = er.mu, er.status
                err = None
            except Exception as exc:
                mu, status, err = float("nan"), "EXCEPTION", repr(exc)
            dt = time.perf_counter() - t0
            ref = ray_reference(cfg, u)
            rel = None
            if isinstance(ref, float) and math.isfinite(mu) and ref != 0.0:
                rel = abs(mu - ref) / abs(ref)
            row = dict(name=name, u=u, mu=mu, ref=ref, rel=rel,
                       status=status, seconds=dt, error=err, cfg=cfg)
            rows.append(row)
            if verbose:
                rs = "  ref n/a" if not isinstance(ref, float) else f"ref={ref:11.5f}"
                rels = "   --   " if rel is None else f"rel={rel:.1e}"
                print(f"  {name:18s} u={u:<3} mu={mu:12.5f} {rs} {rels} "
                      f"{status:24s} {dt:5.1f}s")
    return rows


def summarise(rows):
    have_ref = [r for r in rows if r["rel"] is not None]
    validated = [r for r in have_ref if r["status"] == "OK_VALIDATED"]
    flagged = [r for r in rows if r["status"] not in CLEAN]
    silent = [r for r in have_ref
              if r["status"] in CLEAN and r["rel"] > MISS_RTOL]

    def pct(xs, p):
        return float(np.percentile(xs, p)) if xs else float("nan")

    rels_val = [r["rel"] for r in validated]
    by_status: dict = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    return {
        "n_points": len(rows),
        "n_with_ref": len(have_ref),
        "accuracy_over_validated": {
            "n": len(validated),
            "median_rel": pct(rels_val, 50),
            "p90_rel": pct(rels_val, 90),
            "max_rel": max(rels_val) if rels_val else float("nan"),
        },
        "failure_rate": {
            "n_flagged": len(flagged),
            "fraction": len(flagged) / max(len(rows), 1),
            "by_status": by_status,
        },
        "silent_misses": [
            dict(name=r["name"], u=r["u"], rel=r["rel"], status=r["status"],
                 mu=r["mu"], ref=r["ref"], cfg=r["cfg"]) for r in silent
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="random sample size")
    ap.add_argument("--quick", action="store_true", help="curated set only")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args(argv)

    configs = curated()
    if not args.quick:
        configs += random_sample(args.n, args.seed)

    if lcbinint is None:
        print("WARNING: lcbinint not importable -- no production reference, "
              "silent-miss check degraded")

    print(f"# M5 audit -- {len(configs)} configs x {len(U_VALUES)} u-values")
    t0 = time.perf_counter()
    rows = run(configs)
    summary = summarise(rows)
    summary["wall_seconds"] = time.perf_counter() - t0

    print("\n" + "=" * 70)
    a = summary["accuracy_over_validated"]
    print(f"accuracy (OK_VALIDATED points, n={a['n']}): "
          f"median rel {a['median_rel']:.2e}, p90 {a['p90_rel']:.2e}, "
          f"max {a['max_rel']:.2e}")
    f = summary["failure_rate"]
    print(f"failure rate: {f['n_flagged']}/{summary['n_points']} "
          f"({100 * f['fraction']:.1f}%) flagged non-clean")
    for st, k in sorted(f["by_status"].items()):
        print(f"    {st:26s} {k}")
    sm = summary["silent_misses"]
    print(f"\nSILENT MISSES (status clean but rel > {MISS_RTOL:g}): {len(sm)}")
    for m in sm:
        print(f"    {m['name']:18s} u={m['u']} rel={m['rel']:.2e} "
              f"status={m['status']} cfg={m['cfg']}")
    print("=" * 70)
    print("GATE:", "PASS -- no silent miss" if not sm else "FAIL")

    out = args.out or os.path.join(os.path.dirname(__file__),
                                   "..", "..", "evidence", "holonomic",
                                   "audit_M5.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"summary": summary, "rows": [
            {k: v for k, v in r.items() if k != "cfg"} | {"cfg": r["cfg"]}
            for r in rows]}, fh, indent=2, default=str)
    print(f"wrote {out}")
    return 0 if not sm else 1


if __name__ == "__main__":
    raise SystemExit(main())
