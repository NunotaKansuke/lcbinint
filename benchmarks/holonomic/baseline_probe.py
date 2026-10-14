"""M0 baseline probe (plan sec. 14, milestone M0).

Fixes the cost/accuracy baseline that the M7 C++ holonomic solver has to
beat.  The M7 adoption gate is

    "同じ精度・被覆で linear-LD value+Jacobian の end-to-end median >= 2x,
     p95 悪化 <= 25%"

so the baseline has to be a *value + 5-component Jacobian* per epoch, over
the design domain, with a documented accuracy and coverage.

Baseline choice (design decision 18, recorded in ``docs/holonomic/milestones.md``
and ``docs/holonomic/checkpoint_M7.md``)
-----------------------------------------------------------------------------
The natural baseline is the JAX differentiable backend --
``binary_ray_shooting(..., jax=True)`` under one forward/reverse pass.  It
is **unavailable in this build**::

    >>> lcbinint.binary_ray_shooting(0.2, 0.1, s=1.2, q=0.5, rho=0.1, jax=True)
    RuntimeError: lcbinint was built without the polar epoch FFI
    #   _lcbinint._jax_ir has no attribute 'polar_epoch_directional_ffi'

and it cannot be rebuilt here: the FFI lives in the single shared
``site-packages/_lcbinint*.so`` that every other session and the user's own
sweeps import, and a ``pip install`` rebuild overwrites it globally.

The practical production route for a value + Jacobian today is therefore a
central finite-difference sweep of the native inverse-ray solver
``lcbinint.binary_ray_shooting`` (11 calls for 5 parameters).  That is what
this probe measures and what M7 must beat by >= 2x on the median.

For honesty the probe also reports the single value-only call cost (the
absolute floor any Jacobian-free consumer sees) and the Python
``holonomic_ref`` reference cost (not a target -- it is the oracle, pure
Python, and slow).

Outcomes, reported separately for uniform and linear-LD (u = 0.5):

  * value-only call cost            median / p95      (native inverse-ray)
  * value+Jacobian cost             median / p95      (central FD, 11 calls)
  * coverage / failure rate         (exception / timeout / non-finite)
  * value accuracy                  vs image_plane_flux_grid dense reference

Usage::

    taskset -c 0-7 python -m benchmarks.holonomic.baseline_probe --quick
    taskset -c 0-7 python -m benchmarks.holonomic.baseline_probe --n 24
    taskset -c 0-7 python -m benchmarks.holonomic.baseline_probe --n 24 --ref
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time

_REF = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", "python", "lcbinint"))
if _REF not in sys.path:
    sys.path.insert(0, _REF)

import numpy as np                                           # noqa: E402

import holonomic_ref as H                                    # noqa: E402
from holonomic_ref import LensParams                          # noqa: E402
from holonomic_ref.direct_quadrature import (                  # noqa: E402
    image_plane_flux_grid)

from benchmarks.holonomic.audit import (                       # noqa: E402
    curated, random_sample, _covers_a_lens)

try:
    import lcbinint                                           # noqa: E402
except Exception as _exc:                                     # pragma: no cover
    lcbinint = None
    _IMPORT_ERR = _exc

# central-difference relative steps, per user parameter (xs, ys, rho, q, a)
FD_REL = dict(xs=1e-4, ys=1e-4, rho=1e-3, q=1e-3, a=1e-4)
FD_FLOOR = dict(xs=1e-7, ys=1e-7, rho=1e-7, q=1e-9, a=1e-7)
PARAMS = ("xs", "ys", "rho", "q", "a")
U_VALUES = (0.0, 0.5)
CALL_TIMEOUT = 90.0            # s, per single binary_ray_shooting call budget
JAC_TIMEOUT = 900.0           # s, per value+Jacobian (11 calls) budget


# --------------------------------------------------------------- ray helper
def _ray_mu(cfg, u):
    """Native inverse-ray ``mu`` for a holonomic_ref-convention config
    (primary at origin, source at (xs, ys))."""
    q, a = cfg["q"], cfg["a"]
    m1a = (q / (1.0 + q)) * a
    ld = lcbinint.LimbDarkening.linear(u) if u else None
    return float(lcbinint.binary_ray_shooting(
        cfg["xs"] - m1a, cfg["ys"], s=a, q=q, rho=cfg["rho"],
        limb_darkening=ld))


def _value_and_jac(cfg, u):
    """(mu, grad[5], n_calls, per_call_seconds[list]).  Central FD."""
    calls = []

    def timed(c):
        t0 = time.perf_counter()
        v = _ray_mu(c, u)
        calls.append(time.perf_counter() - t0)
        return v

    mu = timed(cfg)
    grad = []
    for key in PARAMS:
        h = max(FD_REL[key] * abs(cfg[key]), FD_FLOOR[key])
        cp = dict(cfg); cp[key] = cfg[key] + h
        cm = dict(cfg); cm[key] = cfg[key] - h
        vp = timed(cp)
        vm = timed(cm)
        grad.append((vp - vm) / (2.0 * h))
    return mu, grad, len(calls), calls


# ------------------------------------------------------- subprocess wrapper
def _worker(cfg, u, q):
    try:
        t0 = time.perf_counter()
        v_only = _ray_mu(cfg, u)
        t_value = time.perf_counter() - t0
        mu, grad, ncalls, calls = _value_and_jac(cfg, u)
        q.put(dict(ok=True, v_only=v_only, t_value=t_value,
                   mu=mu, grad=grad, ncalls=ncalls,
                   t_jac=float(sum(calls))))
    except Exception as exc:                                  # pragma: no cover
        q.put(dict(ok=False, err=repr(exc)))


def probe_one(name, cfg, u):
    """Run one (config, u) in a child process with a hard timeout."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(cfg, u, q))
    t0 = time.perf_counter()
    p.start()
    p.join(JAC_TIMEOUT)
    wall = time.perf_counter() - t0
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(name=name, u=u, status="TIMEOUT", wall=wall, cfg=cfg)
    try:
        res = q.get_nowait()
    except Exception:
        return dict(name=name, u=u, status="NO_RESULT", wall=wall, cfg=cfg)
    if not res.get("ok"):
        return dict(name=name, u=u, status="EXCEPTION", wall=wall,
                    error=res.get("err"), cfg=cfg)
    res.update(name=name, u=u, status="OK", wall=wall, cfg=cfg)
    return res


# --------------------------------------------------------------- references
def _grid_mu(cfg, u):
    p = LensParams(**cfg)
    f0, fh = image_plane_flux_grid(p, n_r=1600, n_theta=24000)
    rho2 = cfg["rho"] ** 2
    return float(((1.0 - u) * f0 + u * fh) / (math.pi * rho2 * (1.0 - u / 3.0)))


def _ref_mu(cfg, u):
    p = LensParams(**cfg)
    er = H.solve_epoch(p, u)
    return float(er.mu), er.status


# --------------------------------------------------------------------- main
def _pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="random sample size")
    ap.add_argument("--quick", action="store_true", help="curated only")
    ap.add_argument("--ref", action="store_true",
                    help="also time the Python holonomic_ref solver + grid "
                         "accuracy (slow: minutes/point)")
    ap.add_argument("--json", type=str, default="",
                    help="extra copy of the raw rows (evidence/holonomic/"
                         "baseline_M0.json is always written)")
    args = ap.parse_args(argv)
    default_json = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "evidence", "holonomic",
        "baseline_M0.json"))

    if lcbinint is None:
        print(f"lcbinint unavailable: {_IMPORT_ERR!r}", file=sys.stderr)
        return 2

    os.system("uptime")
    aff = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    print(f"cpu affinity: {sorted(aff) if aff else 'n/a'}   "
          f"jax-diff backend: UNAVAILABLE (polar_epoch_directional_ffi)")

    configs = list(curated())
    if not args.quick:
        configs += random_sample(args.n)

    # skip configs the inverse-ray oracle cannot handle (source covers a
    # lens -> adaptive refinement never returns).  These are covered by the
    # holonomic solver's singular patch but have no native baseline.
    usable = [(n, c) for (n, c) in configs if not _covers_a_lens(c)]
    skipped = [n for (n, c) in configs if _covers_a_lens(c)]
    if skipped:
        print(f"skipped (source covers a lens, no inverse-ray baseline): "
              f"{skipped}")

    rows = []
    for name, cfg in usable:
        for u in U_VALUES:
            r = probe_one(name, cfg, u)
            rows.append(r)
            tag = "uniform" if u == 0.0 else f"LD u={u}"
            if r["status"] != "OK":
                print(f"  {name:16s} {tag:10s}  {r['status']}  "
                      f"({r['wall']:.1f}s)")
                continue
            acc = ""
            if args.ref:
                try:
                    g = _grid_mu(cfg, u)
                    acc = f" acc_vs_grid={abs(r['mu']-g)/abs(g):.1e}"
                except Exception as exc:
                    acc = f" acc=ERR:{exc!r}"
            print(f"  {name:16s} {tag:10s}  value={r['t_value']*1e3:6.0f}ms  "
                  f"val+jac={r['t_jac']:5.2f}s ({r['ncalls']} calls)"
                  f"  mu={r['mu']:.5f}{acc}")

    ok = [r for r in rows if r["status"] == "OK"]
    for u in U_VALUES:
        sub = [r for r in ok if r["u"] == u]
        if not sub:
            continue
        tv = [r["t_value"] for r in sub]
        tj = [r["t_jac"] for r in sub]
        tag = "uniform" if u == 0.0 else f"linear-LD (u={u})"
        print(f"\n=== {tag}   n={len(sub)} ===")
        print(f"  value-only     median {_pct(tv,50)*1e3:7.0f} ms   "
              f"p95 {_pct(tv,95)*1e3:7.0f} ms")
        print(f"  value+Jacobian median {_pct(tj,50):7.2f} s    "
              f"p95 {_pct(tj,95):7.2f} s   (central FD, 11 calls)")
        print(f"  --> M7 gate: value+Jacobian median must drop to "
              f"<= {_pct(tj,50)/2.0:.2f} s (2x), p95 <= "
              f"{_pct(tj,95)*1.25:.2f} s (+25%)")

    n_total = len(rows)
    n_fail = sum(1 for r in rows if r["status"] != "OK")
    print(f"\ncoverage: {n_total - n_fail}/{n_total} ok, "
          f"{n_fail} failed  ({[r['name']+'/'+str(r['u']) for r in rows if r['status']!='OK']})")

    if args.ref:
        print("\n=== Python holonomic_ref solver (oracle, not a target) ===")
        for name, cfg in usable[:8]:
            for u in U_VALUES:
                t0 = time.perf_counter()
                try:
                    mu, st = _ref_mu(cfg, u)
                    dt = time.perf_counter() - t0
                    print(f"  {name:16s} u={u:<4}  {dt:6.1f}s  mu={mu:.5f}  {st}")
                except Exception as exc:
                    print(f"  {name:16s} u={u:<4}  ERR {exc!r}")

    targets = [default_json] + ([args.json] if args.json else [])
    for tgt in targets:
        with open(tgt, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
        print(f"wrote {tgt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
