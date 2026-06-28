from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for build_dir in ("build_new", "build"):
    build_path = next(
        (root / build_dir
         for root in (Path.cwd(), *Path.cwd().parents)
         if (root / build_dir).is_dir()),
        None,
    )
    if build_path is not None:
        sys.path.insert(0, str(build_path))
        break

import lcbinint


@dataclass(frozen=True)
class Case:
    s: float = 0.95
    q: float = 1.0e-2
    t0: float = 0.0
    tE: float = 1.0
    u0: float = -1.0e-3
    alpha: float = 0.5
    rho: float = 5.0e-3
    t_min: float = -0.8
    t_max: float = 0.8
    n_times: int = 400


CASE = Case()
TIMES = np.linspace(CASE.t_min, CASE.t_max, CASE.n_times)

OPTIONS = lcbinint.Options(
    param_type="vbm",
    source_bins=50,
    max_source_bins=400,
    reltol=1.0e-3,
)
LIMB_DARKENING = lcbinint.LimbDarkening.linear(0.5)
TIMING_REPEATS = 7


def evaluate_lcbinint(limb_darkening: lcbinint.LimbDarkening):
    lightcurve = lcbinint.LightCurve(
        lens="binary_lens",
        options=OPTIONS,
        limb_darkening=limb_darkening,
    )
    lightcurve(
        TIMES,
        t0=CASE.t0,
        tE=CASE.tE,
        u0=CASE.u0,
        alpha=CASE.alpha,
        s=CASE.s,
        q=CASE.q,
        rho=CASE.rho,
    )
    elapsed_samples = []
    values = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        values = lightcurve(
            TIMES,
            t0=CASE.t0,
            tE=CASE.tE,
            u0=CASE.u0,
            alpha=CASE.alpha,
            s=CASE.s,
            q=CASE.q,
            rho=CASE.rho,
        )
        elapsed_samples.append(time.perf_counter() - start)
    elapsed = statistics.median(elapsed_samples)
    info = lightcurve.info(
        TIMES,
        t0=CASE.t0,
        tE=CASE.tE,
        u0=CASE.u0,
        alpha=CASE.alpha,
        s=CASE.s,
        q=CASE.q,
        rho=CASE.rho,
    )
    return lightcurve, np.asarray(values), elapsed, info, elapsed_samples


def evaluate_vbm(limb_darkening_gamma: float):
    try:
        import VBMicrolensing
    except ImportError:
        return None, np.full_like(TIMES, np.nan), np.nan

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1.0e-3
    vbm.a1 = limb_darkening_gamma
    vbm.a2 = 0.0
    params = [
        np.log(CASE.s),
        np.log(CASE.q),
        CASE.u0,
        CASE.alpha,
        np.log(CASE.rho),
        np.log(CASE.tE),
        CASE.t0,
    ]
    vbm.BinaryLightCurve(params, TIMES.tolist())
    elapsed_samples = []
    values = None
    times_list = TIMES.tolist()
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        base = vbm.BinaryLightCurve(params, times_list)
        values = np.asarray(base[0])
        elapsed_samples.append(time.perf_counter() - start)
    elapsed = statistics.median(elapsed_samples)
    return vbm, values, elapsed, elapsed_samples


def relative_error(reference, values):
    return np.abs(values - reference) / np.maximum(np.abs(reference), 1.0e-12)


def error_summary(reference, values):
    rel = relative_error(reference, values)
    return {
        "max": float(np.nanmax(rel)),
        "p99": float(np.nanpercentile(rel, 99.0)),
        "median": float(np.nanmedian(rel)),
        "rms": float(np.sqrt(np.nanmean(rel * rel))),
    }


def main():
    lightcurve, lc_no_ld, lc_no_ld_time, lc_no_ld_info, lc_no_ld_samples = evaluate_lcbinint(
        lcbinint.LimbDarkening.none()
    )
    _, lc_ld, lc_ld_time, lc_ld_info, lc_ld_samples = evaluate_lcbinint(LIMB_DARKENING)

    _, vbm_no_ld, vbm_no_ld_time, vbm_no_ld_samples = evaluate_vbm(0.0)
    _, vbm_ld, vbm_ld_time, vbm_ld_samples = evaluate_vbm(0.5)

    def ms_per_point(elapsed):
        return 1e3 * elapsed / TIMES.size

    def spread(samples):
        values = [ms_per_point(sample) for sample in samples]
        return f"median={statistics.median(values):.4f} min={min(values):.4f} max={max(values):.4f}"

    print(f"ms/point ({TIMING_REPEATS} repeats, median)")
    print(f"  lcbinint no LD: {1e3 * lc_no_ld_time / TIMES.size:.4f}")
    print(f"  lcbinint LD   : {1e3 * lc_ld_time / TIMES.size:.4f}")
    if np.isfinite(vbm_no_ld_time):
        print(f"  VBM no LD     : {1e3 * vbm_no_ld_time / TIMES.size:.4f}")
        print(f"  VBM LD        : {1e3 * vbm_ld_time / TIMES.size:.4f}")
        print("timing spread")
        print(f"  lcbinint no LD: {spread(lc_no_ld_samples)}")
        print(f"  lcbinint LD   : {spread(lc_ld_samples)}")
        print(f"  VBM no LD     : {spread(vbm_no_ld_samples)}")
        print(f"  VBM LD        : {spread(vbm_ld_samples)}")
        print("relative error vs VBM")
        for label, ref, values in [
            ("no LD", vbm_no_ld, lc_no_ld),
            ("LD", vbm_ld, lc_ld),
        ]:
            stats = error_summary(ref, values)
            print(
                f"  {label:5s} max={stats['max']:.3e} "
                f"p99={stats['p99']:.3e} median={stats['median']:.3e} "
                f"rms={stats['rms']:.3e}"
            )
    print("lcbinint method mix")
    for label, info in [("no LD", lc_no_ld_info), ("LD", lc_ld_info)]:
        counts = collections.Counter(info.finite_source_method_names)
        converged = sum(info.finite_source_converged)
        print(f"  {label:5s} {dict(counts)} converged={converged}/{TIMES.size}")

    trajectory = lightcurve.source_trajectory(
        TIMES,
        t0=CASE.t0,
        tE=CASE.tE,
        u0=CASE.u0,
        alpha=CASE.alpha,
        s=CASE.s,
        q=CASE.q,
    )
    caustics = lightcurve.caustics(s=CASE.s, q=CASE.q, n_points=900)

    fig = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.4])
    ax_mag = fig.add_subplot(grid[0])
    ax_res = fig.add_subplot(grid[1], sharex=ax_mag)
    ax_geo = fig.add_subplot(grid[2])

    ax_mag.plot(TIMES, lc_no_ld, label="lcbinint no LD", lw=1.8)
    ax_mag.plot(TIMES, lc_ld, label="lcbinint LD", lw=1.8)
    if np.all(np.isfinite(vbm_no_ld)):
        ax_mag.plot(TIMES, vbm_no_ld, "--", label="VBM no LD", lw=1.2)
        ax_mag.plot(TIMES, vbm_ld, "--", label="VBM LD", lw=1.2)
    ax_mag.set_ylabel("magnification")
    ax_mag.legend(loc="best", fontsize=8)

    if np.all(np.isfinite(vbm_no_ld)):
        ax_res.semilogy(TIMES, relative_error(vbm_no_ld, lc_no_ld), label="no LD")
        ax_res.semilogy(TIMES, relative_error(vbm_ld, lc_ld), label="LD")
        ax_res.legend(loc="best", fontsize=8)
    else:
        ax_res.text(0.5, 0.5, "VBM is not installed", ha="center", va="center")
    ax_res.set_ylabel("relative error")
    ax_res.set_xlabel("time")

    for xs, ys in zip(caustics.x, caustics.y):
        ax_geo.plot(xs, ys, color="black", lw=0.8)
    ax_geo.plot(trajectory.x, trajectory.y, color="tab:blue", lw=1.5)
    ax_geo.set_aspect("equal", adjustable="datalim")
    ax_geo.set_xlabel("source x")
    ax_geo.set_ylabel("source y")

    output = Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=160)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
