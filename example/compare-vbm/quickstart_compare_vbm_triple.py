from __future__ import annotations

import cmath
import collections
import math
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
    s: float = 1.2
    q: float = 1.0e-2
    q2: float = 1.0e-3
    sep2: float = 0.2
    ang: float = 0.0
    t0: float = 0.0
    tE: float = 1.0
    u0: float = -0.05
    alpha: float = 0.0
    rho: float = 5.0e-3
    t_min: float = -0.8
    t_max: float = 0.8
    n_times: int = 400


CASE = Case()
TIMES = np.linspace(CASE.t_min, CASE.t_max, CASE.n_times)

OPTIONS = lcbinint.Options(
    coordinates="vbm",
    source_bins=50,
    max_source_bins=400,
    reltol=1.0e-3,
)
LIMB_DARKENING = lcbinint.LimbDarkening.linear(0.5)
TIMING_REPEATS = 7


def _lcbi_params(case: Case) -> dict:
    return {
        "t0": case.t0,
        "tE": case.tE,
        "u0": case.u0,
        "alpha": case.alpha,
        "s": case.s,
        "q": case.q,
        "q2": case.q2,
        "sep2": case.sep2,
        "ang": case.ang,
        "rho": case.rho,
    }


def _vbm_setup(case: Case) -> tuple[list, list]:
    eps2 = case.q / (1 + case.q + case.q2)
    eps3 = case.q2 / (1 + case.q + case.q2)
    eps1 = 1.0 - eps2 - eps3
    eps4 = eps2 + eps3

    z1 = complex(-eps4 * case.s, 0.0)
    z2 = complex(
        eps1 * case.s + eps3 / eps4 * case.sep2 * math.cos(case.ang),
        eps3 / eps4 * case.sep2 * math.sin(case.ang),
    )
    z3 = complex(
        eps1 * case.s - eps2 / eps4 * case.sep2 * math.cos(case.ang),
        -eps2 / eps4 * case.sep2 * math.sin(case.ang),
    )
    v12 = z2 - z1
    v13 = z3 - z1
    a12 = math.atan2(v12.imag, v12.real)
    psi = math.atan2(v13.imag, v13.real) - a12
    com12 = (eps1 * z1 + eps2 * z2) / (eps1 + eps2)

    D = com12 * cmath.exp(-1j * case.alpha)
    u0_vbm = -case.u0 + D.imag
    alpha_vbm = case.alpha - a12
    times_vbm = (-(TIMES - case.t0) / case.tE + D.real).tolist()

    params = [
        math.log(abs(v12)),
        math.log(case.q),
        u0_vbm,
        alpha_vbm,
        math.log(case.rho),
        0.0,
        0.0,
        math.log(abs(v13)),
        math.log(case.q2),
        psi,
    ]
    return params, times_vbm


def evaluate_lcbinint(limb_darkening: lcbinint.LimbDarkening):
    lightcurve = lcbinint.LightCurve(
        lens="triple_lens",
        options=OPTIONS,
        limb_darkening=limb_darkening,
    )
    p = _lcbi_params(CASE)
    lightcurve(TIMES, p)
    elapsed_samples = []
    values = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        values = lightcurve(TIMES, p)
        elapsed_samples.append(time.perf_counter() - start)
    elapsed = statistics.median(elapsed_samples)
    info = lightcurve.info(TIMES, p)
    return lightcurve, np.asarray(values), elapsed, info, elapsed_samples


def evaluate_vbm(limb_darkening_gamma: float):
    try:
        import VBMicrolensing
    except ImportError:
        return None, np.full_like(TIMES, np.nan), np.nan, []

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1.0e-3
    vbm.a1 = limb_darkening_gamma
    vbm.a2 = 0.0
    vbm_params, times_vbm = _vbm_setup(CASE)
    vbm.TripleLightCurve(vbm_params, times_vbm)
    elapsed_samples = []
    values = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        result = vbm.TripleLightCurve(vbm_params, times_vbm)
        values = np.asarray(result[0])
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

    print("limb-darkened finite-source light curve")
    print(f"lcbinint: {ms_per_point(lc_no_ld_time):.4f} ms/point")
    if np.isfinite(vbm_no_ld_time):
        print(f"VBM     : {ms_per_point(vbm_no_ld_time):.4f} ms/point")
        print("timing spread")
        print(f"  lcbinint: {spread(lc_no_ld_samples)}")
        print(f"  VBM     : {spread(vbm_no_ld_samples)}")
        print(
            "relative error vs VBM: "
            f"max={error_summary(vbm_no_ld, lc_no_ld)['max']:.3e}, "
            f"p99={error_summary(vbm_no_ld, lc_no_ld)['p99']:.3e}, "
            f"median={error_summary(vbm_no_ld, lc_no_ld)['median']:.3e}, "
            f"rms={error_summary(vbm_no_ld, lc_no_ld)['rms']:.3e}"
        )
    print("lcbinint method mix")
    for label, info in [("no LD", lc_no_ld_info), ("LD", lc_ld_info)]:
        counts = collections.Counter(info.finite_source_method_names)
        converged = sum(info.finite_source_converged)
        print(f"  {label:5s} {dict(counts)} converged={converged}/{TIMES.size}")

    p = _lcbi_params(CASE)
    trajectory = lightcurve.source_trajectory(TIMES, p)
    caustics = lightcurve.caustics(p, n_points=900)

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
