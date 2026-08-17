from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_lcbinint():
    """Load the in-tree build even when an editable install is active."""

    build_path = next(
        (root / "build"
         for root in (Path.cwd(), *Path.cwd().parents)
         if (root / "build").is_dir()),
        None,
    )
    if build_path is None:
        return importlib.import_module("lcbinint")

    package = build_path / "lcbinint"
    root = package / "__init__.py"
    extensions = sorted(package.glob("_lcbinint*.so"))
    if not root.is_file() or not extensions:
        raise FileNotFoundError(
            f"in-tree build is incomplete: expected {root} and _lcbinint*.so"
        )

    # The development environment can install a different lcbinint through
    # an editable-package finder.  Loading both modules from this build keeps
    # the example reproducible and makes the warm-up API check meaningful.
    sys.modules.pop("lcbinint._lcbinint", None)
    sys.modules.pop("lcbinint", None)
    root_spec = importlib.util.spec_from_file_location(
        "lcbinint",
        root,
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(root_spec)
    sys.modules["lcbinint"] = module
    extension_spec = importlib.util.spec_from_file_location(
        "lcbinint._lcbinint", extensions[0]
    )
    extension = importlib.util.module_from_spec(extension_spec)
    sys.modules["lcbinint._lcbinint"] = extension
    extension_spec.loader.exec_module(extension)
    root_spec.loader.exec_module(module)
    return module


lcbinint = load_lcbinint()


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
    coordinates="vbm",
    nbin="auto",
    tol=1.0e-3,
    reltol=1.0e-3,
)
LIMB_DARKENING = lcbinint.LimbDarkening.linear(0.5)
RELATIVE_TOLERANCE = 1.0e-3
TIMING_REPEATS = 7


def light_curve_parameters():
    return {
        "t0": CASE.t0,
        "tE": CASE.tE,
        "u0": CASE.u0,
        "alpha": CASE.alpha,
        "s": CASE.s,
        "q": CASE.q,
        "rho": CASE.rho,
    }


def evaluate_lcbinint(limb_darkening: lcbinint.LimbDarkening):
    lightcurve = lcbinint.LightCurve(
        lens="binary",
        options=OPTIONS,
        limb_darkening=limb_darkening,
    )
    params = light_curve_parameters()

    # This quickstart intentionally does not call curve.warmup().  The first
    # ordinary LightCurve call is part of the timing samples below.
    elapsed_samples = []
    values = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        values = lightcurve(TIMES, params)
        elapsed_samples.append(time.perf_counter() - start)
    elapsed = statistics.median(elapsed_samples)
    return {
        "curve": lightcurve,
        "values": np.asarray(values),
        "elapsed": elapsed,
        "samples": elapsed_samples,
    }


def evaluate_vbm(limb_darkening_gamma: float):
    try:
        import VBMicrolensing
    except ImportError:
        return {
            "values": np.full_like(TIMES, np.nan),
            "elapsed": np.nan,
            "samples": [],
            "warmup_elapsed": np.nan,
        }

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = RELATIVE_TOLERANCE
    vbm.RelTol = RELATIVE_TOLERANCE
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
    times_list = TIMES.tolist()

    # No separate VBM warm-up: the first complete BinaryLightCurve call is
    # included in the timing samples, matching the lcbinint measurement.
    elapsed_samples = []
    values = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        base = vbm.BinaryLightCurve(params, times_list)
        values = np.asarray(base[0])
        elapsed_samples.append(time.perf_counter() - start)
    elapsed = statistics.median(elapsed_samples)
    return {
        "engine": vbm,
        "values": values,
        "elapsed": elapsed,
        "samples": elapsed_samples,
    }


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
    lc_no_ld = evaluate_lcbinint(lcbinint.LimbDarkening.none())
    lc_ld = evaluate_lcbinint(LIMB_DARKENING)
    vbm_no_ld = evaluate_vbm(0.0)
    vbm_ld = evaluate_vbm(0.5)

    def ms_per_point(elapsed):
        return 1e3 * elapsed / TIMES.size

    def spread(samples):
        if not samples:
            return "unavailable"
        values = [ms_per_point(sample) for sample in samples]
        return f"median={statistics.median(values):.4f} min={min(values):.4f} max={max(values):.4f}"

    def speed_ratio(vbm_record, lc_record):
        if not np.isfinite(vbm_record["elapsed"]):
            return np.nan
        return vbm_record["elapsed"] / lc_record["elapsed"]

    print(f"light-curve timing without explicit warm-up ({TIMING_REPEATS} repeats, median)")
    for label, record in [
        ("lcbinint uniform", lc_no_ld),
        ("lcbinint LD     ", lc_ld),
        ("VBM no LD       ", vbm_no_ld),
        ("VBM LD          ", vbm_ld),
    ]:
        print(
            f"  {label}: total={1e3 * record['elapsed']:.3f} ms, "
            f"ms/epoch={ms_per_point(record['elapsed']):.5f}"
        )
    if np.isfinite(vbm_no_ld["elapsed"]):
        print("timing spread")
        print(f"  lcbinint uniform: {spread(lc_no_ld['samples'])}")
        print(f"  lcbinint LD     : {spread(lc_ld['samples'])}")
        print(f"  VBM uniform     : {spread(vbm_no_ld['samples'])}")
        print(f"  VBM LD          : {spread(vbm_ld['samples'])}")
        print(
            "speed ratio R=t_VBM/t_lcbinint "
            f"(R>1 means lcbinint is faster): uniform={speed_ratio(vbm_no_ld, lc_no_ld):.4f}, "
            f"LD={speed_ratio(vbm_ld, lc_ld):.4f}"
        )
        print("relative error vs VBM")
        for label, ref, values in [
            ("uniform", vbm_no_ld["values"], lc_no_ld["values"]),
            ("LD", vbm_ld["values"], lc_ld["values"]),
        ]:
            stats = error_summary(ref, values)
            print(
                f"  {label:5s} max={stats['max']:.3e} "
                f"p99={stats['p99']:.3e} median={stats['median']:.3e} "
                f"rms={stats['rms']:.3e}"
            )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.5),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    lc_color = "tab:blue"
    vbm_color = "tab:orange"
    error_color = "C0"
    profiles = [
        ("C0: uniform", vbm_no_ld, lc_no_ld, axes[0, 0], axes[1, 0]),
        ("C1: linear LD ($c=0.5$)", vbm_ld, lc_ld, axes[0, 1], axes[1, 1]),
    ]
    for label, vbm_record, lc_record, ax_mag, ax_res in profiles:
        lc_values = lc_record["values"]
        ax_mag.scatter(
            TIMES,
            lc_values,
            color=lc_color,
            s=28,
            alpha=0.9,
            zorder=3,
            label="lcbinint",
        )
        if np.all(np.isfinite(vbm_record["values"])):
            ax_mag.plot(
                TIMES,
                vbm_record["values"],
                color=vbm_color,
                lw=1.4,
                label="VBMicrolensing",
            )
            ax_res.semilogy(
                TIMES,
                relative_error(vbm_record["values"], lc_values),
                color=error_color,
                lw=1.2,
            )
            ax_res.axhline(
                RELATIVE_TOLERANCE,
                color="0.35",
                linestyle=":",
                linewidth=1.0,
            )
        else:
            ax_mag.text(0.5, 0.5, "VBM is not installed", ha="center", va="center")
        ax_mag.set_title(label)
        ax_mag.set_ylabel("magnification")
        ax_mag.legend(loc="best", fontsize=8)
        ax_res.set_title(f"{label} residual")
        ax_res.set_ylabel("relative error")
        ax_res.set_xlabel("time")
    fig.suptitle("Binary light curve: lcbinint vs VBMicrolensing")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output = Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=160)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
