"""
Triple-lens seed quality diagnostic.

For each source position along a trajectory that crosses or grazes the caustic:
  - compute magnification at multiple bin counts (12, 24, 64, 128)
  - record seed count (image_count), error estimate
  - flag points where convergence is slow, non-monotone, or jumpy
  - compute second-difference of mag curve to detect discontinuities (missing images)
  - optionally compare against VBM finite (disabled by default to avoid crashes)

Typical usage:
    PYTHONPATH=build_new python tests/diagnostics/triple_seed_diagnostic.py
    PYTHONPATH=build_new python tests/diagnostics/triple_seed_diagnostic.py --vbm-finite
    PYTHONPATH=build_new python tests/diagnostics/triple_seed_diagnostic.py --case resonant_small_source
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    s: float
    q: float
    q2: float
    sep2: float
    ang: float
    rho: float
    x_min: float
    x_max: float
    y: float = 0.0
    points: int = 51


CASES = {
    "resonant_tiny_source": Case(
        "resonant_tiny_source",
        s=1.0, q=1.0e-3, q2=1.0e-4, sep2=0.5, ang=1.2,
        rho=1.0e-3, x_min=-0.003, x_max=0.003, y=0.0, points=51,
    ),
    "resonant_small_source": Case(
        "resonant_small_source",
        s=1.0, q=1.0e-3, q2=1.0e-4, sep2=0.5, ang=1.2,
        rho=3.0e-3, x_min=-0.010, x_max=0.010, y=0.0, points=51,
    ),
    "moderate_inner_pair": Case(
        "moderate_inner_pair",
        s=0.8, q=0.03, q2=0.02, sep2=0.35, ang=-0.7,
        rho=2.0e-3, x_min=0.28, x_max=0.42, y=-0.22, points=51,
    ),
}


def load_lcbinint(extension: Path | None):
    if extension is None:
        import lcbinint as module
        return module
    spec = importlib.util.spec_from_file_location("lcbinint", extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lcbinint extension from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lcbinint"] = module
    spec.loader.exec_module(module)
    return module


def evaluate_sweep(
    lcbinint,
    case: Case,
    source_bins: int,
    caustic_bins: int,
    xs: np.ndarray,
) -> dict:
    options = lcbinint.Options(source_bins=source_bins, caustic_bins=caustic_bins)
    curve = lcbinint.LightCurve(lens="triple_lens", options=options)
    params = {
        "t0": 0.0, "tE": 1.0,
        "u0": case.y, "alpha": 0.0,
        "s": case.s, "q": case.q, "q2": case.q2,
        "sep2": case.sep2, "ang": case.ang, "rho": case.rho,
    }
    info = curve.info(xs.tolist(), params)
    return {
        "mag": np.asarray(info.magnifications, dtype=float),
        "seeds": np.asarray(info.image_counts, dtype=int),
        "err": np.asarray(info.finite_source_error_estimates, dtype=float),
        "methods": list(info.finite_source_method_names),
    }


def old_lens_positions(case: Case) -> list[complex]:
    eps2 = case.q / (1.0 + case.q + case.q2)
    eps3 = case.q2 / (1.0 + case.q + case.q2)
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
    return [z1, z2, z3]


def vbm_finite_sweep(
    case: Case, xs: np.ndarray, tol: float
) -> np.ndarray | None:
    """Return VBM finite magnification array, or None on failure."""
    try:
        import VBMicrolensing
    except ImportError:
        print("  [vbm] VBMicrolensing not available", file=sys.stderr)
        return None

    z1, z2, z3 = old_lens_positions(case)
    v12 = z2 - z1
    v13 = z3 - z1
    angle12 = math.atan2(v12.imag, v12.real)
    psi = math.atan2(v13.imag, v13.real) - angle12
    eps2 = case.q / (1.0 + case.q + case.q2)
    eps1 = 1.0 - eps2 - case.q2 / (1.0 + case.q + case.q2)
    com12 = (eps1 * z1 + eps2 * z2) / (eps1 + eps2)

    # VBM geometry params: [log(s12), log(q2), u0, alpha, log(rho), log(tE), t0, log(s13), log(q3), psi]
    s12 = abs(v12)
    s13 = abs(v13)
    q2_vbm = case.q      # mass ratio lens2/lens1 in VBM
    q3_vbm = case.q2     # mass ratio lens3/lens1 in VBM

    vbm_mags = []
    failed = []
    for i, x in enumerate(xs):
        source = complex(x, case.y)
        vbm_source = (source - com12) * complex(math.cos(-angle12), math.sin(-angle12))
        params = [
            math.log(s12),
            math.log(q2_vbm),
            -vbm_source.imag,  # u0
            0.0,               # alpha
            math.log(max(case.rho, 1.0e-12)),
            0.0,               # log(tE)
            0.0,               # t0
            math.log(s13),
            math.log(q3_vbm),
            psi,
        ]
        try:
            vbb = VBMicrolensing.VBMicrolensing()
            vbb.Tol = tol
            vbb.RelTol = 0.0
            result = vbb.TripleLightCurve(params, [-vbm_source.real])
            val = float(np.asarray(result[0], dtype=float)[0])
            vbm_mags.append(val)
        except Exception as exc:
            failed.append((i, x, str(exc)))
            vbm_mags.append(float("nan"))

    if failed:
        print(
            f"  [vbm] {len(failed)}/{len(xs)} points failed "
            f"(first: x={failed[0][1]:.5f}: {failed[0][2][:60]})",
            file=sys.stderr,
        )
    return np.array(vbm_mags)


def second_difference(arr: np.ndarray) -> np.ndarray:
    """Central second difference: |a[i-1] - 2*a[i] + a[i+1]| for interior points."""
    d2 = np.full_like(arr, np.nan)
    if len(arr) >= 3:
        d2[1:-1] = np.abs(arr[:-2] - 2.0 * arr[1:-1] + arr[2:])
    return d2


def rel(a: float, b: float) -> float:
    if b == 0.0 or not math.isfinite(b):
        return float("nan")
    return (a - b) / abs(b)


def bins_convergence_at_point(
    lcbinint,
    case: Case,
    x: float,
    caustic_bins: int,
    bins_list: list[int],
) -> None:
    """Print a convergence table at a single source position."""
    print(f"\n  bins convergence at x={x:.6f} y={case.y:.6f}")
    params = {
        "t0": 0.0, "tE": 1.0,
        "u0": case.y, "alpha": 0.0,
        "s": case.s, "q": case.q, "q2": case.q2,
        "sep2": case.sep2, "ang": case.ang, "rho": case.rho,
    }
    prev_mag = None
    for bins in bins_list:
        options = lcbinint.Options(source_bins=bins, caustic_bins=caustic_bins)
        curve = lcbinint.LightCurve(lens="triple_lens", options=options)
        info = curve.info([x], params)
        mag = float(info.magnifications[0])
        seeds = int(info.image_counts[0])
        method = info.finite_source_method_names[0]
        delta = f"  rel_prev={rel(mag, prev_mag):+.4e}" if prev_mag is not None else ""
        print(f"    bins={bins:4d}  mag={mag:.8f}  seeds={seeds:3d}  {method}{delta}")
        prev_mag = mag


def flag_summary(flags: list[tuple]) -> None:
    if not flags:
        print("  (no flagged points)")
        return
    for x, reason, extra in flags:
        print(f"    x={x:.6f}  {reason}  {extra}")


def run_case(
    lcbinint,
    case: Case,
    caustic_bins: int,
    bins_fast: int,
    bins_mid: int,
    bins_hi: int,
    bins_ref: int,
    vbm_finite: bool,
    vbm_tol: float,
    verbose: bool,
) -> None:
    print(f"\n{'='*70}")
    print(f"case={case.name}  rho={case.rho:.2e}  y={case.y}")
    xs = np.linspace(case.x_min, case.x_max, case.points)

    # Evaluate at multiple bin counts
    print(f"  evaluating at bins={bins_fast},{bins_mid},{bins_hi},{bins_ref} ...")
    r_fast = evaluate_sweep(lcbinint, case, bins_fast, caustic_bins, xs)
    r_mid  = evaluate_sweep(lcbinint, case, bins_mid,  caustic_bins, xs)
    r_hi   = evaluate_sweep(lcbinint, case, bins_hi,   caustic_bins, xs)
    r_ref  = evaluate_sweep(lcbinint, case, bins_ref,  caustic_bins, xs)

    mag_fast = r_fast["mag"]
    mag_mid  = r_mid["mag"]
    mag_hi   = r_hi["mag"]
    mag_ref  = r_ref["mag"]

    seeds_fast = r_fast["seeds"]
    seeds_hi   = r_hi["seeds"]

    # Reference-relative errors
    rel_fast = np.abs(mag_fast - mag_ref) / np.maximum(np.abs(mag_ref), 1.0e-12)
    rel_mid  = np.abs(mag_mid  - mag_ref) / np.maximum(np.abs(mag_ref), 1.0e-12)
    rel_hi   = np.abs(mag_hi   - mag_ref) / np.maximum(np.abs(mag_ref), 1.0e-12)

    # Hi vs ref relative error (should be small if converged)
    rel_hi_ref = np.abs(mag_hi - mag_ref) / np.maximum(np.abs(mag_ref), 1.0e-12)

    # Seed count range
    seeds_min, seeds_max = int(seeds_hi.min()), int(seeds_hi.max())
    seed_jumps = np.where(np.abs(np.diff(seeds_hi.astype(float))) >= 2)[0]

    # Smoothness: second-difference on reference curve, normalized by mag
    d2_ref = second_difference(mag_ref) / np.maximum(np.abs(mag_ref), 1.0e-12)

    # Methods used (hi)
    methods_set = sorted(set(r_hi["methods"]))

    print(f"  methods(bins={bins_hi}): {', '.join(methods_set)}")
    print(f"  seeds(bins={bins_hi}): min={seeds_min} max={seeds_max}")
    print(f"  max_rel vs ref(={bins_ref}):  "
          f"bins{bins_fast}={rel_fast.max():.4e}  "
          f"bins{bins_mid}={rel_mid.max():.4e}  "
          f"bins{bins_hi}={rel_hi.max():.4e}")
    worst_hi_ref = rel_hi_ref.max()
    print(f"  convergence bins{bins_hi} vs {bins_ref}: max_rel={worst_hi_ref:.4e}")

    # Flag slow convergence at hi vs ref
    slow_conv_threshold = 0.02
    flags_slow = []
    for i, x in enumerate(xs):
        if rel_hi_ref[i] > slow_conv_threshold:
            flags_slow.append((x, f"slow conv bins{bins_hi}vs{bins_ref}", f"rel={rel_hi_ref[i]:.4e} mag={mag_ref[i]:.4f} seeds={seeds_hi[i]}"))
    if flags_slow:
        print(f"\n  SLOW CONVERGENCE (rel>{slow_conv_threshold:.0%} between bins={bins_hi} and {bins_ref}):")
        flag_summary(flags_slow)

    # Flag seed count jumps
    if len(seed_jumps) > 0:
        print(f"\n  SEED COUNT JUMPS (|Δseeds|>=2 between adjacent points):")
        for ji in seed_jumps[:10]:
            x_a, x_b = xs[ji], xs[ji + 1]
            s_a, s_b = seeds_hi[ji], seeds_hi[ji + 1]
            print(f"    x={x_a:.6f}→{x_b:.6f}  seeds {s_a}→{s_b}")

    # Flag large second-difference on reference curve (discontinuity)
    smooth_threshold = 0.05
    d2_flags = np.where(d2_ref[1:-1] > smooth_threshold)[0] + 1
    if len(d2_flags) > 0:
        print(f"\n  SMOOTHNESS FLAGS (|d2 mag/mag|>{smooth_threshold:.0%} in ref curve):")
        for i in d2_flags[:10]:
            print(f"    x={xs[i]:.6f}  d2_rel={d2_ref[i]:.4e}  mag={mag_ref[i]:.4f}  seeds={seeds_hi[i]}")

    # Flag non-monotone convergence: fast→mid→hi should all be decreasing in rel error
    non_mono = []
    for i, x in enumerate(xs):
        if rel_mid[i] > rel_fast[i] * 1.5 and rel_fast[i] > 1.0e-4:
            non_mono.append((x, "non-mono fast→mid", f"rel_fast={rel_fast[i]:.4e} rel_mid={rel_mid[i]:.4e}"))
        elif rel_hi[i] > rel_mid[i] * 1.5 and rel_mid[i] > 1.0e-4:
            non_mono.append((x, "non-mono mid→hi", f"rel_mid={rel_mid[i]:.4e} rel_hi={rel_hi[i]:.4e}"))
    if non_mono:
        print(f"\n  NON-MONOTONE CONVERGENCE:")
        flag_summary(non_mono[:10])

    # VBM finite comparison
    if vbm_finite:
        print(f"\n  VBM finite comparison (tol={vbm_tol:.1e}) ...")
        vbm_mag = vbm_finite_sweep(case, xs, vbm_tol)
        if vbm_mag is not None:
            valid = np.isfinite(vbm_mag) & np.isfinite(mag_ref)
            if valid.any():
                rel_vbm = (mag_ref[valid] - vbm_mag[valid]) / np.maximum(np.abs(vbm_mag[valid]), 1.0e-12)
                print(f"  vs VBM (bins_ref={bins_ref}): max_rel={np.abs(rel_vbm).max():.4e}  rms={np.sqrt(np.mean(rel_vbm**2)):.4e}  n_valid={valid.sum()}/{len(xs)}")
                # Large VBM discrepancy at specific points
                vbm_flag_threshold = 0.05
                for i, x in enumerate(xs):
                    if valid[i] and abs(rel_vbm[list(np.where(valid)[0]).index(i)] if i in np.where(valid)[0] else 0.0) > vbm_flag_threshold:
                        pass
                # simpler: iterate
                vi_list = np.where(valid)[0]
                for k, vi in enumerate(vi_list):
                    if abs(rel_vbm[k]) > vbm_flag_threshold:
                        print(f"    VBM discrepancy x={xs[vi]:.6f}  lcb={mag_ref[vi]:.6f}  vbm={vbm_mag[vi]:.6f}  rel={rel_vbm[k]:+.4e}")
            else:
                print("  vs VBM: no valid comparison points")

    # Verbose: print per-point table
    if verbose:
        print(f"\n  per-point table (x, mag_{bins_ref}, rel_{bins_fast}, rel_{bins_hi}, seeds_{bins_hi}):")
        print(f"  {'x':>10}  {'mag_ref':>12}  {'rel_f':>9}  {'rel_h':>9}  {'seeds':>6}  {'d2_rel':>9}")
        for i, x in enumerate(xs):
            print(
                f"  {x:10.6f}  {mag_ref[i]:12.4f}  {rel_fast[i]:+9.4e}  {rel_hi[i]:+9.4e}  "
                f"{seeds_hi[i]:6d}  {d2_ref[i]:9.4e}"
            )

    # Convergence drill-down at worst point (by hi-vs-ref error)
    worst_i = int(np.argmax(rel_hi_ref))
    worst_x = xs[worst_i]
    if rel_hi_ref[worst_i] > 1.0e-3:
        bins_drill = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
        bins_drill = [b for b in bins_drill if b <= 256]
        bins_convergence_at_point(
            lcbinint, case, worst_x, caustic_bins, bins_drill
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, default=None)
    parser.add_argument("--case", choices=list(CASES.keys()), default=None,
                        help="Run only this case (default: all)")
    parser.add_argument("--caustic-bins", type=int, default=1400)
    parser.add_argument("--bins-fast", type=int, default=12)
    parser.add_argument("--bins-mid", type=int, default=24)
    parser.add_argument("--bins-hi", type=int, default=64)
    parser.add_argument("--bins-ref", type=int, default=128)
    parser.add_argument("--vbm-finite", action="store_true",
                        help="Compare against VBM finite source (may crash for some points)")
    parser.add_argument("--vbm-tol", type=float, default=1.0e-4)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-point table")
    args = parser.parse_args()

    lcbinint = load_lcbinint(args.extension)
    cases = [CASES[args.case]] if args.case else list(CASES.values())

    for case in cases:
        run_case(
            lcbinint=lcbinint,
            case=case,
            caustic_bins=args.caustic_bins,
            bins_fast=args.bins_fast,
            bins_mid=args.bins_mid,
            bins_hi=args.bins_hi,
            bins_ref=args.bins_ref,
            vbm_finite=args.vbm_finite,
            vbm_tol=args.vbm_tol,
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
