"""Triple lens finalize diagnostic: VBM finite comparison, accuracy sweep, speed benchmark."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build_new"))
import lcbinint

try:
    import VBMicrolensing
    HAS_VBM = True
except ImportError:
    HAS_VBM = False


# ---------------------------------------------------------------------------
# VBM geometry helpers (legacy_edges convention — identical to triple_reference_compare.py)
# ---------------------------------------------------------------------------

def _old_lens_positions(s, q, q2, sep2, ang):
    eps2 = q / (1.0 + q + q2)
    eps3 = q2 / (1.0 + q + q2)
    eps1 = 1.0 - eps2 - eps3
    eps4 = eps2 + eps3
    z1 = complex(-eps4 * s, 0.0)
    z2 = complex(eps1 * s + eps3 / eps4 * sep2 * math.cos(ang),
                  eps3 / eps4 * sep2 * math.sin(ang))
    z3 = complex(eps1 * s - eps2 / eps4 * sep2 * math.cos(ang),
                 -eps2 / eps4 * sep2 * math.sin(ang))
    return z1, z2, z3


def vbm_triple_finite(sx, sy, s, q, q2, sep2, ang, rho, tol=1e-5):
    """Call VBM TripleLightCurve using legacy_edges convention (matches triple_reference_compare.py)."""
    if not HAS_VBM:
        return None
    z1, z2, z3 = _old_lens_positions(s, q, q2, sep2, ang)
    v12 = z2 - z1
    v13 = z3 - z1
    angle12 = math.atan2(v12.imag, v12.real)
    psi = math.atan2(v13.imag, v13.real) - angle12
    eps2 = q / (1.0 + q + q2)
    eps1 = 1.0 - eps2 - q2 / (1.0 + q + q2)
    com12 = (eps1 * z1 + eps2 * z2) / (eps1 + eps2)
    source = complex(sx, sy)
    vbm_source = (source - com12) * complex(math.cos(-angle12), math.sin(-angle12))
    geom = [
        math.log(abs(v12)),
        math.log(q),
        math.log(max(rho, 1e-12)),
        0.0,
        0.0,
        math.log(abs(v13)),
        math.log(q2),
        psi,
    ]
    params = [
        geom[0], geom[1],
        -vbm_source.imag,  # u0 in VBM convention
        0.0,               # alpha
        geom[2], geom[3], geom[4], geom[5], geom[6], geom[7],
    ]
    vbb = VBMicrolensing.VBMicrolensing()
    vbb.Tol = tol
    vbb.RelTol = 0.0
    result = vbb.TripleLightCurve(params, [-vbm_source.real])
    return float(np.asarray(result[0], dtype=float)[0])


def lcbi_finite(sx, sy, s, q, q2, sep2, ang, rho, source_bins=64, caustic_bins=1400,
                inverse_ray_grid="auto"):
    lc = lcbinint.LightCurve(
        lens="triple_lens",
        options=lcbinint.Options(
            source_bins=source_bins,
            caustic_bins=caustic_bins,
            inverse_ray_grid=inverse_ray_grid,
        ),
    )
    params = {
        "t0": 0.0, "tE": 1.0, "u0": sy, "alpha": 0.0,
        "s": s, "q": q, "q2": q2, "sep2": sep2, "ang": ang, "rho": rho,
    }
    info = lc.info(np.array([sx]), params)
    return float(info.magnifications[0]), info.finite_source_method_names[0]


# ---------------------------------------------------------------------------
# Section 1: VBM finite-source comparison
# ---------------------------------------------------------------------------

FINITE_CASES = [
    # (name, sx, sy, s, q, q2, sep2, ang, rho)
    ("planetary_high_mag",    0.0,   0.008,  1.2, 1e-3, 1e-3, 0.1,  0.0,  2e-3),
    ("planetary_left",       -0.09,  0.0,    1.0, 1e-3, 1e-4, 0.5,  1.2,  2e-3),
    ("planetary_right",       0.17,  0.1,    1.0, 1e-3, 1e-4, 0.5,  1.2,  5e-3),
    ("hex_off_caustic",      -0.2,   0.0,    1.0, 1e-3, 1e-4, 0.5,  1.2,  5e-3),
    ("moderate_finite",       0.35, -0.22,   0.8, 0.03, 0.02, 0.35, -0.7, 2e-3),
    ("wide_finite",          -0.45,  0.18,   1.4, 0.2,  0.05, 0.7,  2.1,  3e-3),
]


def section_vbm():
    print("=" * 75)
    print("Section 1: VBM finite-source comparison (source_bins=64, vbm_tol=1e-6)")
    print("=" * 75)
    if not HAS_VBM:
        print("VBMicrolensing not available — skipping")
        return
    print(f"{'name':<25} {'method':<22} {'lcbi_mag':>12} {'vbm_mag':>12} {'rel':>10}")
    print("-" * 85)
    for (name, sx, sy, s, q, q2, sep2, ang, rho) in FINITE_CASES:
        val, method = lcbi_finite(sx, sy, s, q, q2, sep2, ang, rho)
        try:
            vbm = vbm_triple_finite(sx, sy, s, q, q2, sep2, ang, rho, tol=1e-6)
        except Exception as e:
            print(f"{name:<25} {method:<22} {val:>12.6f} {'VBM_ERR':>12}  ({e})")
            continue
        if vbm is None:
            print(f"{name:<25} {method:<22} {val:>12.6f} {'n/a':>12} {'n/a':>10}")
        else:
            rel = val / vbm - 1.0
            print(f"{name:<25} {method:<22} {val:>12.6f} {vbm:>12.6f} {rel:>10.3e}")


# ---------------------------------------------------------------------------
# Section 2: Accuracy sweep
# ---------------------------------------------------------------------------

def section_accuracy():
    print("\n" + "=" * 75)
    print("Section 2: Accuracy sweep — source_bins convergence")
    print("Reference = Cartesian source_bins=256")
    print("=" * 75)

    acc_cases = [
        ("high_mag u0=0.008 rho=0.002", 0.0, 0.008, 1.2, 1e-3, 1e-3, 0.1, 0.0, 2e-3),
        ("moderate u0=0.1   rho=0.01",  0.0, 0.1,   1.0, 1e-3, 1e-4, 0.5, 1.2, 1e-2),
        ("wide     u0=0.0   rho=5e-4",  0.0, 0.0,   2.0, 1e-3, 1e-4, 0.1, 0.0, 5e-4),
    ]

    for (label, sx, sy, s, q, q2, sep2, ang, rho) in acc_cases:
        ref, _ = lcbi_finite(sx, sy, s, q, q2, sep2, ang, rho,
                             source_bins=256, inverse_ray_grid="cartesian")
        print(f"\n  {label}  (ref={ref:.8f})")
        print(f"  {'bins':>5}  {'auto mag':>14}  {'cart mag':>14}  {'auto rel':>10}  {'cart rel':>10}  {'method'}")
        for bins in [8, 16, 32, 64, 128]:
            val_a, meth = lcbi_finite(sx, sy, s, q, q2, sep2, ang, rho,
                                      source_bins=bins, inverse_ray_grid="auto")
            val_c, _   = lcbi_finite(sx, sy, s, q, q2, sep2, ang, rho,
                                      source_bins=bins, inverse_ray_grid="cartesian")
            rel_a = val_a / ref - 1.0
            rel_c = val_c / ref - 1.0
            print(f"  {bins:>5}  {val_a:>14.6f}  {val_c:>14.6f}  {rel_a:>10.3e}  {rel_c:>10.3e}  {meth}")


# ---------------------------------------------------------------------------
# Section 3: Speed benchmark
# ---------------------------------------------------------------------------

def _bench(lc, params_dict, N=40, warmup=5):
    t = np.array([0.0])
    for _ in range(warmup):
        lc.info(t, params_dict)
    t0 = time.perf_counter()
    for _ in range(N):
        lc.info(t, params_dict)
    return (time.perf_counter() - t0) / N * 1e3


def section_speed():
    print("\n" + "=" * 75)
    print("Section 3: Speed benchmark (ms/point, N=40 each)")
    print("=" * 75)

    lc_auto = lcbinint.LightCurve(lens="triple_lens",
                                   options=lcbinint.Options(inverse_ray_grid="auto"))
    lc_cart = lcbinint.LightCurve(lens="triple_lens",
                                   options=lcbinint.Options(inverse_ray_grid="cartesian"))
    lc_pol  = lcbinint.LightCurve(lens="triple_lens",
                                   options=lcbinint.Options(inverse_ray_grid="polar"))
    lc_bin  = lcbinint.LightCurve(lens="binary_lens")

    triple_cases = [
        ("near-caustic  rho=2e-3",
         {"t0":0,"tE":1,"u0":0.008,"alpha":0,"s":1.2,"q":1e-3,"q2":1e-3,"sep2":0.1,"ang":0,"rho":2e-3}),
        ("off-caustic   rho=5e-4 (auto→polar)",
         {"t0":0,"tE":1,"u0":0.008,"alpha":0,"s":1.2,"q":1e-3,"q2":1e-3,"sep2":0.1,"ang":0,"rho":5e-4}),
        ("moderate      rho=1e-2",
         {"t0":0,"tE":1,"u0":0.1,"alpha":0,"s":1.0,"q":1e-3,"q2":1e-4,"sep2":0.5,"ang":1.2,"rho":1e-2}),
        ("wide off-caus rho=5e-4",
         {"t0":0,"tE":1,"u0":0.003,"alpha":0,"s":2.0,"q":1e-3,"q2":1e-4,"sep2":0.1,"ang":0,"rho":5e-4}),
    ]

    print(f"\n  {'case':<38} {'auto':>8} {'cart':>8} {'polar':>8}  {'auto_method'}")
    print("  " + "-" * 80)
    for label, params in triple_cases:
        ta = _bench(lc_auto, params)
        tc = _bench(lc_cart, params)
        tp = _bench(lc_pol,  params)
        info = lc_auto.info(np.array([0.0]), params)
        meth = info.finite_source_method_names[0]
        print(f"  {label:<38} {ta:>7.2f}  {tc:>7.2f}  {tp:>7.2f}  {meth}")

    print(f"\n  Binary equivalent (no q2)")
    print(f"  {'case':<38} {'binary':>8}  {'binary_method'}")
    print("  " + "-" * 60)
    for label, params in triple_cases[:2]:
        params_b = {k: params[k] for k in ("t0","tE","u0","alpha","s","q","rho")}
        tb = _bench(lc_bin, params_b)
        info_b = lc_bin.info(np.array([0.0]), params_b)
        meth_b = info_b.finite_source_method_names[0]
        print(f"  {label:<38} {tb:>7.2f}  {meth_b}")


if __name__ == "__main__":
    section_vbm()
    section_accuracy()
    section_speed()
