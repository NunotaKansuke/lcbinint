from __future__ import annotations

import math

import numpy as np

import lcbinint


TIMES = np.linspace(2.0, 69.0, 41)

PARAMS = dict(
    t0=10.0,
    tE=math.exp(1.5),
    u0=0.01,
    alpha=0.1,
    s=0.97,
    q=10.0 ** -1.5,
    rho=0.0,
    g1=0.004,
    g2=0.011,
    g3=0.006,
    lom_szs=0.2,
    lom_ar=1.4,
)

OPTIONS = lcbinint.Options(coordinates="vbm", source_bins=50)


def relative_error(reference, values):
    return np.abs(values - reference) / np.maximum(np.abs(reference), 1.0e-12)


def lcbinint_kepler(tfix):
    curve = lcbinint.LightCurve(
        lens="binary_lens",
        event=lcbinint.EventCoordinates(tfix=tfix),
        options=OPTIONS,
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.KEPLER,
    )
    return np.asarray(curve(TIMES, **PARAMS))


def vbmicrolensing_kepler():
    try:
        import VBMicrolensing
    except ImportError:
        return None

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1.0e-5
    vbm.SetObjectCoordinates("17:45:40.04 -29:00:28.1")
    vbm.t0_par_fixed = 1
    vbm.t0_par = PARAMS["t0"]
    values = vbm.BinaryLightCurveKepler(
        [
            math.log(PARAMS["s"]),
            math.log(PARAMS["q"]),
            PARAMS["u0"],
            PARAMS["alpha"],
            math.log(1.0e-8),
            math.log(PARAMS["tE"]),
            PARAMS["t0"],
            0.0,
            0.0,
            PARAMS["g1"],
            PARAMS["g2"],
            PARAMS["g3"],
            PARAMS["lom_szs"],
            PARAMS["lom_ar"],
        ],
        TIMES.tolist(),
    )[0]
    return np.asarray(values)


def main():
    compatible = lcbinint_kepler(tfix=PARAMS["t0"])
    fixed_reference = lcbinint_kepler(tfix=7000.0)

    reference = vbmicrolensing_kepler()
    if reference is None:
        print("VBMicrolensing is not installed; skipped external comparison.")
    else:
        rel = relative_error(reference, compatible)
        print("lcbinint vs VBMicrolensing, Kepler LOM with tfix=t0")
        print(
            f"  max={np.max(rel):.3e} p99={np.percentile(rel, 99):.3e} "
            f"median={np.median(rel):.3e}"
        )

    delta = relative_error(compatible, fixed_reference)
    print("effect of using a fixed lcbinint LOM reference epoch")
    print(f"  tfix=t0 vs tfix=7000: max relative difference={np.max(delta):.3e}")


if __name__ == "__main__":
    main()
