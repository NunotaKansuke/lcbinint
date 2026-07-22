import math

import numpy as np
import pytest


def test_vbm_binary_source_binary_lens_example_matches_reference_curve():
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBMicrolensing")

    params = dict(
        s=0.9,
        q=0.1,
        u0=0.0,
        alpha=1.0,
        rho=0.01,
        tE=30.0,
        t0=7500.0,
        piEN=0.3,
        piEE=-0.2,
        g1=0.011,
        g2=-0.005,
        g3=0.005,
        u0_2=0.2,
        t0_2=7500.0,
        q_source=1.0,
        w1=0.01,
        w2=0.02,
        w3=-0.015,
    )
    times = np.linspace(7470.0, 7530.0, 61)

    vbm = vbm_module.VBMicrolensing()
    vbm.SetObjectCoordinates("17:59:02.3 -29:04:15.2")
    vbm.t0_par_fixed = 1
    vbm.t0_par = params["t0"]
    reference_parameters = [
        math.log(params["s"]),
        math.log(params["q"]),
        params["u0"],
        params["alpha"],
        math.log(params["rho"]),
        math.log(params["tE"]),
        params["t0"],
        params["piEN"],
        params["piEE"],
        params["g1"],
        params["g2"],
        params["g3"],
        params["u0_2"],
        params["t0_2"],
        math.log(params["q_source"]),
        params["w1"],
        params["w2"],
        params["w3"],
    ]
    reference = np.asarray(
        vbm.BinSourceBinLensLightCurve(reference_parameters, times.tolist())[0]
    )

    actual = lcbinint.vbm_binary_source_binary_lens(
        times,
        params,
        sky=lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2"),
        options=lcbinint.Options(
            coordinates="vbm", tol=1.0e-3, reltol=1.0e-3
        ),
        t_ref=params["t0"],
    )

    # The bundled Earth ephemeris differs slightly from VBMicrolensing's
    # independently sampled table, most visibly at finite-source peaks.
    np.testing.assert_allclose(actual, reference, rtol=3.0e-3, atol=3.0e-3)
