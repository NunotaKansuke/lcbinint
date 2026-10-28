"""Lost faint images must be recovered without loosening lens residuals."""
import numpy as np
import pytest
import lcbinint as lc

CASES = [({'alpha': 0.8612762503892952,
   'ang': 0.2867185402828061,
   'q': 0.001500592899602178,
   'q2': 3.540801293749738e-07,
   'rho': 1.2464141068164914e-05,
   's': 1.1149328298269279,
   'sep2': 0.06812199547345756,
   't0': 1.8987643258841391,
   'tE': 8.305714375134011,
   'u0': -1.6910166462929639},
  [-23.018378799517897,
   -6.406950049249872,
   -0.17766426789936363,
   1.8987643258841391,
   3.975192919667642,
   10.20447870101815,
   26.815907451286172]),
 ({'alpha': 2.8944815515702835,
   'ang': -0.5474561800399194,
   'q': 0.0015842690902937912,
   'q2': 0.060797995726234205,
   'rho': 0.0011390322758264607,
   's': 19.042132509079135,
   'sep2': 9.474508131656254,
   't0': 0.7782405440384803,
   'tE': 0.11494875386920894,
   'u0': 0.2152445678010877},
  [0.4333942824308535,
   0.6632917901692714,
   0.7495033555711781,
   0.7782405440384803,
   0.8069777325057825,
   0.8931892979076892,
   1.1230868056461072]),
 ({'alpha': 1.2275528470999797,
   'ang': 1.6866585393122415,
   'q': 2.2278491909147505e-05,
   'q2': 8.000605261857085e-07,
   'rho': 0.012080414958408113,
   's': 0.09590379055435382,
   'sep2': 4.020676534095628,
   't0': -1.907715117050747,
   'tE': 0.6155222530934628,
   'u0': 0.5152676039147752},
  [-3.7542818763311354,
   -2.5232373701442095,
   -2.0615956803241127,
   -1.907715117050747,
   -1.7538345537773812,
   -1.2921928639572842,
   -0.06114835777035865]),
 ({'alpha': 1.8829340731786983,
   'ang': 1.2583530060205224,
   'q': 1.9369707704174523e-05,
   'q2': 0.17690748711220722,
   'rho': 0.0006324020103386295,
   's': 1.4064831367120951,
   'sep2': 2.7799018093967423,
   't0': -1.5472048852728217,
   'tE': 0.32690674863077407,
   'u0': -0.18954404576211026},
  [-2.527925131165144,
   -1.874111633903596,
   -1.6289315724305153,
   -1.5472048852728217,
   -1.4654781981151281,
   -1.2202981366420476,
   -0.5664846393804995]),
 ({'alpha': 2.5547471791436953,
   'ang': -1.665641427810518,
   'q': 0.18379666188119348,
   'q2': 0.0005425905448433531,
   'rho': 0.011785708982042295,
   's': 2.4533912802736615,
   'sep2': 17.432897470886907,
   't0': -2.835134654284131,
   'tE': 4.856851295095781,
   'u0': -1.1980898154932151},
  [-17.405688539571475,
   -7.691985949379912,
   -4.049347478058076,
   -2.835134654284131,
   -1.6209218305101856,
   2.0217166408116505,
   11.735419231003213])]

@pytest.mark.parametrize("params,times", CASES)
def test_faint_image_count_is_stable_under_full_turn(params, times):
    model = lc.LightCurve(lens="triple", param_type="vbm")
    results = []
    for phase in (0.0, 2*np.pi):
        info = model._native.info(np.asarray(times), **{**params, "rho": 0.0,
                                            "alpha": params["alpha"] + phase})
        assert set(info.image_counts) == {4}
        results.append(info.magnifications)
    np.testing.assert_allclose(*results, rtol=1e-10, atol=0.0)

@pytest.mark.parametrize("params,times", [CASES[1], CASES[3]])
def test_finite_source_route_preserves_periodic_gradient(params, times):
    model = lc.LightCurve(lens="triple", param_type="vbm", options=lc.Options(
        finite_source_tol=0.0, finite_source_reltol=1e-4))
    first = model.info(times, params, with_gradient=True)
    rotated = model.info(times, {**params, "alpha": params["alpha"] + 2*np.pi,
                                 "ang": params["ang"] + 2*np.pi}, with_gradient=True)
    assert not np.any(first.status)
    assert not np.any(rotated.status)
    assert np.all(first.gradient_valid) and np.all(rotated.gradient_valid)
    np.testing.assert_allclose(first.magnifications, rotated.magnifications,
                               rtol=1e-9, atol=0.0)
    delta = np.abs(first.gradients - rotated.gradients)/(1 + np.abs(first.gradients))
    assert np.max(delta) < 1e-7
