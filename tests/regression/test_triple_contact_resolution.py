"""Retain contact packets when radial coordinates hit binary64 resolution."""
import numpy as np
import pytest
import lcbinint as lc

# References use VBM MultiPoly/NoPoly at 1e-7; case940 center uses an
# independent LD integral of uniform disks because VBM LD disagrees there.
CASES = [({'alpha': -0.3120178496523116,
   'ang': -1.181717300983814,
   'q': 0.0006611429042828148,
   'q2': 0.0046395981378236145,
   'rho': 0.001,
   's': 2.955117693649623,
   'sep2': 1.3431020279379906,
   't0': 0.4475777223551338,
   'tE': 1.0,
   'u0': 0.5326342554821512},
  [-1e-05, 0.0, 1e-05],
  0.5,
  0.001,
  [21.683906330680934, 21.695144688028726, 21.70662987050143]),
 ({'alpha': -0.3120178496523116,
   'ang': -1.181717300983814,
   'q': 0.0006611429042828148,
   'q2': 0.0046395981378236145,
   'rho': 0.001,
   's': 2.955117693649623,
   'sep2': 1.3431020279379906,
   't0': 2.4946396897009873,
   'tE': 1.0,
   'u0': -0.8049140825679901},
  [-1e-05, 0.0, 1e-05],
  1.0,
  0.0001,
  [5.530931341216932, 5.4932916459682435, 5.456872833304259]),
 ({'alpha': -1.7292072992630243,
   'ang': 1.5461976432747724,
   'q': 0.7811429674468207,
   'q2': 0.00024085875183855085,
   'rho': 0.001,
   's': 2.1495379800947068,
   'sep2': 1.4841532946318223,
   't0': -0.879121302137277,
   'tE': 1.0,
   'u0': 0.926457176704351},
  [-1e-05, 0.0, 1e-05],
  1.0,
  0.001,
  [2.8734999574560063, 2.8602026346206126, 2.8492218339567956]),
 ({'alpha': 0.003671324967900347,
   'ang': 0.7776922930606145,
   'q': 0.0010886471615905282,
   'q2': 0.0024460734995836874,
   'rho': 0.001,
   's': 1.7918573818493355,
   'sep2': 1.0400206072513902,
   't0': 1.255968029012023,
   'tE': 1.0,
   'u0': 0.004574505644092094},
  [-1e-05, 0.0, 1e-05],
  0.5,
  0.0001,
  [5.118894885474472, 5.111701720190406, 5.1045406518487715])]

@pytest.mark.parametrize("params,times,limb,rtol,reference", CASES)
def test_contact_resolution_retains_converged_gradient(params,times,limb,rtol,reference):
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=limb,
                         options=lc.Options(finite_source_tol=0., finite_source_reltol=rtol))
    result = model.info(times, params, with_gradient=True)
    assert not np.any(result.status)
    assert np.all(result.gradient_valid)
    assert np.all(np.isfinite(result.gradients))
    np.testing.assert_allclose(result.magnifications, reference, rtol=rtol, atol=0.)


def test_contact_time_gradient_matches_independent_ld_integral():
    params,times,limb,rtol,_ = CASES[0]
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=limb,
                         options=lc.Options(finite_source_tol=0., finite_source_reltol=rtol))
    result = model.info([times[1]], params, with_gradient=True)
    index = list(result.gradient_parameters).index("t0")
    # 256-node LD integral, central differences h=1e-5 and 5e-6,
    # followed by Richardson extrapolation. Both VBM uniform engines agree.
    assert result.gradients[0,index] == pytest.approx(-1135.785123, rel=3e-5)


def test_contact_mass_gradient_matches_independent_ld_integral():
    params,times,limb,rtol,_ = CASES[2]
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=limb,
                         options=lc.Options(finite_source_tol=0., finite_source_reltol=rtol))
    result = model.info([times[2]], params, with_gradient=True)
    index = list(result.gradient_parameters).index("q2")
    # 1024-node independent LD integration, two finite-difference steps;
    # VBM's direct LD derivative is spuriously stable near -3249 here.
    assert result.gradients[0,index] == pytest.approx(-3194.13, rel=3e-4)
