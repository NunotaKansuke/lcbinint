"""Certified event refinement for wide, very faint triple components."""
import numpy as np
import pytest
import lcbinint as lc

# LD values: independent uniform-disk integration with MP/NP and nested
# Gauss-Legendre convergence. The flat wide case uses independent 80-digit
# original-equation solves and source-disk GL2/4/8 with 8/16/32 angles.
# These fixtures check the requested native value tolerance only.
CASES = [({'alpha': -0.9190931997774809,
   'ang': 3.141592653589793,
   'q': 1e-08,
   'q2': 1.0,
   'rho': 0.1,
   's': 100.0,
   'sep2': 100.1,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.5},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  1.0,
  [1.116551716997568,
   1.5932440062451427,
   1.6827175792810576,
   1.5980153701722584,
   1.1181002429113076]),
 ({'alpha': 2.6894446844287794,
   'ang': 0.0,
   'q': 1.0,
   'q2': 1e-07,
   'rho': 1e-05,
   's': 100.0,
   'sep2': 100.1,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.001},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  0.0,
  [1.0000002404510284,
   1.0000002399871888,
   1.0000002399680283,
   1.000000239987551,
   1.000000240452843])]


# Fresh-seed pole-local cases; independent tight MP/NP value references.
CASES += [({'alpha': 0.031623515241303135,
   'ang': 3.141592653612933,
   'q': 3.2603069899243716e-09,
   'q2': 0.7935392202810023,
   'rho': 0.06699926684438164,
   's': 98.1900391759434,
   'sep2': 102.92350649972043,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.3367958906193129},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  0.0,
  [1.1539063710482131,
   2.0948687834887014,
   2.3934245549315145,
   2.114398549028193,
   1.1568903256267125]),
 ({'alpha': 2.274697284235639,
   'ang': 3.141592653634099,
   'q': 5.411357678863148e-09,
   'q2': 1.302307905145199,
   'rho': 0.11597927188174925,
   's': 95.47547814279011,
   'sep2': 95.48724325291167,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.3603001631701901},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  0.0,
  [1.1118837650667035,
   1.8352826940439544,
   2.0334674881010937,
   1.822869532160449,
   1.1099381567923283]),
 ({'alpha': 1.70857521143345,
   'ang': 4.7520385172833964e-11,
   'q': 0.6064644373269596,
   'q2': 2.4180297267465928e-09,
   'rho': 0.00037622479207914634,
   's': 109.50122121427005,
   'sep2': 114.25141995448374,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.026328732977432368},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  1.0,
  [1.0, 1.0, 1.0, 1.0, 1.0])]

# Independent uniform-disk GL 128/256/512 LD integration, MP and NP.
CASES += [({'alpha': 0.6837289487109501,
   'ang': -1.0471975511965976,
   'q': 10000.0,
   'q2': 1e-07,
   'rho': 1.0,
   's': 10.0,
   'sep2': 17.0,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.5},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  1.0,
  [1.3848047531067091,
   2.2752186226386075,
   2.3155378065746155,
   2.274594454153849,
   1.383491619251293])]

@pytest.mark.parametrize("params,epochs,limb,reference", CASES)
@pytest.mark.parametrize("gradient", [False, True])
def test_wide_faint_event_precision(params, epochs, limb, reference, gradient):
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=limb,
        options=lc.Options(finite_source_tol=0., finite_source_reltol=1e-3))
    result = model.magnification(epochs, params, with_gradient=gradient)
    assert np.all(result.status == 0)
    np.testing.assert_allclose(result.magnifications, reference, rtol=1e-3, atol=0.)
    if gradient:
        assert np.all(result.gradient_valid)
        assert np.all(np.isfinite(result.gradients))


@pytest.mark.parametrize("case_index,epoch_index,parameter,reference,tolerance", [
    (0, 2, "t0", -.0153760564, 1e-4),
    (0, 2, "rho", .13254049, 1e-4),
    (5, 0, "t0", -.85691791415, 1e-6),
    (5, 0, "rho", .35785029400, 1e-6),
])
def test_recovered_ld_gradient_against_uniform_disks(
        case_index, epoch_index, parameter, reference, tolerance):
    # Independent uniform-disk VBM MP/NP finite differences, n=64/128/256,
    # h=2e-4/1e-4 (case 504), and n=128/256/512, h=2e-3/1e-3
    # (case 483). The diagnostic bound includes numerical scatter;
    # it is not a universal gradient-accuracy certificate.
    params, epochs, limb, _ = CASES[case_index]
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=limb,
        options=lc.Options(finite_source_tol=0., finite_source_reltol=1e-4 if case_index == 5 else 1e-3))
    result = model.magnification(epochs, params, with_gradient=True)
    column = list(result.gradient_parameters).index(parameter)
    assert result.gradient_valid[epoch_index]
    assert result.gradients[epoch_index, column] == pytest.approx(reference, abs=tolerance)


def test_cold_query_succeeds_without_warmup():
    # A fresh interpreter is essential: a second query previously succeeded
    # because the first refused support proof had refined the caustic tree.
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    case = {'epochs': [-1.0, -0.2, 0.0, 0.2, 1.0],
     'family': 'wide_tiny_secondary',
     'limb': 0.0,
     'name': 'wide-faint-holdout-009',
     'params': {'alpha': -2.054792796072074,
                'ang': 3.1415926534917897,
                'q': 2.9198833313501203e-09,
                'q2': 1.7304494453174033,
                'rho': 0.07525908655273247,
                's': 96.03972329879036,
                'sep2': 96.05834419780993,
                't0': 0.0,
                'tE': 1.0,
                'u0': 0.3968957118855627},
     'rtol': 0.0001}
    code = r"""
import json,sys
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)]
sys.path.insert(0,sys.argv[1])
import numpy as np
import lcbinint as lc
c=json.load(sys.stdin)
m=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],
    options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))
a=m.magnification(c['epochs'],c['params'],with_gradient=True)
b=m.magnification(c['epochs'],c['params'],with_gradient=True)
assert np.all(a.status==0) and np.all(a.gradient_valid)
assert np.all(b.status==0) and np.all(b.gradient_valid)
# Tight independent MP/NP uniform-disk references.
ref=[1.08622390698,1.64583346924,1.78810474136,1.63826516826,1.08492032301]
np.testing.assert_allclose(a.magnifications,ref,rtol=c['rtol'],atol=0.)
np.testing.assert_allclose(b.magnifications,ref,rtol=c['rtol'],atol=0.)
# A cold full integral and a later certified hex packet may use different
# value-controlled meshes. Check the requested value contract and the same
# scaled-gradient invariance criterion as the broader transform campaign.
np.testing.assert_allclose(a.magnifications,b.magnifications,rtol=c['rtol'],atol=0.)
assert np.max(abs(a.gradients-b.gradients)/(1+abs(b.gradients))) < 1e-4
"""
    result = subprocess.run([sys.executable, "-c", code,
        str(Path(lc.__file__).resolve().parent.parent)], input=json.dumps(case),
        text=True, capture_output=True, timeout=60,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr


def test_frame_choice_retains_independent_alpha_derivative():
    # Uniform-disk MultiPoly RelTol=1e-10, central h=1e-3..5e-5:
    # 0.3957030..0.3957051. NoPoly crashes in this diagnostic.
    # An unconditional pole-origin preference gave 0.39594235 despite
    # satisfying the value tolerance; keep this independent tangent guard.
    params = {'alpha': -1.8594437607447576,
     'ang': 2.6734634421752164,
     'q': 8.141453873680264,
     'q2': 1.436263027565987e-07,
     'rho': 0.19199011802322175,
     's': 6.909760726105167,
     'sep2': 4.6541218209399,
     't0': 1.1569446546896547,
     'tE': 0.9865772278741568,
     'u0': -1.3048876043881141}
    epochs = [-1.8027870289328156, 0.17036742681549788, 0.9103003477211155, 1.1569446546896547, 1.4035889616581938, 2.1435218825638116, 4.116676338312125]
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=0.,
        options=lc.Options(finite_source_tol=0., finite_source_reltol=1e-4))
    result = model.magnification(epochs, params, with_gradient=True)
    assert np.all(result.status == 0) and np.all(result.gradient_valid)
    column = list(result.gradient_parameters).index("alpha")
    assert result.gradients[4, column] == pytest.approx(.3957046, abs=2e-6)


@pytest.mark.parametrize("gradient", [False, True])
def test_periodic_large_ld_source_retains_support(gradient):
    params, epochs, limb, reference = CASES[5]
    shifted = dict(params, alpha=params["alpha"]+2*np.pi, ang=params["ang"]+2*np.pi)
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=limb,
        options=lc.Options(finite_source_tol=0., finite_source_reltol=1e-4))
    result = model.magnification(epochs, shifted, with_gradient=gradient)
    assert np.all(result.status == 0)
    np.testing.assert_allclose(result.magnifications, reference, rtol=1e-4, atol=0.)
    if gradient:
        assert np.all(result.gradient_valid)
        for name, value in [("t0", -.85691791415), ("rho", .35785029400)]:
            column = list(result.gradient_parameters).index(name)
            assert result.gradients[0, column] == pytest.approx(value, abs=1e-6)
