"""Pole-local image transport and optional-partition fallback regressions."""
import numpy as np
import pytest
import lcbinint as lc

POINT_CASES = [({'alpha': -3.02576713601945,
   'ang': 0.7277460252170198,
   'q': 15.323735743955856,
   'q2': 5.397231219594437e-06,
   'rho': 3.90315194701206e-05,
   's': 19.75305012937681,
   'sep2': 19.051792915401325,
   't0': 1.6379183964539363,
   'tE': 2.709562670099395,
   'u0': 1.7531007289722664},
  [-6.490769613844249,
   -1.0716442736454588,
   0.9605277289290876,
   1.6379183964539363,
   2.315309063978785,
   4.347481066553332,
   9.766606406752121],
  1.0,
  0.0001),
 ({'alpha': 1.5406510228124688,
   'ang': 1.9826520680699726,
   'q': 1.436858977965615e-07,
   'q2': 91.0362826939728,
   'rho': 0.0004226806992084662,
   's': 10.690757151389821,
   'sep2': 0.30669514889554955,
   't0': -1.4009862584032844,
   'tE': 1.2926518142044545,
   'u0': 1.5330410230329328},
  [-5.278941701016648,
   -2.693638072607739,
   -1.724149211954398,
   -1.4009862584032844,
   -1.0778233048521708,
   -0.10833444419882987,
   2.4769691842100796],
  0.5,
  0.001)]

@pytest.mark.parametrize("params,times,limb,rtol", POINT_CASES)
def test_local_faint_images_enable_certified_finite_source_route(params,times,limb,rtol):
    model=lc.LightCurve(lens="triple",param_type="vbm",ld_c=limb,
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=rtol))
    values=[]
    for phase in [0.,2*np.pi]:
        par=dict(params,alpha=params['alpha']+phase,ang=params['ang']+phase)
        result=model.info(times,par,with_gradient=True)
        assert not np.any(result.status)
        assert np.all(result.gradient_valid)
        assert np.all(np.isfinite(result.gradients))
        assert np.all(np.asarray(result.diagnostics['center_image_count'])==4)
        assert np.all(np.asarray(result.diagnostics['finite_source_method'])==1)
        values.append(result.magnifications)
    np.testing.assert_allclose(values[0],values[1],rtol=1e-11,atol=0.)

PARTITION_CASES = [({'alpha': 1.9653439739564496,
   'ang': 1e-10,
   'q': 10000.0,
   'q2': 1e-07,
   'rho': 0.1,
   's': 100.0,
   'sep2': 100.1,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.1},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  0.5,
  0.0001),
 ({'alpha': 1.5735276632170336,
   'ang': 0.0,
   'q': 1e-08,
   'q2': 1e-07,
   'rho': 0.1,
   's': 10.0,
   'sep2': 5.0,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.001},
  [-1.0, -0.2, 0.0, 0.2, 1.0],
  0.0,
  0.001)]

@pytest.mark.parametrize("params,times,limb,rtol", PARTITION_CASES)
def test_narrow_component_masks_preserve_complete_integral(params,times,limb,rtol):
    model=lc.LightCurve(lens="triple",param_type="vbm",ld_c=limb,
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=rtol))
    scalar=model.info(times,params,with_gradient=False)
    tangent=model.info(times,params,with_gradient=True)
    assert not np.any(scalar.status)
    assert not np.any(tangent.status)
    assert np.all(tangent.gradient_valid)
    assert np.all(np.isfinite(tangent.gradients))
    np.testing.assert_allclose(tangent.magnifications,scalar.magnifications,rtol=rtol,atol=0.)


SIX_IMAGE_CASES = [({'alpha': 1.806912430583373,
   'ang': -0.30732930862308194,
   'q': 28.452407713501238,
   'q2': 9.810429332179342e-07,
   'rho': 0.0,
   's': 1.2725950334694136,
   'sep2': 0.15575480539239409,
   't0': 0.397435854365388,
   'tE': 1.1107195526187446,
   'u0': -0.5105713979766136},
  0.397435854365388,
  6.444764519040969),
 ({'alpha': -2.701112752507171,
   'ang': 1.0471975511965976,
   'q': 1.0,
   'q2': 1e-07,
   'rho': 0.0,
   's': 1.0,
   'sep2': 1.7,
   't0': 0.0,
   'tE': 1.0,
   'u0': 0.1},
  -0.2,
  8.096020451794297)]

@pytest.mark.parametrize("params,epoch,reference", SIX_IMAGE_CASES)
def test_even_root_count_does_not_hide_negative_images(params,epoch,reference):
    model=lc.LightCurve(lens="triple",param_type="vbm",
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=1e-4))
    for gradient in (False,True):
        result=model.info([epoch,epoch],params,with_gradient=gradient)
        assert not np.any(result.status)
        np.testing.assert_allclose(result.magnifications,reference,rtol=2e-10,atol=0.)
        if gradient:
            # The scalar info adapter does not export center-image counts;
            # its value is checked against the independent root reference.
            np.testing.assert_array_equal(result.diagnostics["center_image_count"],6)
            assert np.all(result.gradient_valid)
            assert np.all(np.isfinite(result.gradients))
