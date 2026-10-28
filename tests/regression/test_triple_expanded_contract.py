"""Explicit triple configuration must be independent of keyword ordering."""
import pytest
import lcbinint as lc

@pytest.mark.parametrize('param_type',['vbm','lcbinint','center_of_mass'])
@pytest.mark.parametrize('options_first',[False,True])
def test_explicit_coordinates_override_options_independently_of_order(param_type,options_first):
    opts=lc.Options(finite_source_tol=0.,finite_source_reltol=1e-4)
    kwargs=dict(options=opts,param_type=param_type)if options_first else dict(param_type=param_type,options=opts)
    curve=lc.LightCurve(lens='triple',**kwargs)
    assert curve.options.param_type==param_type
    assert opts.param_type=='vbm'

@pytest.mark.parametrize('options_first',[False,True])
def test_explicit_accuracy_overrides_options_independently_of_order(options_first):
    opts=lc.Options(finite_source_tol=0.,finite_source_reltol=1e-3)
    kwargs=dict(options=opts,finite_source_reltol=1e-4)if options_first else dict(finite_source_reltol=1e-4,options=opts)
    curve=lc.LightCurve(lens='triple',**kwargs)
    assert curve.options.finite_source_reltol==1e-4
    assert opts.finite_source_reltol==1e-3


def test_uniform_exterior_fold_chart_resolves_material_transport():
    # Independent VBM NoPoly central differences at RelTol=1e-8; three
    # relative step sizes and MultiPoly agreement are saved in expanded audit.
    # The former wide chart passed the scalar budget but lost substantial
    # boundary motion, including the sign of dmu/ds.
    import numpy as np
    params = dict(t0=0.8008668614346952, u0=-1.6747956982741097,
        tE=4.330794712745609, alpha=-1.9183887569616975,
        s=0.32016878236693824, q=0.020446716237316108,
        q2=0.00014782157687368968, sep2=2.247456473008263,
        ang=-0.23499439560369773, rho=0.09442552750842173)
    epochs = [-12.191517276802132, -3.529927851310914,
        -0.2818318167517071, 0.8008668614346952, 1.8835655396210975,
        5.131661574180304, 13.793250999671523]
    reference = [0.03037392553989668, 0.07170469774697708,
        0.007593532235419543, -0.2023797843473396,
        0.07539406324875134, -0.0023770548764988726,
        0.00011064553801358116, 25.67313705910861,
        -0.08471372613232556, 0.2030237419146097]
    curve = lc.LightCurve(lens='triple',options=lc.Options(
        finite_source_tol=0.,finite_source_reltol=1e-3))
    forward = curve.magnification(epochs,params,with_gradient=True)
    reverse = curve.magnification(epochs[::-1]+[epochs[0]],params,with_gradient=True)
    for result,index in [(forward,4),(reverse,2)]:
        assert result.status[index] == 0
        assert result.gradient_valid[index]
        np.testing.assert_allclose(result.gradients[index],reference,
            rtol=2e-4,atol=1e-4)
