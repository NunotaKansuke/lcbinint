"""LD radial resolution against tighter scalar differences.

References use the native value route at rtol=1e-9/1e-10 with central
h=1e-5,3e-6,1e-6. They are a different numerical derivative check, not
independent lens-engine certification. Raw step/tolerance studies are in
triple_component_completion_20261029/*-ld-disagreement-reference.json
and triple_ld_inner_precision_20261029/0945-ld-disagreement-reference.json.
"""
import numpy as np
import pytest
import lcbinint as lc

CASES = {940: {'epochs': [-1e-05, 0.0, 1e-05],
       'index': 1,
       'limb': 0.5,
       'params': {'alpha': -0.3120178496523116,
                  'ang': -1.181717300983814,
                  'q': 0.0006611429042828148,
                  'q2': 0.0046395981378236145,
                  'rho': 0.001,
                  's': 2.955117693649623,
                  'sep2': 1.3431020279379906,
                  't0': 0.4475777223551338,
                  'tE': 1.0,
                  'u0': 0.5326342554821512},
       'rtol': 0.001},
 1070: {'epochs': [-1e-05, 0.0, 1e-05],
        'index': 2,
        'limb': 1.0,
        'params': {'alpha': -0.05906734845791073,
                   'ang': 1.4377465377920817,
                   'q': 0.00013294697046209046,
                   'q2': 0.0005104111450999354,
                   'rho': 0.001,
                   's': 2.5012359395929527,
                   'sep2': 2.9341053763138,
                   't0': 2.101579634134624,
                   'tE': 1.0,
                   'u0': -0.12439386609457687},
        'rtol': 0.001}}
CASES[945] = dict(CASES[940], epochs=[-0.0003, 0., 0.0003],
                  params=dict(CASES[940]['params'], rho=0.03))

REFERENCES = [(940,'alpha',-2.52720645),(940,'ang',-.3258105),
              (1070,'alpha',-12.1934985),(1070,'ang',-.04751065),
              (945,'alpha',-.02570795),(945,'ang',-.000918638)]

@pytest.mark.parametrize('case,parameter,reference',REFERENCES)
def test_ld_angular_tangent_has_sufficient_radial_resolution(case,parameter,reference):
    c=CASES[case]
    model=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))
    result=model.magnification(c['epochs'],c['params'],with_gradient=True)
    assert np.all(result.status==0) and np.all(result.gradient_valid)
    value=result.gradients[c['index'],list(result.gradient_parameters).index(parameter)]
    assert abs(value-reference)/(1+abs(reference))<=1e-4


@pytest.mark.parametrize("transform", ["period", "epoch_shift"])
def test_ld_small_angular_tangents_preserve_geometry(transform):
    c = CASES[1070]
    times = np.asarray(c["epochs"])
    params = c["params"]
    model = lc.LightCurve(lens="triple", param_type="vbm", ld_c=c["limb"],
        options=lc.Options(finite_source_tol=0., finite_source_reltol=c["rtol"]))
    base = model.magnification(times, params, with_gradient=True)
    if transform == "period":
        other_times = times
        other_params = dict(params, alpha=params["alpha"] + 2*np.pi,
                            ang=params["ang"] + 2*np.pi)
    else:
        other_times = times + 7
        other_params = dict(params, t0=params["t0"] + 7)
    other = model.magnification(other_times, other_params, with_gradient=True)
    for result in (base, other):
        assert np.all(result.status == 0) and np.all(result.gradient_valid)
    assert np.max(abs(other.magnifications / base.magnifications - 1)) <= 1e-4
    assert np.max(abs(other.gradients-base.gradients)/(1+abs(base.gradients))) <= 1e-4
