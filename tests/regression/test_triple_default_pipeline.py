"""Normal triple API: shared scalar routing and analytic parameter tangents."""
import math
import numpy as np
import pytest
import lcbinint as lc


def parameters(x=.1,y=.15,rho=.15):
    return dict(t0=x-.2/.92,u0=-y,tE=1.,alpha=0.,s=1.,q=.2/.72,
                q2=.08/.72,sep2=math.hypot(.4,.6),ang=math.atan2(.6,-.4),rho=rho)


def curve(limb=0.,rtol=1e-4,param_type='vbm',**kwargs):
    return lc.LightCurve(lens='triple',param_type=param_type,ld_c=limb,
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=rtol,**kwargs))


@pytest.mark.parametrize('limb',[0.,.5])
def test_default_and_explicit_backend_agree(limb):
    normal=curve(limb)
    assert normal.options.finite_source_backend=='algebraic_boundary'
    explicit=curve(limb,finite_source_backend='algebraic_boundary')
    p=parameters()
    for gradient in [False,True]:
        a=normal.magnification([-.01,0,.01],p,with_gradient=gradient)
        b=explicit.magnification([-.01,0,.01],p,with_gradient=gradient)
        assert np.all(a.status==0)
        np.testing.assert_array_equal(a.magnifications,b.magnifications)
        if gradient:np.testing.assert_array_equal(a.gradients,b.gradients)
        assert a.diagnostics is None


@pytest.mark.parametrize('limb',[0.,.5])
@pytest.mark.parametrize('x,y,rho,method',[(3.,.8,.001,0),(.5,.3,.01,1),(.1,.15,.15,7)])
def test_routes_and_tangents(limb,x,y,rho,method):
    model=curve(limb);p=parameters(x,y,rho)
    value=model.info([0.],p)
    gradient=model.info([0.],p,with_gradient=True)
    assert value.status.tolist()==gradient.status.tolist()==[0]
    assert value.diagnostics['finite_source_method'].tolist()==[method]
    assert gradient.diagnostics['finite_source_method'].tolist()==[method]
    assert gradient.gradient_valid.tolist()==[True]
    assert gradient.gradients.shape==(1,10)
    # Approximation tangents differentiate their selected scalar formula;
    # integral tangents are checked against tighter scalar integrals below.
    reference=model if method!=7 else curve(limb,1e-7)
    for j,name in enumerate(gradient.gradient_parameters):
        estimates=[]
        for h in ([1e-5,3e-6] if method!=7 else [1e-4,3e-5]):
            pp=dict(p);pm=dict(p);pp[name]+=h;pm[name]-=h
            a=reference.magnification([0.],pp);b=reference.magnification([0.],pm)
            assert a.status.tolist()==b.status.tolist()==[0]
            estimates.append((a.magnifications[0]-b.magnifications[0])/(2*h))
        assert abs(estimates[0]-estimates[1])<1e-4+2e-4*abs(estimates[1])
        np.testing.assert_allclose(gradient.gradients[0,j],estimates[1],rtol=2e-4,atol=1e-4,err_msg=name)


def test_zero_radius_uses_point_tangent():
    model=curve();p=parameters(3.,.8,0.)
    out=model.info([0.],p,with_gradient=True)
    assert out.status.tolist()==[0]
    assert out.diagnostics['finite_source_method'].tolist()==[0]
    assert out.gradients[0,4]==0


@pytest.mark.parametrize('x,rho,reference',[
    (.0840625,.075,10.347258346635238),
    (.125,.3,5.846141393365811),
    (.121484375,.3,5.835729710839479),
])
def test_value_error_regressions(x,rho,reference):
    model=curve();p=parameters(x,.15,rho)
    for gradient in [False,True]:
        result=model.info([0.],p,with_gradient=gradient)
        assert result.status.tolist()==[0]
        assert abs(result.magnifications[0]/reference-1)<=1e-4


def test_unsupported_default_modes_keep_the_existing_backend():
    assert lc.LightCurve(lens='triple',jax=True).options.finite_source_backend=='default'
    assert lc.LightCurve(lens='triple',ld_d=.1).options.finite_source_backend=='default'
    assert lc.LightCurve(lens='triple',options=lc.Options(finite_source_backend='default')).options.finite_source_backend=='default'


@pytest.mark.parametrize('param_type',['vbm','lcbinint'])
@pytest.mark.parametrize('x,y,rho',[(3.,.8,.001),(1.2,.5,.03)])
def test_approximation_pullback_in_both_coordinate_conventions(param_type,x,y,rho):
    model=curve(.5,param_type=param_type);p=parameters(x,y,rho)
    result=model.info([0.],p,with_gradient=True)
    assert result.status.tolist()==[0]
    assert result.diagnostics['finite_source_method'][0] in (0,1)
    for j,name in enumerate(result.gradient_parameters):
        h=1e-5
        pp=dict(p);pm=dict(p);pp[name]+=h;pm[name]-=h
        vp=model.magnification([0.],pp);vm=model.magnification([0.],pm)
        fd=(vp.magnifications[0]-vm.magnifications[0])/(2*h)
        np.testing.assert_allclose(result.gradients[0,j],fd,rtol=2e-4,atol=1e-5,err_msg=param_type+' '+name)


def test_point_pullback_with_secondary_dominant_mass():
    model=curve(.5);p=parameters(3.,.8,.001);p.update(q=4.,q2=5/3,s=1.1)
    out=model.info([0.],p,with_gradient=True)
    assert out.status.tolist()==[0]
    assert out.diagnostics['finite_source_method'].tolist()==[0]
    for j,name in enumerate(out.gradient_parameters):
        h=1e-5;pp=dict(p);pm=dict(p);pp[name]+=h;pm[name]-=h
        fd=(model.magnification([0.],pp).magnifications[0]-model.magnification([0.],pm).magnifications[0])/(2*h)
        np.testing.assert_allclose(out.gradients[0,j],fd,rtol=2e-4,atol=1e-5,err_msg=name)
