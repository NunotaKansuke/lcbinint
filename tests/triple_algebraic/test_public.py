import argparse
import pathlib
import sys
import math
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--build',type=pathlib.Path,required=True)
args=parser.parse_args()
build=args.build.resolve()
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)]
sys.path.insert(0,str(build))
import lcbinint as native
assert pathlib.Path(native.__file__).resolve().is_relative_to(build)
assert pathlib.Path(native._native.__file__).resolve().is_relative_to(build)
lenses=[0j,1+0j,-.4+.6j]
result=native.triple_algebraic(lenses,[.72,.2,.08],.1+.15j,.15,limb=.5,with_gradient=True)
assert result['value_converged'] and result['gradient_valid'] and result['support_complete'],result
assert abs(result['mu']-8.630749933917409)<1e-4,result
assert result['quad_samples']<result['node_evaluations'],result
failed=native.triple_algebraic(lenses,[.72,.2,.08],.1+.15j,.15,
    with_gradient=True,event_box_budget=1)
assert not failed['value_converged'] and not failed['gradient_valid']
assert not failed['support_complete'] and math.isnan(failed['mu'])
assert all(math.isnan(x) for x in failed['gradient']),failed
assert failed['stop']=='event_box_budget',failed
value=native.triple_algebraic(lenses,[.72,.2,.08],.1+.15j,.15,limb=.5)
assert value['value_converged'] and value['parameter_derivative_evaluations']==0,value
assert abs(value['mu']-result['mu'])<1e-4
# Physical gradients obey translation and rotation invariance.
g=np.asarray(result['gradient'])
assert abs(g[0]+g[5]+g[7]+g[9])<1e-8
assert abs(g[1]+g[6]+g[8]+g[10])<1e-8
rotation=-.15*g[0]+.1*g[1]
for j,z in enumerate(lenses):rotation+=-z.imag*g[5+2*j]+z.real*g[6+2*j]
assert abs(rotation)<1e-8
params=dict(t0=-(.2/.92-.1),u0=-.15,tE=1.,alpha=0.,s=1.,q=.2/.72,q2=.08/.72,
            sep2=math.hypot(.4,.6),ang=math.atan2(.6,-.4),rho=.15)
curve=native.LightCurve(lens='triple',param_type='vbm',options=native.Options(
    finite_source_backend='algebraic_boundary',finite_source_tol=1e-6,finite_source_reltol=1e-5),ld_c=.5)
info=curve.info([0.],params,with_gradient=True)
assert info.status.tolist()==[0] and info.gradient_valid.tolist()==[True],info
assert info.gradient_parameters==('t0','u0','tE','alpha','rho','q','s','q2','sep2','ang')
assert info.gradients.shape==(1,10) and np.isfinite(info.gradients).all()
plain=curve.info([0.],params)
assert plain.status.tolist()==[0] and abs(plain.magnifications[0]-info.magnifications[0])<1e-4
assert 'jax' not in sys.modules,'Native analytic gradients imported JAX'
print('PASS: verified native public triple API, invariances, LightCurve values and 10 analytic derivatives')
# Changing epochs must use fresh source proofs while reusing lens preparation.
# Match physical source positions in the VBM convention (primary at +offset).
for rtol in (1e-3,1e-4):
    for limb in (0.,.5):
        batched=native.LightCurve(lens='triple',param_type='vbm',options=native.Options(
            finite_source_backend='algebraic_boundary',finite_source_tol=0.,
            finite_source_reltol=rtol),ld_c=limb)
        for x,y,rho in ((.5,.3,.01),(.1,.15,.15),(.1,.15,.5),(2.,1.,.01)):
            request=dict(params,t0=0.,u0=-y,rho=rho)
            xs=np.array([x-.002*rho,x+.002*rho,x-.002*rho])
            epochs=.2/.92-xs
            batch=batched.info(epochs,request)
            assert batch.status.tolist()==[0,0,0],batch
            direct=[native.triple_algebraic(lenses,[.72,.2,.08],complex(xx,y),rho,
                limb=limb,atol=0.,rtol=rtol) for xx in xs]
            assert all(r['value_converged'] and r['parameter_derivative_evaluations']==0 for r in direct)
            assert np.allclose(batch.magnifications,[r['mu'] for r in direct],rtol=rtol,atol=0)
print('PASS: moving and repeated epochs match direct values for uniform/LD and both loose tolerances')
# A noncentral large/LD source previously exhausted the derivative budget
# while refining the angular rule for a radial error. Keep both tolerances.
for rtol in (1e-3,1e-4):
    regression=native.triple_algebraic(lenses,[.72,.2,.08],
        complex(.1258549239677236,.06379371886791989),.3,
        limb=.5,with_gradient=True,atol=0.,rtol=rtol)
    assert regression['value_converged'] and regression['gradient_valid'],regression
    assert regression['gradient_policy']=='value_mesh',regression
    assert regression['support_complete'] and regression['refinements']<=160,regression
print('PASS: near-contact large/LD derivative refinement regression at both tolerances')

# Explicit binary-compatible value-mesh policy never claims derivative accuracy.
for source,rho in ((.5+.4j,.03),(.1+.15j,.15)):
    for limb in (0.,.5):
        for rtol in (1e-3,1e-4):
            kw=dict(limb=limb,atol=0.,rtol=rtol)
            vm=native.triple_algebraic(lenses,[.72,.2,.08],source,rho,with_gradient=True,**kw)
            assert vm['value_converged'] and vm['gradient_valid'] and vm['support_complete'],vm
            assert not {'gradient_reliable','gradient_precision_checked','canonical_gradient_error_estimate'}.intersection(vm),vm
            assert vm['gradient_policy']=='value_mesh' and np.isfinite(vm['gradient']).all(),vm
            plain=native.triple_algebraic(lenses,[.72,.2,.08],source,rho,**kw)
            assert plain['parameter_derivative_evaluations']==0 and plain['gradient_policy']=='none',plain
            assert abs(vm['mu']-plain['mu'])<=vm['error_estimate']+plain['error_estimate']+1e-9,(vm,plain)
assert result['gradient_valid'] and result['gradient_policy']=='value_mesh'
# The old policy selector is removed, rather than silently aliasing strict.
for policy in ('strict','value_mesh'):
    try:
        native.triple_algebraic(lenses,[.72,.2,.08],.5+.4j,.03,gradient_policy=policy)
    except TypeError:
        pass
    else:
        raise AssertionError('removed gradient policy keyword accepted')
print('PASS: value-controlled derivatives, removed policy selector and value-only isolation')
