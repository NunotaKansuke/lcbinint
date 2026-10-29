"""All-image pole partitions, independent 70-digit Green values and tangents."""
import numpy as np
import pytest
import lcbinint as lc

CASES = [{'case': 439,
  'epochs': [-1.0, -0.2, 0.0, 0.2, 1.0],
  'family': 'near_collinear_or_clustered',
  'limb': 0.0,
  'name': 'near_collinear_or_clustered-0439',
  'params': {'alpha': 2.912898561527994,
             'ang': 3.141592643589793,
             'q': 10000.0,
             'q2': 0.01,
             'rho': 0.003,
             's': 100.0,
             'sep2': 100.00009999999999,
             't0': 0.0,
             'tE': 1.0,
             'u0': 0.001},
  'references': [1.3347254921422793,
                 4.8459560423663355,
                 103.0663678694944,
                 5.3267712705734995,
                 1.3486616843789607],
  'rtol': 0.0001},
 {'case': 499,
  'epochs': [-1.0, -0.2, 0.0, 0.2, 1.0],
  'family': 'near_collinear_or_clustered',
  'limb': 0.0,
  'name': 'near_collinear_or_clustered-0499',
  'params': {'alpha': -1.239697241437293,
             'ang': 3.141592643589793,
             'q': 10000.0,
             'q2': 1e-07,
             'rho': 0.1,
             's': 100.0,
             'sep2': 100.00009999999999,
             't0': 0.0,
             'tE': 1.0,
             'u0': 0.001},
  'references': [1.3453506785010318,
                 5.332826142016287,
                 19.964142263792183,
                 5.155227199066124,
                 1.3406725656344682],
  'rtol': 0.0001}]

REFERENCES = [(439, 't0', -10665.519813894838),
 (439, 'u0', 1387.2896225654852),
 (439, 'tE', 0.0),
 (439, 'alpha', -10.665519813898076),
 (439, 'rho', 851.8826650181624),
 (439, 's', -1.0702335107420184),
 (439, 'q', 0.010699635762488592),
 (439, 'q2', 0.00019783160200144458),
 (439, 'sep2', -2.676195226004392e-07),
 (439, 'ang', -2.668983876255528e-06),
 (499, 't0', 3.252600810373537),
 (499, 'u0', -10.462764906441869),
 (499, 'tE', 0.0),
 (499, 'alpha', 0.003252600810373537),
 (499, 'rho', -197.93796396006704),
 (499, 's', -0.0010951850392417403),
 (499, 'q', 1.104812408416322e-05),
 (499, 'q2', -0.000990113837848104),
 (499, 'sep2', -2.7377975440616926e-15),
 (499, 'ang', 8.132977486621895e-15)]

def curve(c):
    return lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))

@pytest.mark.parametrize('c',CASES,ids=lambda c:str(c['case']))
@pytest.mark.parametrize('gradient',[False,True])
def test_full_curve_against_independent_green_boundary(c,gradient):
    result=curve(c).info(c['epochs'],c['params'],with_gradient=gradient)
    assert np.all(result.status==0)
    np.testing.assert_allclose(result.magnifications,c['references'],rtol=c['rtol'],atol=0.)
    assert result.diagnostics['stop'][2]=='complete_pole_partition'
    assert result.diagnostics['support_certified'][2]
    if gradient:
        assert np.all(result.gradient_valid) and np.all(np.isfinite(result.gradients))

@pytest.mark.parametrize('case,name,reference',REFERENCES)
def test_partition_tangents_against_independent_green_differences(case,name,reference):
    c=next(c for c in CASES if c['case']==case)
    result=curve(c).magnification([0.],c['params'],with_gradient=True)
    assert result.status.tolist()==[0] and result.gradient_valid.tolist()==[True]
    native=result.gradients[0,list(result.gradient_parameters).index(name)]
    assert abs(native-reference)/(1+abs(reference))<1e-4

@pytest.mark.parametrize('c',CASES,ids=lambda c:str(c['case']))
def test_partition_order_and_period(c):
    m=curve(c);p=c['params'];times=np.asarray(c['epochs'])
    a=m.magnification(times,p,with_gradient=True)
    for ts,par,idx in [(times[::-1],p,np.arange(5)[::-1]),(times,dict(p,alpha=p['alpha']+2*np.pi,ang=p['ang']+2*np.pi),np.arange(5))]:
        b=m.magnification(ts,par,with_gradient=True)
        assert np.all(b.status==0) and np.all(b.gradient_valid)
        assert np.max(abs(b.magnifications/a.magnifications[idx]-1))<c['rtol']
        assert np.max(abs(b.gradients-a.gradients[idx])/(1+abs(a.gradients[idx])))<1e-4
