"""Independent uniform-disk LD differences; tolerances are diagnostic."""
import pytest
import lcbinint as lc

# The t0 references use uniform disks with MP RelTol=1e-9, nested GL
# 128/256/512 and central h=2e-6/1e-6 with Richardson extrapolation.
# The alpha reference uses MP RelTol=1e-10, GL 128/256/512 and
# one-sided h=1e-4 through 1.25e-5. Its last mesh gives -0.014974
# through -0.014982. NoPoly crashes on these reference attempts;
# these checks are not a two-engine or universal derivative certificate.
# The rho reference uses GL 128/256 and one-sided h down to 4.6875e-6.
# Last forward/backward values span 91.376..91.414; the 0.3 bound
# conservatively includes the preceding step and mesh variation.
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
  0,
  0.5,
  0.001,
  't0',
  -1278.179187861698,
  1e-05,
  0.0),
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
  0,
  0.5,
  0.0001,
  't0',
  720.5067093186321,
  1e-05,
  0.0),
 ({'alpha': 3.0210722997708777,
   'ang': -2.9649440787105177,
   'q': 0.656718789272734,
   'q2': 0.0016528384914630317,
   'rho': 0.03,
   's': 2.4619692800350417,
   'sep2': 0.5859189243968717,
   't0': -1.3228554819628284,
   'tE': 1.0,
   'u0': 0.16019985077825086},
  [-0.0003, 0.0, 0.0003],
  1,
  1.0,
  0.001,
  'alpha',
  -0.01498,
  0.0,
  0.0001),
 ({'alpha': 0.8676124891755457,
   'ang': -1.4588327189644557,
   'q': 0.0031095254424007655,
   'q2': 0.01937108018185108,
   'rho': 0.03,
   's': 1.069260594334073,
   'sep2': 1.0337726985181008,
   't0': -0.2924773685373103,
   'tE': 1.0,
   'u0': 0.2638718925803957},
  [-0.0003, 0.0, 0.0003],
  1,
  1.0,
  0.001,
  'rho',
  91.4,
  0.0,
  0.3)]

@pytest.mark.parametrize("params,times,index,limb,rtol,parameter,reference,relative,absolute", CASES)
def test_ld_contact_derivative_matches_uniform_disk_difference(
        params,times,index,limb,rtol,parameter,reference,relative,absolute):
    model=lc.LightCurve(lens="triple",param_type="vbm",ld_c=limb,
        options=lc.Options(finite_source_tol=0.,finite_source_reltol=rtol))
    result=model.magnification(times,params,with_gradient=True)
    assert all(x==0 for x in result.status)
    assert all(result.gradient_valid)
    column=list(result.gradient_parameters).index(parameter)
    assert result.gradients[index,column]==pytest.approx(reference,rel=relative,abs=absolute)
