"""Run from the parent checkout on one pinned CPU, JAX x64."""
import json
from pathlib import Path
import sys
from statistics import median
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
build = Path("build-accuracy-20261127/algebraic").resolve()
ad_src = Path("../autodiff-vbm/src").resolve()
sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f)]
sys.path[:0] = [str(build), str(ad_src)]
import lcbinint as lc
from autodiff_vbm import AutodiffVBM, _autodiff_vbm_native
import VBMicrolensing as V
assert Path(lc._native.__file__).is_relative_to(build)
assert Path(_autodiff_vbm_native.__file__).is_relative_to(ad_src)
tol = 1e-4

def measure(fn):
    fn()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        out = fn()
        samples.append(1e3*(time.perf_counter()-start))
    return median(samples),out

def run(orbit, limb, p, times):
    triple = orbit == "triple"
    moving = orbit in ("circular","kepler")
    method = {"static":"BinaryLightCurve","circular":"BinaryLightCurveOrbital",
              "kepler":"BinaryLightCurveKepler","triple":"TripleLightCurve"}[orbit]
    model = lc.LightCurve(lens="triple" if triple else "binary", ld_c=limb, t_ref=0.,
        orbital_motion=orbit if moving else "static",
        options=lc.Options(coordinates="vbm",finite_source_tol=tol,finite_source_reltol=tol))
    v = V.VBMicrolensing()
    ad = AutodiffVBM()
    for obj in (v,ad):
        obj.Tol = obj.RelTol = tol
        if moving: obj.SetObjectCoordinates("17:57:38.03 -28:38:28.53")
    v.a1 = limb
    if triple: v.SetMethod(v.Method.Multipoly)
    names = ["s","q","u0","alpha","rho","tE","t0"]
    if moving: names += ["g1","g2","g3"]
    if orbit == "kepler": names += ["lom_szs","lom_ar"]
    if triple: names += ["sep2","q2","ang"]
    physical = np.array([p[n] for n in names])
    logs = [i for i,n in enumerate(names) if n in ("s","q","rho","tE","sep2","q2")]
    params = physical.copy()
    params[logs] = np.log(params[logs])
    if moving: params = np.insert(params,7,[0.,0.])
    args,epochs = params.tolist(),times.tolist()
    tm,ref = measure(lambda:getattr(v,method)(args,epochs)[0])
    te,r = measure(lambda:model(times,p))
    row = dict(orbit=orbit,ld=limb,q=p["q"],epochs=len(times),value_exp=te,value_vbm=tm,
        mu_rel=float(np.max(np.abs(r.magnifications/np.asarray(ref)-1))),
        ok=int(sum(r.status==0)),integral=int(sum(r.support_required)))
    if not triple:
        jt=jnp.array(times)
        def flux(theta):
            vp=theta.at[jnp.array(logs)].set(jnp.log(theta[jnp.array(logs)]))
            if moving: vp=jnp.concatenate((vp[:7],jnp.zeros(2),vp[7:]))
            mag=getattr(ad,method)(vp,jt,a1=limb if limb else None)[0]
            return mag,mag
        both=jax.jit(jax.jacfwd(flux,has_aux=True))
        theta=jnp.array(physical)
        ta,(jac,mag)=measure(lambda:jax.block_until_ready(both(theta)))
        tg,g=measure(lambda:model(times,p,with_gradient=True))
        actual=np.asarray(g.gradients)[:,[g.gradient_parameters.index(n) for n in names]]
        refjac=np.asarray(jac)
        finite=np.isfinite(refjac).all(axis=1)
        valid=finite & g.gradient_valid
        err=np.max(abs(actual[valid]-refjac[valid]),axis=1)/np.maximum(1.,np.max(abs(refjac[valid]),axis=1)) if valid.any() else []
        row.update(grad_exp=tg,grad_ad=ta,ad_finite=int(sum(finite)),exp_valid=int(sum(g.gradient_valid)),
            grad_mu_rel=float(np.max(abs(g.magnifications/np.asarray(mag)-1))),
            grad_max=float(max(err)) if len(err) else None)
    print(json.dumps(row),flush=True)

for orbit in ("static","circular","kepler","triple"):
    for ld in (0.,.5):
        if orbit=="triple":
            for i in (568,889):
                c=json.loads(Path(f"experiments/algebraic_boundary/evidence/triple_event_precision_20261029/inputs/{i:04d}.json").read_text())
                run(orbit,ld,c["params"],np.array(c["epochs"]))
            continue
        for q in (.01,10.):
            p=dict(s=1.2,q=q,u0=.03,alpha=-.08,rho=.003,tE=1.,t0=.1)
            if orbit!="static":p.update(g1=.011,g2=-.005,g3=.009)
            if orbit=="kepler":p.update(lom_szs=.7,lom_ar=1.3)
            run(orbit,ld,p,np.linspace(-.2,.6,13))
