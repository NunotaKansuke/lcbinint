"""Independent single-lens limits, wrap/symmetry and adaptive API recovery."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('POLAR_CHECK_OUTPUT_DIR', ROOT / 'tests/diagnostics/results/recal2026/polar-local-adaptive-20261014'))

def load_native():
    path = next((ROOT / 'build-adaptive-production').glob('_lcbinint*.so'))
    spec = importlib.util.spec_from_file_location('_lcbinint', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native

def main():
    m = load_native()
    rows = []
    def record(name, ok, **details):
        rows.append(dict(name=name, passed=bool(ok), **details))
    def call(x=.02, y=.0001, s=1.3, q=.01, rho=.003, c=.5, tol=1e-4, **kwargs):
        return m.binary_adaptive_polar_flux(x, y, s=s, q=q, rho=rho,
            limb_darkening=m.LimbDarkening(c, 0), relative_tolerance=tol, **kwargs)
    for rho in [.01, .1]:
        for c in [0., .5, 1.]:
            # At source centre u=0, the point-lens source integral reduces to
            # this independent one-dimensional source-plane quadrature.
            ref = quad(lambda t: 2*rho*math.cos(t)*(1-c+c*math.cos(t)) *
                ((rho*math.sin(t))**2+2)/math.sqrt((rho*math.sin(t))**2+4),
                0, math.pi/2, epsabs=1e-12, epsrel=1e-12)[0] / (rho*rho*(1-c/3))
            for tol in [1e-3, 1e-4]:
                got = call(x=1e-8, y=1e-8, s=1e-8, q=1., rho=rho, c=c, tol=tol)
                error = abs(got['magnification']/ref-1)
                # The existing binary seed certificate rejects s -> 0. The
                # analytic check still verifies the numerical flux; that
                # support rejection must remain visible and fail closed.
                support_ok = got['estimated_target_met'] or (got['stop_reason']=='unresolved_support' and math.isinf(got['error_indicator']))
                record('single_lens', error <= tol and support_ok,
                    rho=rho, c=c, tolerance=tol, reference=ref, relative_error=error, result=got)
    plus, minus = call(y=1e-8), call(y=-1e-8)
    error = abs(plus['magnification']/minus['magnification']-1)
    record('wrap_reflection', error < 1e-9 and plus['estimated_target_met'] and minus['estimated_target_met'],
        relative_difference=error, plus=plus, minus=minus)
    baseline = call()
    for kwargs in [dict(initial_bins=1), dict(initial_angular_ratio=0),
                   dict(max_angular_depth=21), dict(max_radial_depth=17), dict(max_lattice_maps=0)]:
        try:
            call(**kwargs)
            record('invalid_configuration', False, options=kwargs)
        except ValueError:
            record('invalid_configuration', True, options=kwargs)
    try:
        call(max_lattice_maps=1)
        record('discovery_budget', False)
    except RuntimeError:
        record('discovery_budget', True)
    recovered = call()
    record('context_recovery', baseline == recovered, baseline=baseline, recovered=recovered)
    for extreme_rho in [1e-200,1e200]:
        try:
            call(rho=extreme_rho)
            record('unrepresentable_source_flux',False,rho=extreme_rho)
        except ValueError:
            record('unrepresentable_source_flux',True,rho=extreme_rho)
    limited = call(tol=1e-8, max_angular_depth=0)
    record('angular_limit', not limited['estimated_target_met'] and limited['stop_reason']=='resolution_limit', result=limited)
    limited = call(tol=1e-12, max_radial_depth=0)
    record('radial_limit', not limited['estimated_target_met'] and
        limited['largest_radial_depth']==0 and
        limited['radial_error_indicator']>limited['requested_error_budget'], result=limited)
    record('acceptance_contract', baseline['error_indicator'] <= baseline['requested_error_budget'] and
        baseline['requested_error_budget'] == 1e-4*abs(baseline['magnification']) and
        baseline['error_indicator'] == baseline['angular_error_indicator'] + baseline['radial_error_indicator'] and
        not baseline['certified'] and not baseline['complete_error_model'], result=baseline)
    source=json.loads((ROOT/'tests/diagnostics/results/recal2026/polar-local-adaptive-20261014/high-selected.json').read_text())[0]
    reference=float(source['reference']);absolute_goal=reference*1e-4
    absolute=call(x=float(source['time']),y=float(source['y']),s=float(source['s']),
        q=1/float(source['q']),rho=float(source['rho']),c=float(source['c']),
        tol=0.,absolute_tolerance=absolute_goal)
    record('absolute_target', absolute['estimated_target_met'] and
        absolute['error_indicator']<=absolute_goal and abs(absolute['magnification']-reference)<=absolute_goal,
        absolute_tolerance=absolute_goal,reference=reference,result=absolute)
    for c,d in [(1.1,0),(-.1,0),(.5,.1),(float('nan'),0)]:
        try:
            m.binary_adaptive_polar_flux(.02,.0001,s=1.3,q=.01,rho=.003,
                limb_darkening=m.LimbDarkening(c,d),relative_tolerance=1e-4)
            record('invalid_limb_darkening', False, c=c,d=d)
        except ValueError:
            record('invalid_limb_darkening', True, c=c,d=d)
        try:
            # Validation must precede an upstream point-lens shortcut too.
            m.binary_adaptive_polar_flux(.02,.0001,s=1.,q=0.,rho=.003,
                limb_darkening=m.LimbDarkening(c,d),relative_tolerance=1e-4)
            record('invalid_limb_darkening_before_route',False,c=c,d=d)
        except ValueError as exc:
            record('invalid_limb_darkening_before_route','linear LD' in str(exc),c=c,d=d)
    saved_path = ROOT/'tests/diagnostics/results/recal2026/polar-maturity-20261014/validation.json'
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        for c in (.75,1.):
            thin = next(r for r in saved['inputs'] if r['geom_id']==f'new-25-{c}')
            reference = next(r['fine'] for r in saved['references'] if r['id']==thin['geom_id'])
            got = call(x=float(thin['time']),y=float(thin['y']),s=float(thin['s']),
                q=1/float(thin['q']),rho=float(thin['rho']),c=c)
            record('thin_image_between_nodes', got['estimated_target_met'] and
                abs(got['magnification']/reference-1)<1e-4, input=thin,result=got,reference=reference)
    skipped = call(x=4.755372789476849, y=-6.054918770666082e-05,
        s=.20172824570010092, q=1/.016865361132352322, rho=3.437662552950719e-05, c=0.)
    # Reflection canonicalisation can recover this formerly skipped request.
    # If it executes, require accuracy against a fresh VBM uniform reference;
    # if the route is skipped, retain the explicit infinite-error contract.
    recovered_reference=1.0024676281318516
    route_ok=(not skipped['estimated_target_met'] and skipped['stop_reason']=='route_not_executed'
        and math.isinf(skipped['error_indicator'])) if skipped['stop_reason']=='route_not_executed' else (
        skipped['estimated_target_met'] and abs(skipped['magnification']/recovered_reference-1)<=1e-4)
    record('upstream_route_skip_or_recovery',route_ok,result=skipped,reference=recovered_reference)
    # Counterexample: a source-centre-only angular guard missed a wing
    # already present in the coarse discovery footprint.
    wing=call(x=.06425474103755756,y=-.03874449718131291,s=2.3091029719179157,
        q=1/.06978542551163065,rho=.02920185180106385,c=0.,tol=1e-3)
    wing_reference=11.850079854586937
    record('discovery_wing_counterexample', wing['estimated_target_met'] and
        abs(wing['magnification']/wing_reference-1)<=1e-3, result=wing, reference=wing_reference)
    result=dict(native_path=m.__file__, checks=rows, failures=sum(not r['passed'] for r in rows))
    OUT.mkdir(exist_ok=True)
    (OUT/'api-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print('checks',len(rows),'failures',result['failures'])
    for r in rows:
        if not r['passed']: print(r['name'],r.get('relative_error'),r.get('relative_difference'),r.get('result'))
    return int(result['failures']>0)

if __name__=='__main__':
    sys.exit(main())
