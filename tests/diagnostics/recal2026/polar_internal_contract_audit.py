"""Polar resource limits, recovery, mesh variation, and LD linearity audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from polar_local_adaptive_check import ROOT, load_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extreme-references', type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {0})
    native = load_native()
    matrix = json.loads((ROOT / 'tests/diagnostics/results/recal2026/'
        'polar-initial-estimator-20261014/matrix.json').read_text())['cases']
    checks = []
    radial_rules = set()

    def record(name, passed, **data):
        checks.append(dict(name=name, passed=bool(passed), **data))

    def call(inp, c=None, **options):
        return native.binary_adaptive_polar_flux(inp['time'], inp['y'],
            s=inp['s'], q=1/inp['q'], rho=inp['rho'],
            limb_darkening=native.LimbDarkening(inp['c'] if c is None else c, 0),
            **dict(dict(relative_tolerance=1e-4), **options))

    def contract(result, options):
        radial_rules.add(result['radial_quadrature'])
        error = result['error_indicator']
        accepted = result['estimated_target_met']
        return (not math.isnan(error) and error >= 0 and
            error == result['angular_error_indicator'] + result['radial_error_indicator'] and
            (not accepted or result['angular_error_indicator'] == sum(result[k] for k in
                ('smooth_angular_error', 'topology_angular_error', 'seed_angular_error',
                 'coverage_angular_error', 'cap_angular_error'))) and
            (not accepted or (math.isfinite(error) and
                error <= result['requested_error_budget'] and
                result['stop_reason'] == 'estimated_target_met')) and
            not result['certified'] and not result['complete_error_model'] and
            result['new_lattice_maps'] <= options.get('max_lattice_maps', 1000000) and
            result['largest_radial_depth'] <= options.get('max_radial_depth', 10) and
            result['largest_angular_depth'] <= options.get('max_angular_depth', 18))

    for index in [0, 7, 15, 23, 31, 47]:
        case = matrix[index]
        inp = case['input']
        baseline = call(inp)
        discovery = baseline['discovery_lattice_maps']
        variants = [dict(initial_angular_ratio=ratio) for ratio in (16., 64., 256.)]
        variants += [dict(initial_bins=bins) for bins in (2, 4, 8, 16)]
        variants += [dict(max_angular_depth=0, relative_tolerance=1e-10),
                     dict(max_radial_depth=0, relative_tolerance=1e-10)]
        variants += [dict(max_lattice_maps=budget) for budget in
                     sorted({discovery+1, max(discovery+1, baseline['new_lattice_maps']-1),
                             max(discovery+1, 8*discovery-1)})]
        for options in variants:
            try:
                result = call(inp, **options)
                # Frozen references were selected for 1e-3..1e-5 checks;
                # they cannot validate absolute accuracy at 1e-10. Those
                # forced-limit requests check resource/acceptance contracts.
                goal = options.get('relative_tolerance', 1e-4)
                reference_usable = goal >= 1e-5
                record('resource_and_accuracy_contract', contract(result, options) and
                    (not reference_usable or not result['estimated_target_met'] or
                     abs(result['magnification']/case['reference']-1) <= goal),
                    index=index, options=options, result=result, reference=case['reference'],
                    accuracy_reference_usable=reference_usable)
            except RuntimeError as exc:
                # Discovery can fail before an adaptive result exists. This
                # must remain an explicit exception, never an accepted flux.
                record('explicit_discovery_failure', 'max_lattice_maps' in options,
                       index=index, options=options, exception=str(exc))
            recovered = call(inp)
            record('request_context_recovery', recovered == baseline, index=index)

        reflected = call(dict(inp, y=-inp['y']))
        record('reflection', contract(reflected, {}) and
            abs(reflected['magnification']/baseline['magnification']-1) < 1e-9,
            index=index, result=reflected)
        swapped=call(dict(inp,time=-inp['time'],q=1/inp['q']))
        record('mass_ratio_reflection', contract(swapped,{}) and swapped['estimated_target_met'] and
            abs(swapped['magnification']/case['reference']-1)<=1e-4,
            index=index,result=swapped,reference=case['reference'])
        uniform, dark, half = [call(inp, c=c) for c in (0., 1., .5)]
        # Unnormalised flux is linear in c. Compare independently adapted
        # results using the sum of their requested budgets, not a tighter
        # arbitrary requirement than the API was asked to satisfy.
        delta = abs((5/6)*half['magnification'] -
                    .5*uniform['magnification'] - (1/3)*dark['magnification'])
        budget = ((5/6)*half['requested_error_budget'] +
                  .5*uniform['requested_error_budget'] + (1/3)*dark['requested_error_budget'])
        record('ld_flux_linearity', delta <= budget and
               all(contract(r, {}) and r['estimated_target_met'] for r in (uniform, dark, half)),
               index=index, absolute_flux_difference=delta, budget=budget)

    record('radial_rule_coverage', {'exact_uniform', 'witnessed_gauss_chebyshev_u_1',
        'witnessed_simpson_budget_fallback'} <= radial_rules,
        observed_rules=sorted(radial_rules))
    # U_7 is zero at all seven fine-rule nodes, including the three coarse
    # nodes. g=1+U_7^2 therefore makes both rules return pi/2 although its
    # exact weighted integral is pi by Chebyshev-U orthogonality.
    def u7(x):
        old, value = 1., 2*x
        for _ in range(2, 8):
            old, value = value, 2*x*value-old
        return value
    def embedded(divisor):
        return math.pi/divisor*sum(math.sin(math.pi*j/divisor)**2 *
            (1+u7(math.cos(math.pi*j/divisor))**2) for j in range(1, divisor))
    coarse, fine = embedded(4), embedded(8)
    witness_mismatch = max(u7(x)**2 for x in (-1/3, 1/3))
    record('nested_radial_alias_counterexample', abs(coarse-fine) < 1e-12 and
        abs(fine-math.pi) > 1 and witness_mismatch > 0,
        coarse=coarse, fine=fine, exact=math.pi, witness_mismatch=witness_mismatch,
        scope='analytic estimator counterexample, not a physical lens input')
    radius, uncertainty = .25, .5
    swept = .5*((radius+uncertainty)**2-radius**2)
    bound = radius*uncertainty+.5*uncertainty**2
    record('boundary_swept_area_bound', radius*uncertainty < swept and bound >= swept,
           radius=radius, uncertainty=uncertainty, actual_area=swept, bound=bound)

    adjudication = ROOT/'tests/diagnostics/results/recal2026/' \
        'polar-full-accuracy-20261015/independent-uniform.json'
    refs = {str(r['id']): r for r in json.loads(adjudication.read_text())}
    for id in ('4906', '7196'):
        entry = refs[id]
        inp = {k: float(v) if k in ('s','q','rho','time','y','c') else v
               for k,v in entry['input'].items()}
        fine = entry['result']['values'][-1]
        if fine['warnings'] or fine['quadrature_indicator']/fine['value']>2e-7:
            raise RuntimeError('independent angular counterexample reference unresolved')
        for goal in (1e-3,1e-4,1e-5):
            result = call(inp, relative_tolerance=goal)
            error = abs(result['magnification']/fine['value']-1)
            record('independent_angular_counterexample', contract(result, {}) and
                   result['estimated_target_met'] and error <= goal,
                   id=id, target=goal, relative_error=error, reference=fine, result=result)

    extreme_path = ROOT / 'tests/diagnostics/results/recal2026/polar-quadrature-rework/' \
        'independent-angular-extreme-final-20261015/validation.json'
    extreme = json.loads((args.extreme_references or extreme_path).read_text())
    references = {r['id']: r for r in extreme['references']}
    for geometry in (0, 15):
        for inp in extreme['inputs'][3*geometry:3*geometry+3]:
            for goal in (1e-3, 1e-4, 1e-5):
                # The default now starts at2 and avoids the failed coarse
                # lattice. Explicitly request4 to exercise the retry itself.
                options = dict(relative_tolerance=goal, initial_bins=4)
                result = call(inp, **options)
                ref = references[inp['id']]
                record('discovery_retry_regression', contract(result, options) and
                    result['discovery_retries'] == 1 and result['discovery_source_bins'] == 2 and
                    (goal != 1e-3 or result['estimated_target_met']) and
                    (not result['estimated_target_met'] or not ref['valid'] or
                     abs(result['magnification']/ref['value']-1) <= goal),
                    id=inp['id'], options=options, result=result, reference=ref,
                    accuracy_reference_usable=ref['valid'])

    payload = dict(checks=checks, failures=sum(not r['passed'] for r in checks),
        native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT/'src/lcbinint/magnification/'
            'adaptive_polar_impl.inc').read_bytes()).hexdigest(), certified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print('checks', len(checks), 'failures', payload['failures'], flush=True)
    for row in checks:
        if not row['passed']:
            print(row['name'], row.get('index'), row.get('options'), flush=True)
    return int(payload['failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
