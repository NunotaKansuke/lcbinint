"""Audit the ordinary-build, no-environment relative-tolerance API."""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'tests/diagnostics/results/recal2026/adaptive-production-port-20261013'
BASE = OUT.parent / 'deferred-cache-production-20261013'
os.environ.setdefault('LCBININT_PROBE_BUILD', str(ROOT / 'build-adaptive-production'))
from probe_build import activate

def main():
    native = activate()._native
    for flag in ['DIRECTIONAL_ADAPTIVE', 'LOCAL_ROW_INTERPOLATION', 'LOCAL_BOUNDARY_STRIPS',
                 'LOCAL_VERTICAL_CAPS', 'DEFECT_CORRECTION', 'DISABLE_MISSING_CAP_GUARD',
                 'DISABLE_DEFER_LATTICE_CACHE']:
        os.environ.pop('LCBININT_DIAGNOSTIC_' + flag, None)
    refs = {str(i['sample_id']): i for i in json.loads((BASE / 'inputs.json').read_text())}
    expected = {(r['sample_id'], r['target']): r for r in
                csv.DictReader((BASE / 'normal-valid-after-0.tsv').open(), delimiter='\t')}
    rows = []
    for line in (BASE / 'input.tsv').read_text().splitlines():
        columns = line.split()
        key = columns[0]
        s, q, rho, source_x, y, c = map(float, columns[1:])
        for tolerance in [1e-3, 1e-4]:
            target = str(tolerance)
            prior = expected.get((key, target))
            try:
                got = native.binary_adaptive_flux(source_x, y, s=s, q=1/q, rho=rho,
                    limb_darkening=native.LimbDarkening(c, 0), initial_bins=16,
                    max_bins=512, relative_tolerance=tolerance, max_lattice_maps=1000000)
                error = abs(got['magnification'] / refs[key]['reference'] - 1)
                fields = dict(magnification='magnification', error_indicator='indicator',
                              component_refinements='refinements', stop_reason='stop_reason')
                changes = []
                if prior is None:
                    changes.append('unexpected successful return')
                else:
                    for api, native_field in fields.items():
                        want = prior[native_field]
                        if api == 'stop_reason':
                            same = got[api] == want
                        else:
                            same = got[api] == float(want)
                        if not same:
                            changes.append(api)
                    if got['estimated_target_met'] != (prior['converged'] == '1'):
                        changes.append('estimated_target_met')
                rows.append(dict(id=key, tolerance=tolerance, error=error,
                    accepted=got['estimated_target_met'], changes=changes,
                    initial_scale=got['initial_magnification_scale'],
                    false_accept=got['estimated_target_met'] and error > tolerance))
            except Exception as error:
                rows.append(dict(id=key, tolerance=tolerance, exception=str(error),
                                 changes=['unexpected exception'] if prior is not None else []))
    result = dict(native_path=native.__file__, rows=rows,
                  mismatches=sum(bool(r['changes']) for r in rows),
                  false_accepts=sum(r.get('false_accept', False) for r in rows),
                  exceptions=sum('exception' in r for r in rows))
    (OUT / 'relative-api-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('relative API', len(rows), 'attempts, mismatches', result['mismatches'],
          'false accepts', result['false_accepts'], 'exceptions', result['exceptions'])
    if result['mismatches'] or result['false_accepts']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
