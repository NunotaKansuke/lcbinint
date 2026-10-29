"""Serial matched public-API timings; retain failures and every measured output."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
E = D.parent
PACKAGE = E / 'triple_ld_inner_precision_20261029/physical-final-package'
EXPECTED = '863253398bab41c1d9fbc3018282234fbad67ad3ac57ad9d649ea9b32cb5f7df'
ENGINES = ['native-value', 'native-gradient', 'SinglePoly', 'MultiPoly', 'NoPoly']
IDS = [192, 24, 54, 222, 16, 80, 116, 200, 21, 400, 403, 433, 439,
       460, 467, 484, 499, 515, 514, 940, 972, 1508, 1070]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.with_suffix('.writing').write_text(json.dumps(value, indent=2, allow_nan=False))
    path.with_suffix('.writing').replace(path)


def worker(case, engine):
    os.sched_setaffinity(0, {0})
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 62))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    inp = D / 'inputs' / f'{case:04d}.json'
    c = json.loads(inp.read_text())
    p = c['params']
    meta = dict(case=case, engine=engine, input_sha256=sha(inp), rtol=c['rtol'],
                limb=c['limb'], epochs=len(c['epochs']), affinity=[0])
    if engine.startswith('native'):
        sys.meta_path = [f for f in sys.meta_path if '_lcbinint_editable' not in str(f)]
        sys.path.insert(0, str(PACKAGE))
        import lcbinint as lc
        meta['binary_sha256'] = sha(lc._native.__file__)
        assert meta['binary_sha256'] == EXPECTED
        m = lc.LightCurve(lens='triple', param_type='vbm', ld_c=c['limb'],
                         options=lc.Options(finite_source_tol=0., finite_source_reltol=c['rtol']))
        grad = engine == 'native-gradient'

        def evaluate():
            o = m.magnification(c['epochs'], p, with_gradient=grad)
            assert all(o.status == 0)
            if grad:
                assert o.gradient_valid.all() and math.isfinite(float(abs(o.gradients).max()))
            return o.magnifications.tolist()
    else:
        import VBMicrolensing as V
        prov = json.loads((E / 'triple_ld_inner_precision_20261029/PROVENANCE.json').read_text())
        meta['binary_sha256'] = sha(prov['vbm_native'])
        assert meta['binary_sha256'] == prov['vbm_sha256']
        v = V.VBMicrolensing()
        v.SetMethod(getattr(v.Method, dict(SinglePoly='Singlepoly', MultiPoly='Multipoly', NoPoly='Nopoly')[engine]))
        v.Tol = 0.
        v.RelTol = c['rtol']
        v.a1 = c['limb']
        v.a2 = 0.
        v.SetLDprofile(v.LDlinear)
        vp = [math.log(p[k]) if k in ['s', 'q', 'rho', 'tE', 'sep2', 'q2'] else p[k]
              for k in ['s', 'q', 'u0', 'alpha', 'rho', 'tE', 't0', 'sep2', 'q2', 'ang']]

        def evaluate():
            return list(v.TripleLightCurve(vp, c['epochs'])[0])

    outputs = []
    for phase, count in [('cold', 1), ('warmup', 2), ('warm', 9)]:
        for _ in range(count):
            start = time.thread_time()
            mu = evaluate()
            elapsed = time.thread_time() - start
            assert len(mu) == len(c['epochs']) and all(math.isfinite(x) and x > 0 for x in mu)
            outputs.append(dict(phase=phase, cpu_seconds=elapsed, mu=mu))
    meta.update(outputs=outputs, warm=statistics.median(r['cpu_seconds'] for r in outputs if r['phase'] == 'warm'),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return meta


if __name__ == '__main__':
    if '--worker' in sys.argv:
        print(json.dumps(worker(int(sys.argv[2]), sys.argv[3]), allow_nan=False), flush=True)
        raise SystemExit()
    (D / 'inputs').mkdir(exist_ok=True)
    (D / 'rows').mkdir(exist_ok=True)
    cases = json.loads((E / 'triple_event_precision_20261029/cases.json').read_text())
    for i in IDS:
        write(D / 'inputs' / f'{i:04d}.json', cases[i])
    rows = []
    start = time.time()
    for offset, i in enumerate(IDS):
        order = ENGINES[offset % 5:] + ENGINES[:offset % 5]
        for engine in order:
            path = D / 'rows' / f'{i:04d}-{engine}.json'
            if path.exists():
                row = json.loads(path.read_text())
            else:
                try:
                    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(i), engine],
                                       capture_output=True, text=True, timeout=100,
                                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    row = json.loads(r.stdout) if r.returncode == 0 else dict(case=i, engine=engine,
                              returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
                except Exception as exc:
                    row = dict(case=i, engine=engine, error=repr(exc))
                write(path, row)
            rows.append(row)
            write(D / 'status.json', dict(state='RUNNING', completed=len(rows), total=len(IDS)*5,
                                         elapsed=time.time()-start))
            print(i, engine, row.get('warm', row.get('returncode', row.get('error'))), flush=True)
    write(D / 'status.json', dict(state='COMPLETE', completed=len(rows), total=len(IDS)*5,
                                 elapsed=time.time()-start))
