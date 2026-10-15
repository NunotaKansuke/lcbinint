"""Current-kernel seed-inclusive timings, with resumable bounded shards."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'tests/diagnostics/results/recal2026/predictive-route-20261015'
RUNNER = ROOT / 'tests/diagnostics/results/recal2026/refinement-work-optimization-20261015/selected-runner/runner'

def main():
    os.sched_setaffinity(0, {0})
    env={k:v for k,v in os.environ.items() if not k.startswith('LCBININT_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    lines=(OUT/'selected-inputs.tsv').read_text().splitlines()
    identity=json.loads((OUT/'identity.json').read_text())
    assert hashlib.sha256(RUNNER.read_bytes()).hexdigest()==identity['runner_sha256']
    for start in range(0,len(lines),16):
        stem=OUT/f'timing-{start:04d}'; inp=stem.with_suffix('.inputs'); raw=stem.with_suffix('.tsv')
        chunk=lines[start:start+16]; inp.write_text('\n'.join(chunk)+'\n')
        valid=False
        if raw.exists():
            rows=list(csv.DictReader(raw.open(),delimiter='\t'))
            expected={(l.split()[0],m,str(t),str(r)) for l in chunk for m in ['cartesian','polar'] for t in [0.001,0.0001] for r in range(3)}
            actual={(r['sample_id'],r['method'],str(float(r['target'])),r['repeat']) for r in rows}
            valid=actual==expected and len(rows)==len(expected)
        if not valid:
            with stem.with_suffix('.log').open('w') as log:
                subprocess.run([str(RUNNER),str(inp),str(raw),'16','2'],env=env,stdout=log,stderr=log,check=True,timeout=180)
            rows=list(csv.DictReader(raw.open(),delimiter='\t'))
            assert len(rows)==len(chunk)*12, 'Incomplete shard; preserve logs and investigate'
        (OUT/'progress.json').write_text(json.dumps(dict(completed=min(start+16,len(lines)),total=len(lines)))+'\n')
        print(min(start+16,len(lines)), '/',len(lines),flush=True)

if __name__=='__main__':main()
