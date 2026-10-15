import os
"""Put the optional compiled kernel in a private namespace to avoid ODR clashes."""
import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[3]
build=Path(os.environ.get('LCBININT_FUSED_BUILD', str(root/'build-seeded-events/fused-gradient')))
source=build/'include'
target=build/'api-include'
shutil.copytree(source,target,dirs_exist_ok=True)
headers=set(target.rglob('*.hpp'))
for name in ('fused_motion.hpp','fused_k.hpp','fused_gaussian.hpp'):
    path=target/name
    shutil.copyfile(Path(__file__).with_name(name),path)
    headers.add(path)
for path in sorted(headers):
    path.write_text(path.read_text().replace('lcbinint::holonomic','lcbinint::holonomic_fused'))
# Close-pair screening and regularized replay are the default compiled lane.
subprocess.run([sys.executable,str(Path(__file__).with_name('build_geometry_overlay.py')),
                '--source',str(target),'--target',str(target)],check=True)
headers=set(target.rglob('*.hpp'))
manifest={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(headers)}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
