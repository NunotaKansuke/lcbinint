"""Put the optional compiled kernel in a private namespace to avoid ODR clashes."""
import json
import hashlib
import shutil
from pathlib import Path

root=Path(__file__).resolve().parents[3]
source=root/'build-seeded-events/fused-gradient/include'
target=root/'build-seeded-events/fused-gradient/api-include'
shutil.copytree(source,target,dirs_exist_ok=True)
headers=set(target.rglob('*.hpp'))
for name in ('fused_motion.hpp','fused_k.hpp'):
    path=target/name
    shutil.copyfile(Path(__file__).with_name(name),path)
    headers.add(path)
for path in sorted(headers):
    path.write_text(path.read_text().replace('lcbinint::holonomic','lcbinint::holonomic_fused'))
manifest={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(headers)}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
