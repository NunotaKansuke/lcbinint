"""Freeze raw evidence and source snapshots; derived native binaries stay local."""
from pathlib import Path
import json,hashlib,tarfile,gzip
P=Path(__file__).resolve().parent
excluded={'raw-evidence.tar.gz','raw-evidence-manifest.json','raw-evidence-archive.json'}
files=[]
for f in P.rglob('*'):
 if not f.is_file()or f.name in excluded:continue
 rel=f.relative_to(P)
 if any(a.endswith('-package')or a in ['__pycache__','compiler-tmp']for a in rel.parts):continue
 if f.suffix not in ['.py','.json','.jsonl','.hpp','.cpp','.log','.md','.txt']:continue
 files.append(f)
files.sort()
manifest=[dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in files]
(P/'raw-evidence-manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'raw-evidence-manifest.json.gz').write_bytes(gzip.compress((P/'raw-evidence-manifest.json').read_bytes(),mtime=0))
archive=P/'raw-evidence.tar.gz'
with tarfile.open(archive,'w:gz')as tar:
 for f in files:tar.add(f,arcname=str(f.relative_to(P)),recursive=False)
z=dict(files=len(files),uncompressed_bytes=sum(r['bytes']for r in manifest),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),native_binaries_excluded=True,manifest_sha256=hashlib.sha256((P/'raw-evidence-manifest.json').read_bytes()).hexdigest(),extract='tar -xzf raw-evidence.tar.gz -C THIS_EVIDENCE_DIRECTORY')
(P/'raw-evidence-archive.json').write_text(json.dumps(z,indent=2));print(z)
