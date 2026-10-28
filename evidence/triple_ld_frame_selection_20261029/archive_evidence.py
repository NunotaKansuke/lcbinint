from pathlib import Path
import json,hashlib,gzip,tarfile,time
P=Path(__file__).resolve().parent
excluded={'compiler-tmp','profile-include','__pycache__','.pytest_cache'}
files=[]
for f in sorted(P.rglob('*')):
 if not f.is_file():continue
 parts=f.relative_to(P).parts
 if any(x in excluded or x.endswith('-package') for x in parts):continue
 if f.name in ['raw-evidence.tar.gz','raw-evidence-manifest.json.gz','archive.json','REPORT.md']:continue
 if f.suffix in ['.so','.a','.o','.pyc']:continue
 with f.open('rb')as stream:
  if stream.read(4)==b'\x7fELF':continue
 files.append(f)
manifest=[dict(path=str(f.relative_to(P)),size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in files]
with gzip.open(P/'raw-evidence-manifest.json.gz','wt')as stream:json.dump(manifest,stream)
with tarfile.open(P/'raw-evidence.tar.gz','w:gz',compresslevel=6)as tar:
 for f in files:tar.add(f,arcname=str(f.relative_to(P)),recursive=False)
with tarfile.open(P/'raw-evidence.tar.gz','r:gz')as tar:
 entries={e.name:e for e in tar.getmembers()}
 assert len(entries)==len(manifest)
 for m in manifest:assert hashlib.sha256(tar.extractfile(entries[m['path']]).read()).hexdigest()==m['sha256'],m['path']
z=dict(files=len(manifest),uncompressed_bytes=sum(x['size']for x in manifest),verified=True,created=time.time(),accepted_native_sha256=json.loads((P/'accepted-state.json').read_text())['accepted_native_sha256'],candidate_native_sha256=json.loads((P/'accepted-state.json').read_text())['candidate_native_sha256'],adopted=False)
for name in ['raw-evidence.tar.gz','raw-evidence-manifest.json.gz']:z[name]=dict(bytes=(P/name).stat().st_size,sha256=hashlib.sha256((P/name).read_bytes()).hexdigest())
(P/'archive.json').write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
