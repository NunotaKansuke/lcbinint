"""Preserve all text evidence (including failed attempts); verify every member."""
from pathlib import Path
import gzip,hashlib,json,tarfile
P=Path(__file__).resolve().parent
excluded={'REPORT.md','archive.json','raw-evidence.tar.gz','raw-evidence-manifest.json.gz'}
files=[]
for p in sorted(P.rglob('*')):
 if p.is_symlink()or not p.is_file()or p.name in excluded:continue
 rel=p.relative_to(P)
 if any(x.endswith('-package')or x in {'compiler-tmp','__pycache__'}for x in rel.parts):continue
 data=p.read_bytes()
 if data[:4]==b'\x7fELF':continue
 try:data.decode('utf8')
 except UnicodeDecodeError:continue
 files.append(p)
manifest={str(p.relative_to(P)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files}
m=P/'raw-evidence-manifest.json.gz'
with gzip.open(m,'wt')as f:json.dump(manifest,f,sort_keys=True)
a=P/'raw-evidence.tar.gz'
with tarfile.open(a,'w:gz',compresslevel=6)as t:
 for p in files:t.add(p,arcname=str(p.relative_to(P)),recursive=False)
with tarfile.open(a,'r:gz')as t:
 members=t.getmembers();assert len(members)==len(manifest)
 for member in members:
  data=t.extractfile(member).read();v=manifest[member.name];assert len(data)==v['size']and hashlib.sha256(data).hexdigest()==v['sha256'],member.name
out=dict(files=len(files),uncompressed_bytes=sum(v['size']for v in manifest.values()),verified_all_members=True,artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in [a,m]})
(P/'archive.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
