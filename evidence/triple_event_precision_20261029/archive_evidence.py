"""Archive text evidence, excluding binaries and package copies; verify all members."""
from pathlib import Path
import gzip,hashlib,json,tarfile
P=Path(__file__).resolve().parent
exclude={'raw-evidence.tar.gz','raw-evidence-manifest.json.gz','archive.json','REPORT.md'}
allowed={'.py','.cpp','.hpp','.json','.jsonl','.log','.txt','.patch','.input','.md'}
files=[]
for p in sorted(P.rglob('*')):
    rel=p.relative_to(P)
    if p.is_symlink() or not p.is_file() or p.name in exclude: continue
    if any(x.endswith('-package') or x in {'compiler-tmp','before-include','__pycache__'} for x in rel.parts): continue
    if p.suffix not in allowed: continue
    if p.open('rb').read(4)==b'\x7fELF': continue
    files.append(p)
manifest={str(p.relative_to(P)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
m=P/'raw-evidence-manifest.json.gz'
with gzip.open(m,'wt') as f: json.dump(manifest,f,sort_keys=True)
a=P/'raw-evidence.tar.gz'
with tarfile.open(a,'w:gz',compresslevel=6) as t:
    for p in files: t.add(p,arcname=str(p.relative_to(P)),recursive=False)
with tarfile.open(a,'r:gz') as t:
    members=t.getmembers();assert len(members)==len(manifest)
    for member in members:
        content=t.extractfile(member).read();ref=manifest[member.name]
        assert len(content)==ref['size'] and hashlib.sha256(content).hexdigest()==ref['sha256'],member.name
out={'files':len(files),'uncompressed_bytes':sum(v['size']for v in manifest.values()),'verified_all_members':True,'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in [a,m]}}
(P/'archive.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
