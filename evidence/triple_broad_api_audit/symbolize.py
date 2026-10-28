import json,subprocess,re,bisect,collections
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent;prod=R/'build-triple-native/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so';sym=P/'unstripped.so'
identity=json.loads((P/'symbol-identity.json').read_text())
for section in ['.text','.rodata','.eh_frame']:
 assert identity['production'][section]==identity['symbols'][section]
 pattern=r'\[\s*\d+\]\s+'+re.escape(section)+r'\s+\S+\s+(\w+)'
 a=re.search(pattern,identity['production']['sections']).group(1);b=re.search(pattern,identity['symbols']['sections']).group(1);assert a==b
public={}
for line in subprocess.check_output(['nm','-D','-C','--defined-only',str(prod)],text=True).splitlines():
 m=re.match(r'([\da-f]+)\s+\w\s+(.*)',line)
 if m:public[m[2]]=int(m[1],16)
funcs=[]
for line in subprocess.check_output(['nm','-n','-S','-C','--defined-only',str(sym)],text=True).splitlines():
 m=re.match(r'([\da-f]+)\s+([\da-f]+)\s+[tTwW]\s+(.*)',line)
 if m and int(m[2],16):funcs.append((int(m[1],16),int(m[2],16),m[3]))
funcs.sort();addresses=[x[0]for x in funcs];report=[]
frame=re.compile(r'^(\s*)([\da-f]+) (.*?)(?:\+0x([\da-f]+))? \((.*)\)$')
for f in sorted((P/'profiles').glob('*.stacks')):
 if '.symbolized.'in f.name:continue
 lines=f.read_text().splitlines();bases=collections.Counter()
 for line in lines:
  m=frame.match(line)
  if not m or m[5]!=str(prod)or m[3]not in public:continue
  bases[int(m[2],16)-public[m[3]]-int(m[4]or'0',16)]+=1
 assert bases,f
 base=bases.most_common(1)[0][0];resolved=unknown=0;new=[]
 for line in lines:
  m=frame.match(line)
  if m and m[5]==str(prod)and m[3]=='[unknown]':
   unknown+=1;vma=int(m[2],16)-base;i=bisect.bisect_right(addresses,vma)-1
   if i>=0 and vma<funcs[i][0]+funcs[i][1]:
    address,size,name=funcs[i];line=f'{m[1]}{m[2]} {name}+0x{vma-address:x} ({m[5]})';resolved+=1
  new.append(line)
 f.with_name(f.stem+'.symbolized.stacks').write_text('\n'.join(new)+'\n')
 report.append(dict(file=str(f.relative_to(P)),load_base=hex(base),base_votes={hex(k):v for k,v in bases.items()},unknown_frames=unknown,resolved_frames=resolved))
(P/'symbolization.json').write_text(json.dumps(report,indent=2)+'\n');print('resolved',sum(x['resolved_frames']for x in report),'of',sum(x['unknown_frames']for x in report),'unknown native frames')
