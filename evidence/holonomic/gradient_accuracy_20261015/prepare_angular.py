from pathlib import Path
import shutil, hashlib, json
root=Path(__file__).resolve().parents[3]
build=root/'build-algebraic-boundary/gradient-probe'
target=build/'angular511-include'
shutil.copytree(build/'api-include',target,dirs_exist_ok=True)
p=target/'fused_k.hpp';s=p.read_text()
assert s.count('std::array<NestedAngularPoint,31> points;')==1
s=s.replace('std::array<NestedAngularPoint,31> points;','std::array<NestedAngularPoint,511> points;')
needle='    out.ok=state==1;return out;'
assert s.count(needle)==1
s=s.replace(needle,'''    if(state==0)state=stage(std::integral_constant<int,63>{});
    if(state==0)state=stage(std::integral_constant<int,127>{});
    if(state==0)state=stage(std::integral_constant<int,255>{});
    if(state==0)state=stage(std::integral_constant<int,511>{});
'''+needle)
p.write_text(s)
(build/'angular511_manifest.json').write_text(json.dumps({'base':hashlib.sha256((build/'api-include/fused_k.hpp').read_bytes()).hexdigest(),'candidate':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
