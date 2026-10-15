from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[3];build=root/'build-algebraic-boundary/gradient-probe'
target=build/'direct256-include';shutil.copytree(build/'moving-include',target,dirs_exist_ok=True)
p=target/'lcbinint/magnification/holonomic/adaptive_epoch.hpp';s=p.read_text()
a=s.index('        bool success=false;');b=s.index('\n    }\n#ifdef HOLO_FRONTIER_COST_PROFILE',a)
section=s[a:b];needle='        for(bool reciprocal:{false,true}) {';assert section.count(needle)==1
section=section.replace(needle,needle+'\n            if(with_jac)break; // fixed direct angular rule for gradient research')
section=section.replace('static const Cheb1<32> low;','static const Cheb1<128> low;').replace('const auto& high=ang_rule();','static const Cheb1<256> high;').replace('k<64','k<256').replace('k<32','k<128')
p.write_text(s[:a]+section+s[b:])
(build/'direct256_manifest.json').write_text(json.dumps({'base':hashlib.sha256((build/'moving-include/lcbinint/magnification/holonomic/adaptive_epoch.hpp').read_bytes()).hexdigest(),'candidate':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
