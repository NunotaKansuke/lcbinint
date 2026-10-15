from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[3];build=root/'build-algebraic-boundary/gradient-probe'
for source,name in [('api-include','fold-include'),('direct256-include','direct_fold-include')]:
 target=build/name;shutil.copytree(build/source,target,dirs_exist_ok=True)
 p=target/'lcbinint/magnification/holonomic/seeded_radial_frontier.hpp';s=p.read_text()
 needle='exploration_cfg.weighted_radial=true;';assert s.count(needle)==1
 p.write_text(s.replace(needle,'exploration_cfg.weighted_radial=false;'))
 (build/(name+'_manifest.json')).write_text(json.dumps({'base':hashlib.sha256((build/source/'lcbinint/magnification/holonomic/seeded_radial_frontier.hpp').read_bytes()).hexdigest(),'candidate':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
