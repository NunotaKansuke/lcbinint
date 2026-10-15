from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[3];build=root/'build-algebraic-boundary/gradient-probe'
for source,name in [('api-include','post2-include'),('direct256-include','direct_post2-include'),('allmotion-include','allmotion_post2-include')]:
 target=build/name;shutil.copytree(build/source,target,dirs_exist_ok=True)
 p=target/'lcbinint/magnification/holonomic/adaptive_radial.hpp';s=p.read_text()
 needle='    bool final_value_pass=false,final_gradient_pass=false;'
 assert s.count(needle)==1
 addition='''    // Research: fixed tangent enrichment after value acceptance. No gradient
    // error, quality, or tolerance participates in choosing this work.
    if(cfg.fused_value_mesh && value_snapshot.valid && !invalid) {
        gradient_phase=false;
        for(size_t i=0;i<w.panels.size();++i)if(w.panels[i].active) {
            const int target=std::min(cfg.max_level,w.panels[i].level+2);
            if(target<=w.panels[i].level)continue;
            const auto enriched=refine(i,target);
            if(enriched!=AdaptiveStop::Converged) {
                failure=enriched;
                if(enriched!=AdaptiveStop::BudgetExceeded)invalid=true;
                break;
            }
        }
    }
'''
 p.write_text(s.replace(needle,addition+needle))
 (build/(name+'_manifest.json')).write_text(json.dumps({'base':hashlib.sha256((build/source/'lcbinint/magnification/holonomic/adaptive_radial.hpp').read_bytes()).hexdigest(),'candidate':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
