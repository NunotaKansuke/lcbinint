from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[3];build=root/'build-algebraic-boundary/gradient-probe'
target=build/'allmotion_posth-include';shutil.copytree(build/'allmotion-include',target,dirs_exist_ok=True)
p=target/'lcbinint/magnification/holonomic/adaptive_radial.hpp';s=p.read_text()
needle='    bool final_value_pass=false,final_gradient_pass=false;';assert s.count(needle)==1
addition='''    // Fixed, one-pass tangent subdivision; no derivative error or tolerance
    // enters the decision. Keep the accepted primal snapshot below.
    if(cfg.fused_value_mesh && value_snapshot.valid && !invalid) {
        gradient_phase=false;
        const size_t parents=w.panels.size();
        for(size_t i=0;i<parents;++i)if(w.panels[i].active) {
            const auto p=w.panels[i];
            if(w.panels.size()+2>cfg.max_panels) {failure=AdaptiveStop::BudgetExceeded;break;}
            AdaptivePanel l,r;l.parent=r.parent=int(i);l.map=r.map=p.map;
            l.cell=r.cell=p.cell;l.cell_end=r.cell_end=p.cell_end;
            l.evaluator_cell=r.evaluator_cell=p.evaluator_cell;l.depth=r.depth=p.depth+1;
            l.left_uncertainty=p.left_uncertainty;r.right_uncertainty=p.right_uncertainty;
            l.left_radius_lo=p.left_radius_lo;r.right_radius_lo=p.right_radius_lo;
            l.xl=p.xl;l.xr=r.xl=0.5*(p.xl+p.xr);r.xr=p.xr;
            if(p.weighted) {
                const long double mid=((long double)p.map.a+p.left_radius_lo+(long double)p.map.b+p.right_radius_lo)*.5L;
                const double midpoint=double(mid),lo=double(mid-midpoint);
                l.map={p.map.a,midpoint,true,false};r.map={midpoint,p.map.b,false,true};
                l.right_radius_lo=r.left_radius_lo=lo;l.xl=r.xl=-1;l.xr=r.xr=1;
            }
            l.active=r.active=false;
            w.panels.push_back(l);w.panels.push_back(r);
            auto status=refine(w.panels.size()-2,p.level);
            if(status==AdaptiveStop::Converged)status=refine(w.panels.size()-1,p.level);
            if(status!=AdaptiveStop::Converged) {failure=status;if(status!=AdaptiveStop::BudgetExceeded)invalid=true;break;}
            w.panels[i].active=false;w.panels[w.panels.size()-2].active=true;w.panels.back().active=true;
        }
    }
'''
p.write_text(s.replace(needle,addition+needle))
