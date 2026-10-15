"""Apply the close-root geometry overlay used by the compiled gradient API."""
from pathlib import Path
import shutil,os,argparse
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=root/'build-algebraic-boundary/fused-gradient/api-include')
parser.add_argument('--target',type=Path,default=Path(os.environ.get('LCBININT_GEOMETRY_INCLUDE',str(root/'build-algebraic-boundary/gradient-probe/geometry-include'))))
args=parser.parse_args();src=args.source;target=args.target
if src.resolve()!=target.resolve():shutil.copytree(src,target,dirs_exist_ok=True)
# A comparison build may copy an already overlaid default API tree.
frontier=target/'lcbinint/magnification/holonomic/seeded_radial_frontier.hpp'
if 'geometry_experiment::detect(pf,topo,seeds' in frontier.read_text():
    raise SystemExit(0)
shutil.copyfile(out/'fused_geometry.hpp',target/'geometry.hpp')
(target/'fused_geometry_screen.hpp').write_text((out/'fused_geometry_screen.hpp').read_text().replace('lcbinint::holonomic::geometry_experiment','lcbinint::holonomic_fused::geometry_experiment'))
p=target/'geometry.hpp';s=p.read_text().replace('lcbinint::holonomic::geometry_experiment','lcbinint::holonomic_fused::geometry_experiment');s=s.replace('struct Pair {','struct Pair {').replace('\n}\n}\n','\n}\ninline thread_local const std::vector<Pair>* active=nullptr;\nstruct Scope {const std::vector<Pair>* previous;explicit Scope(const std::vector<Pair>&p):previous(active){active=&p;}~Scope(){active=previous;}};\n}\n');p.write_text(s)
p=target/'lcbinint/magnification/holonomic/seeded_radial_frontier.hpp';s=p.read_text();s=s.replace('#include <complex>','#include <complex>\n#include "geometry.hpp"')
at='    AdaptiveConfig exploration_cfg=cfg;'
insert='''    bool geometry_complete=false;
    auto pairs=geometry_experiment::detect(pf,topo,seeds,&geometry_complete,(1<<cfg.initial_level)-1);
    pairs.erase(std::remove_if(pairs.begin(),pairs.end(),[&](const auto&pair){
        if(pair.scale>=(pair.band_hi-pair.band_lo)/((1<<cfg.initial_level)-1))return true;
        if(!pair.real||pair.existing)return false;
        auto known=[&](double r){for(auto e:topo.events)if(std::fabs(e.radius-r)<std::min(pair.scale*.25,1e-10*(1+r)))return true;return false;};
        return known(pair.left)&&known(pair.right);
    }),pairs.end());
    if(pairs.empty()&&geometry_complete){out.frontier.evaluated_nodes=workspace.samples.size();out.integral.stats.topology_ms=out.frontier.planning_ms;return out;}
    const bool retain_geometry=std::all_of(pairs.begin(),pairs.end(),[](const auto&pair){return pair.existing;});
    bool geometry_ok=true;
    for(const auto&pair:pairs)if(pair.real){
        for(double r:{pair.left,pair.right}){
            bool known=false;for(auto e:topo.events)if(std::fabs(e.radius-r)<std::min(pair.scale*.25,1e-10*(1+r)))known=true;
            if(known)continue;
            RadialEvent event;
            if(local_event(r,r-pair.scale*.5,r+pair.scale*.5,event))topo.events.push_back(event);
            else geometry_ok=false;
        }
    }
    if(!geometry_ok){for(int j=0;j<5;++j){out.integral.grad_mu[j]=std::numeric_limits<double>::quiet_NaN();out.integral.grad_quality[j]=GradientQuality::Invalid;out.integral.grad_reason[j]=GradientReason::TopologyUnresolved;}return out;}
    std::vector<CellPlan> repaired_cells;
    for(const auto&cell:topo.cells){
        if(cell.kind==ArcKind::kEmpty){repaired_cells.push_back(cell);continue;}
        std::vector<double>cuts{cell.r_lo,cell.r_hi};
        for(const auto&e:topo.events)if(e.radius>cell.r_lo&&e.radius<cell.r_hi)cuts.push_back(e.radius);
        std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
        for(size_t k=1;k<cuts.size();++k){auto c=cell;c.r_lo=cuts[k-1];c.r_hi=cuts[k];c.r_mid=.5*(c.r_lo+c.r_hi);c.n_crossings=-1;repaired_cells.push_back(c);}
    }
    for(size_t k=0;k<repaired_cells.size();++k)repaired_cells[k].index=int(k);
    topo.cells=std::move(repaired_cells);
    geometry_experiment::Scope geometry_scope(pairs);
'''
assert s.count(at)==1
insert += '''    if(!pairs.empty()){
        const auto first_nodes=out.integral.stats.node_evaluations;
        AdaptiveConfig extra_cfg=exploration_cfg;
        extra_cfg.max_node_evals=first_nodes<cfg.max_node_evals?cfg.max_node_evals-first_nodes:0;
        if(extra_cfg.max_node_evals>0)out.integral=flux_adaptive_integrate(p,u,topo,extra_cfg,workspace,retain_geometry,
            cfg.collect_diagnostics?nullptr:&epoch_event_errors);
        else for(int j=0;j<5;++j){out.integral.grad_mu[j]=std::numeric_limits<double>::quiet_NaN();out.integral.grad_quality[j]=GradientQuality::Invalid;out.integral.grad_reason[j]=GradientReason::BudgetExceeded;}
        out.integral.stats.node_evaluations+=first_nodes;
    }
'''
s=s.replace('    out.frontier.evaluated_nodes=workspace.samples.size();',insert+'    out.frontier.evaluated_nodes=workspace.samples.size();')
s=s.replace('    out.frontier.evaluated_nodes=workspace.samples.size();','''    if(!geometry_complete)for(int j=0;j<5;++j){out.integral.grad_mu[j]=std::numeric_limits<double>::quiet_NaN();out.integral.grad_quality[j]=GradientQuality::Invalid;out.integral.grad_reason[j]=GradientReason::TopologyUnresolved;}
    out.frontier.evaluated_nodes=workspace.samples.size();''')
p.write_text(s)
p=target/'fused_gaussian.hpp' ;s=p.read_text();s=s.replace('#pragma once','#pragma once\n#include "geometry.hpp"\n#include <cstdio>')
s=s.replace(' case 14:', ' case 64:return fixed_gaussian_rule<64>();\n case 128:return fixed_gaussian_rule<128>();\n case 14:')
s=s.replace('  if(panel.depth==0){','''  const geometry_experiment::Pair* hot=nullptr;
  if(geometry_experiment::active)for(const auto&pair:*geometry_experiment::active){
   const double mid=.5*(panel.map.a+panel.map.b);
   const double matching=std::min(pair.scale*.25,1e-10*(1+std::fabs(pair.center)));
   if(pair.real && !(std::fabs(panel.map.a-pair.left)<matching||std::fabs(panel.map.b-pair.left)<matching||std::fabs(panel.map.a-pair.right)<matching||std::fabs(panel.map.b-pair.right)<matching))continue;
   if(mid>pair.band_lo&&mid<pair.band_hi && (!hot||std::fabs(mid-pair.center)<std::fabs(mid-hot->center)))hot=&pair;
  }
  if(panel.depth==0&&!hot){''')
s=s.replace('  if(panel.depth==0&&!hot){',(out/'fused_geometry_pair.inc').read_text()+'  if(panel.depth==0&&!hot){')
# Once topology requires a special replay, replay its smooth background too.
# Keeping the old depth-zero derivative estimate leaves uncancelled background
# quadrature error even when the close-pair integral is well resolved.
s=s.replace('if(panel.depth==0&&!hot){', 'if(panel.depth==0&&!hot&&!(geometry_experiment::active&&!geometry_experiment::active->empty())){')
s=s.replace('const auto&rule=gaussian_rule(((1<<level)-1)*2);','const auto&rule=gaussian_rule(hot?64:((1<<level)-1)*2);')
s=s.replace('   auto mapped=panel.map(x);','   auto map=panel.map;if(hot&&panel.weighted)map.left=map.right=true;\n   auto mapped=map(x);')
s=s.replace('   const double R=mapped[0],jac=mapped[1]*half;','''   double jac=mapped[1]*half;
   if(hot&&!hot->real){
    const double lo=map(panel.xl)[0],hi=map(panel.xr)[0];
    FoldRadialMap hyperbolic{std::asinh((lo-hot->center)/hot->scale),std::asinh((hi-hot->center)/hot->scale),map.left&&panel.xl==-1,map.right&&panel.xr==1};
    const auto y=hyperbolic(rule.x[k]);mapped[0]=hot->center+hot->scale*std::sinh(y[0]);jac=hot->scale*std::cosh(y[0])*y[1];
   }
   double radius_lo=0;
   const double R=mapped[0];''');s=s.replace('if(!sample.reliable){valid=false;break;}','''if(!sample.reliable){if(std::getenv("LCBININT_GEOMETRY_DEBUG"))std::fprintf(stderr,"invalid R=%.17g jac=%.17g a=%.17g b=%.17g reason=%d hot_center=%.17g scale=%.17g real=%d\\n",R,jac,panel.map.a,panel.map.b,int(sample.reject_reason),hot?hot->center:0,hot?hot->scale:0,hot?hot->real:0);valid=false;break;}''')
s=s.replace('   const double R=mapped[0];',(out/'fused_geometry_exterior.inc').read_text()+'   const double R=mapped[0];')
s=s.replace('anchor,false,0,level,nullptr','anchor,false,radius_lo,level,nullptr').replace('nullptr,true,0,level,nullptr','nullptr,true,radius_lo,level,nullptr')
p.write_text(s)
p=target/'lcbinint/magnification/holonomic/adaptive_epoch.hpp'  ;s=p.read_text();s=s.replace('result.value_converged && deferred_children)', 'result.value_converged && (deferred_children || (geometry_experiment::active&&!geometry_experiment::active->empty())))');p.write_text(s)
