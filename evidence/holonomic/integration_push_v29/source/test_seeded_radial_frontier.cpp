#include <cmath>
#include <iostream>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic;
int main() {
 int checks=0,failures=0;
 auto check=[&](bool ok,const char* message){++checks;if(!ok){++failures;std::cerr<<message<<'\n';}};
 // Incremental witness collection must preserve the original union even
 // when a sample changes count/radius or becomes reliable on a later pass.
 {
  seeded_frontier_detail::SampleWitnessCollector collector;
  std::vector<AdaptiveSample> samples;
  std::vector<std::pair<double,int>> incremental,repeated;
  std::size_t prefix=0;
  for(int pass=0;pass<100;++pass) {
   samples.resize(3+pass%23);
   for(std::size_t j=0;j<samples.size();++j) {
    auto& sample=samples[j];sample.R=double(j)+.125*(pass%5);
    sample.local_crossings=((int(j)+pass)%5==0)?-1:2*((int(j)+pass)%3);
    sample.reliable=(int(j)+pass)%4!=0;
   }
   const std::pair<double,int> radial_probe{.0625*pass,2*(pass%3)};
   incremental.push_back(radial_probe);repeated.push_back(radial_probe);
   collector.append(samples,incremental);
   for(const auto& sample:samples)if(sample.reliable && sample.local_crossings!=-1)
    repeated.emplace_back(sample.R,sample.local_crossings);
   seeded_frontier_detail::sort_unique_probes(incremental,prefix);
   std::sort(repeated.begin(),repeated.end());
   repeated.erase(std::unique(repeated.begin(),repeated.end()),repeated.end());
   check(incremental==repeated,"incremental witnesses preserve complete sorted union");
   const auto previous=incremental;
   collector.append(samples,incremental);
   check(incremental==previous,"unchanged reliable samples add no duplicate witnesses");
  }
 }
 // Track an image straddling theta=pi in the reciprocal pair chart and
 // compare its returned arcs against independent cold quartic roots.
 {
  const LensParams lens{-.35,.001,.05,.1,1.,true};
  const auto frame=PrimaryFrame::from(lens);
  RootPairWarm pairs;QuarticWarm quartic;int reciprocal_nodes=0;
  for(int i=0;i<=500;++i) {
   const double R=.5+.002*i;const auto pc=boundary_quartic(R,frame).p;
   const auto cert=certify_quartic(pc);
   if(!cert.certified || cert.root_count<=0){pairs.valid=false;continue;}
   std::vector<std::array<double,2>> direct;
   const auto theta=real_root_thetas_transport(pc,R,frame,pairs,&quartic,cert.root_count,&direct);
   ArcSet actual;
   if(direct.empty())actual=arc_set_from_root_thetas(R,frame,theta);
   else{actual.kind=ArcKind::kArcs;actual.arcs=direct;}
   const auto reference=arc_set_from_root_thetas(R,frame,real_root_thetas(pc));
   check(actual.kind==reference.kind && actual.arcs.size()==reference.arcs.size(),
         "reciprocal pair chart preserves cold arc topology");
   if(actual.arcs.size()==reference.arcs.size())for(std::size_t j=0;j<actual.arcs.size();++j)
    check(std::fabs(actual.arcs[j][0]-reference.arcs[j][0])<1e-7 &&
          std::fabs(actual.arcs[j][1]-reference.arcs[j][1])<1e-7,
          "reciprocal pair angles agree with independent cold roots");
   if(pairs.valid && pairs.reciprocal)++reciprocal_nodes;
  }
  check(reciprocal_nodes>1 && pairs.warm_hits>0,"theta=pi fixture exercises reciprocal warm transport");
 }
 AdaptiveConfig cfg;cfg.tol.mu_rtol=1e-8;AdaptiveWorkspace w;
 for(int i=0;i<2;++i){AdaptivePanel panel;panel.map={double(i),double(i+1),false,false};panel.cell=i;w.panels.push_back(panel);}
 int calls=0;
 auto eval=[&](double R,double jac,int,const AdaptiveSample*) {
  ++calls;AdaptiveSample s;s.value[0]=R*R*jac;return s;
 };
 auto a=adaptive_detail::integrate(w,cfg,eval);
 check(a.value_converged&&std::fabs(a.mu-8./3)<1e-12,"polynomial integral");
 auto slots=w.panels[1].samples;int old_calls=calls;
 auto b=adaptive_detail::integrate(w,cfg,eval);
 check(b.value_converged&&std::fabs(b.mu-a.mu)<1e-12,"resumed result");
 check(calls==old_calls&&b.stats.reused_nodes>0,"resume reuses all accepted nodes");
 check(w.panels[1].samples==slots,"sample indices survive resume");
 w.panels[0].active=false;auto c=adaptive_detail::integrate(w,cfg,eval);
 check(c.value_converged&&std::fabs(c.mu-7./3)<1e-12,"inactive replaced panel is not integrated");
 check(calls==old_calls,"inactive panel does not cause evaluations");
 LensParams p{.1,.2,.01,.1,1.,true};
 auto invalid=epoch_seeded_radial_frontier(p,0,{},cfg,w);
 check(invalid.integral.stop==AdaptiveStop::InvalidConfig,"empty seeds rejected");
 cfg.require_bound=true;
 auto bound=epoch_seeded_radial_frontier(p,0,{{1.,.1}},cfg,w);
 check(bound.integral.stop==AdaptiveStop::BoundUnavailable,"no false mathematical bound");
 cfg.require_bound=false;cfg.tol.mu_rtol=1e-4;
 p={-.15102116501928589,-.056165336366353361,.0534453909027384,
    .30000003000000003,.60000006000000006,true};
 auto topo=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
 auto fresh=flux_adaptive_integrate(p,0,topo,cfg,w);
 check(fresh.value_converged,"resume fixture converges");
 int damaged=-1;
 for(const auto& panel:w.panels)if(panel.active)
   for(int index:panel.samples)if(index>=0){damaged=index;break;}
 check(damaged>=0,"resume fixture has cached nodes");
 if(damaged>=0) {
  w.samples[damaged].reliable=false;w.samples[damaged].value[0]=-123.;
  auto repaired=flux_adaptive_integrate(p,0,topo,cfg,w,true);
  check(repaired.value_converged && std::fabs(repaired.mu-fresh.mu)<1e-12,
        "rejected cached node is reevaluated rather than accepted");
  check(repaired.stats.node_evaluations>=1 && repaired.stats.reused_nodes>0,
        "only damaged node is reevaluated while valid nodes survive");
 }
 int root=-1;
 for(std::size_t i=0;i<w.panels.size();++i)
  if(w.panels[i].active && w.panels[i].parent<0){root=static_cast<int>(i);break;}
 check(root>=0,"fixture has unsplit active root panel");
 if(root>=0) {
  auto left=w.panels[root],right=left;
  left.parent=right.parent=root;left.depth=right.depth=1;
  left.xr=right.xl=.5*(left.xl+left.xr);
  left.right_radius_lo=0;right.left_radius_lo=0;
  left.samples.fill(-1);right.samples.fill(-1);left.level=right.level=0;
  w.panels[root].active=false;w.panels.push_back(left);w.panels.push_back(right);
  auto resumed=flux_adaptive_integrate(p,0,topo,cfg,w,true);
  check(resumed.value_converged && std::fabs(resumed.mu-fresh.mu)<cfg.tol.mu_rtol*std::fabs(fresh.mu),
        "resume preserves both children with asymmetric endpoint remainders");
 }
 {
  std::vector<std::pair<double,adaptive_detail::EventLocation>> cache;
  AdaptiveWorkspace a,b,c;
  auto first=flux_adaptive_integrate(p,.5,topo,cfg,a,false,&cache);
  const auto size=cache.size();
  auto cached=flux_adaptive_integrate(p,.5,topo,cfg,b,false,&cache);
  auto uncached=flux_adaptive_integrate(p,.5,topo,cfg,c);
  check(!cache.empty() && cache.size()==size,"event cache reuses the same epoch records");
  check(first.mu==cached.mu && cached.mu==uncached.mu,
        "event uncertainty reuse preserves the LD integral exactly");
  check(cached.stats.event_dd_checks==0 && cached.stats.event_double_checks==0,
        "cached event uncertainty avoids duplicate precision checks");
 }
 p.rho=10.;const auto pf=PrimaryFrame::from(p);
 CellPlan dynamic{0,0,40,20,ArcKind::kArcs,-1,Status::OK};
 auto full=adaptive_detail::mapped_radius(1.5,1,p,0,pf,dynamic,false,false,nullptr,&cfg);
 auto empty=adaptive_detail::mapped_radius(30,1,p,0,pf,dynamic,false,false,nullptr,&cfg);
 check(full.reliable&&full.local_crossings==-2,"full circle is distinct from empty support");
 check(empty.reliable&&empty.local_crossings==0,"empty support is recorded locally");
 AdaptiveWorkspace yielded;AdaptivePanel yp;yp.map={0.,1.,false,false};yielded.panels.push_back(yp);
 AdaptiveConfig yc;yc.gradient_policy=GradientPolicy::None;yc.yield_on_local_topology_change=true;
 auto topology_eval=[](double R,double jac,int,const AdaptiveSample*) {
  AdaptiveSample sample;sample.value[0]=jac;sample.local_crossings=R<.5?2:4;return sample;
 };
 auto interrupted=adaptive_detail::integrate(yielded,yc,topology_eval);
 check(interrupted.stop==AdaptiveStop::TopologyUnresolved && !interrupted.value_converged,
       "observed count change yields before claiming convergence");
 check(yielded.samples.size()<=7 && !yielded.samples.empty(),
       "count change stops during initial level rather than after full refinement");
 bool all_valid=true;for(const auto& sample:yielded.samples)all_valid=all_valid&&sample.reliable;
 check(all_valid,"yield preserves reliable evaluated samples");
 yc.yield_on_local_topology_change=false;
 auto continued=adaptive_detail::integrate(yielded,yc,topology_eval);
 check(continued.value_converged && std::fabs(continued.mu-1.)<1e-12 && continued.stats.reused_nodes>0,
       "partially evaluated panel resumes with sample reuse");
 std::cout<<checks<<" checks "<<failures<<" failures\n";return failures?1:0;
}
