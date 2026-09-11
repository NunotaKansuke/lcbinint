#include <iostream>
#include <cstdlib>
#include "lcbinint/magnification/holonomic/radial_atlas_topology.hpp"
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::atlas_detail;
void check(bool b,const char* s){if(!b){std::cerr<<s<<'\n';std::exit(1);}}
int main(){
 Tensor<IQ> toy;toy.nr=1;toy.ns=2;toy.c[0][0]=IQ(.3125);toy.c[1][0]=IQ(-.5);toy.c[0][1]=IQ(-.25);toy.c[0][2]=IQ(.25);
 auto t=bernstein(toy);check(prove(t).kind==3,"ordinary Krawczyk fold");
 auto halves=split(t,0);for(double x:{.1,.4,.9}){auto a=evaluate(halves.first,IQ(x),IQ(.3)),b=evaluate(t,IQ(x*.5),IQ(.3));check(a.lo<=b.hi&&b.lo<=a.hi,"subdivision parity");}
 Tensor<IQ> two;two.nr=2;two.ns=2;two.c[0][0]=IQ(-.24);two.c[1][0]=IQ(1);two.c[2][0]=IQ(-1);two.c[0][2]=IQ(1);check(prove(bernstein(two)).kind!=1&&prove(bernstein(two)).kind!=2,"two interior contacts not excluded");
 LensParams p{.3,.2,.01,2,1,true};auto pf=PrimaryFrame::from(p);
 for(int chart=0;chart<2;++chart){Box b{.25Q,.75Q,-1,1,chart,0};auto poly=boundary<IQ>(pf,b);Q R=.4Q,s=.2Q;auto v=evaluate(poly,IQ((R-b.r0)/(b.r1-b.r0),(R-b.r0)/(b.r1-b.r0)),IQ((s-b.s0)/(b.s1-b.s0),(s-b.s0)/(b.s1-b.s0)));auto ref=local_fold_quantities<Q>(R,s,pf,chart);check(v.lo<=ref.P&&ref.P<=v.hi,"boundary/chart parity");}
 RadialAtlasWorkspace ws;RadialAtlasConfig cfg;RadialEventAtlasCache cache;
 auto cold=build_radial_event_atlas(pf,cfg,ws,&cache);check(cold.status==AtlasStatus::Complete,"cold complete");check(cold.contacts.size()==6,"six physical contacts");
 auto area=[](const RadialAtlasResult& result){Q sum=0;for(const auto& l:result.leaves)sum+=(l.box.r1-l.box.r0)*(l.box.s1-l.box.s0);return sum;};
 check(fabsq(area(cold)-4*(Q)cold.rmax)<1e-25Q,"cold cover partition has no missing area");
 auto warm=build_radial_event_atlas(pf,cfg,ws,&cache);check(warm.status==AtlasStatus::Complete&&warm.contacts.size()==cold.contacts.size(),"warm complete");check(fabsq(area(warm)-4*(Q)warm.rmax)<1e-25Q,"warm cover partition has no missing area");
 check(warm.stats.warm_reused>0&&warm.stats.event_seed_updates>0,"warm revalidates both evidence kinds");
 auto moved=pf;moved.X+=1e-5;auto a=build_radial_event_atlas(moved,cfg,ws,&cache);auto b=build_radial_event_atlas(moved,cfg,ws);check(a.status==b.status&&a.contacts.size()==b.contacts.size(),"moving warm/cold contact parity");
 for(int k=0;k<3;++k){auto shifted=pf;shifted.X+=.02*(k+1);auto current=build_radial_event_atlas(shifted,cfg,ws,&cache);
  for(size_t i=0;i<current.contacts.size();++i)for(size_t j=0;j<i;++j)check(current.contacts[i].id!=current.contacts[j].id,"warm partial seed survival keeps IDs unique");
 }
 auto old=cache.result.generation;auto limit=cfg;limit.max_boxes=0;auto failed=build_radial_event_atlas(pf,limit,ws,&cache);check(failed.status!=AtlasStatus::Complete&&cache.result.generation==old,"failure preserves cache");
 auto topo=classify_cells_from_atlas(pf,cold);check(topo.status==Status::OK,"atlas cells");AdaptiveConfig ac;ac.tol.mu_rtol=1e-4;ac.preserve_radial_offset=true;AdaptiveWorkspace aw;AtlasSampleContext ctx{&cold.contacts};AdaptiveResult val;
 {AtlasSampleScope scope(&ctx);val=flux_adaptive_integrate(p,.6,topo,ac,aw);}
 ac.preserve_radial_offset=false;
 auto base=epoch_adaptive(p,.6,ac,aw);check(val.value_converged&&base.value_converged,"adaptive convergence");check(std::abs(val.mu-base.mu)<1e-4*std::abs(base.mu),"value reference parity");check(ctx.accepted>0,"certified root-pair path exercised");
 std::cout<<"atlas kernels, cover, warm proof update, anchor/value tests passed\n";
}
