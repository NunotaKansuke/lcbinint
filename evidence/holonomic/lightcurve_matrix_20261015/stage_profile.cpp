#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){if(argc!=3)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);out<<std::setprecision(17)<<"case\tindex\tt\tu\ttol\texpected\tmu\tconverged\tnodes\tseeds\tseed_ms\tkernel_ms\tplanning_ms\tintegration_ms\tfrontier_event_ms\tfrontier_count_ms\tendpoint_ms\trootpair_ms\tk_ms\tangular_rescue_ms\testimator_ms\tallocation_ms\tlocal_repairs\tintegration_calls\testimator_prediction_ms\tnode_loop_ms\theader_ms\tcoeff_ms\tarc_body_ms\ttail_ms\tscheduler_ms\tquartic_cold_calls\tquartic_warm_calls\n";
std::string name,previous;int index;double t,u,tol,expected;LensParams p;std::unique_ptr<lcbinint::magnification::FiniteSourceMagnifier> finite;lcbinint::magnification::PointSourceMagnifier point;
while(in>>name>>index>>t>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>u>>tol>>expected){if(name!=previous){lcbinint::magnification::FiniteSourceSettings settings;settings.caustic_bins=1400;finite=std::make_unique<lcbinint::magnification::FiniteSourceMagnifier>(settings);finite->binary_caustic_distance_for_source(p.a,p.q,{p.xs,p.ys});previous=name;}
auto elapsed=[](auto start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};auto start=Clock::now();auto seeds=make_caustic_aware_binary_seeds(*finite,point,p.a,p.q,{p.xs,p.ys},p.rho);double seedms=elapsed(start);AdaptiveWorkspace w;V2Profile prof;prof.collect_d14_posthoc=false;SeededFrontierResult r;start=Clock::now();{V2ProfileScope scope(prof);r=epoch_seeded_radial_frontier_value(p,u,seeds.radial_seeds,tol,tol,w);}double ms=elapsed(start);
out<<name<<'\t'<<index<<'\t'<<t<<'\t'<<u<<'\t'<<tol<<'\t'<<expected<<'\t'<<r.integral.mu<<'\t'<<r.integral.value_converged<<'\t'<<r.integral.stats.node_evaluations<<'\t'<<seeds.radial_seeds.size()<<'\t'<<seedms<<'\t'<<ms<<'\t'<<r.frontier.planning_ms<<'\t'<<r.frontier.integration_ms<<'\t'<<prof.frontier_event_ms<<'\t'<<prof.frontier_count_ms<<'\t'<<prof.endpoint_ms<<'\t'<<prof.rootpair_ms<<'\t'<<prof.k_ms<<'\t'<<prof.angular_rescue_ms<<'\t'<<r.frontier.estimator_ms<<'\t'<<r.frontier.allocation_ms<<'\t'<<r.frontier.local_repairs<<'\t'<<r.frontier.integration_calls<<'\t'<<prof.estimator_prediction_ms<<'\t'<<prof.adaptive_node_loop_ms<<'\t'<<prof.mapped_header_ms<<'\t'<<prof.mapped_coeff_ms<<'\t'<<prof.mapped_arc_body_ms<<'\t'<<prof.mapped_tail_ms<<'\t'<<r.frontier.scheduler_ms<<'\t'<<prof.quartic_cold_calls<<'\t'<<prof.quartic_warm_calls<<'\n';}
}
