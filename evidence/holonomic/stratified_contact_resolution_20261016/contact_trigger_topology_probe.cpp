#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic_fused;
void run(const char*name,double t,double u0,double alpha,double tE,double rho,double q,double s){std::ifstream in(std::string("evidence/holonomic/stratified_contact_resolution_20261016/seeds_")+name+".dat");std::string line;std::getline(in,line);auto n=std::stoull(line.substr(line.find('=')+1));std::vector<Cplx<double>> seeds;for(size_t k=0;k<n;++k){double x,y;in>>x>>y;seeds.push_back({x,y});}double c=cos(alpha),si=sin(alpha),tau=t/tE;LensParams p{tau*c-u0*si,tau*si+u0*c,rho,q,s,true};AdaptiveConfig cfg;cfg.tol.mu_atol=1e-3;cfg.tol.mu_rtol=1e-3;cfg.with_jacobian=true;cfg.fused_value_mesh=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;AdaptiveWorkspace w;auto r=epoch_seeded_radial_frontier(p,.5,seeds,cfg,w);auto pairs=geometry_experiment::known_upper_edge_pairs(PrimaryFrame::from(p),r.topology,7);std::cout<<std::setprecision(17)<<name<<" frontier="<<r.frontier.reason<<" completed="<<r.frontier.discovery_completed<<" topo="<<int(r.topology.status)<<" cells="<<r.topology.cells.size()<<" events="<<r.topology.events.size()<<" edgepairs="<<pairs.size()<<" mu="<<r.integral.mu<<" stop="<<adaptive_stop_name(r.integral.stop)<<" nodes="<<r.integral.stats.node_evaluations<<"\n";for(size_t i=0;i<r.topology.events.size();++i){auto&e=r.topology.events[i];if(e.physically_real&&e.radius>1.8)std::cout<<" ev"<<i<<" "<<e.radius<<" ";}std::cout<<"\n";for(auto&x:pairs)std::cout<<" pair "<<x.left<<","<<x.right<<" scale="<<x.scale<<"\n";}
int main(){run("center",0,9.846481110931334e-7,.5,10,1e-6,1e7,.95);run("negative",-1.7455113096375089e-6,9.846481110931334e-7,.5,10,1e-6,1e7,.95);}
