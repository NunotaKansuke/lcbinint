#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace lcbinint::holonomic;
int main(){std::ifstream f("evidence/holonomic/accuracy_three_v44/input.tsv");std::string l;
while(std::getline(f,l)){if(l.empty()||l[0]=='#')continue;int id,conf,bin,epoch;std::string profile;double s,q,rho,x,y,t,u,X,ref;
std::istringstream in(l);in>>id>>conf>>profile>>bin>>epoch>>s>>q>>rho>>x>>y>>t>>u>>X>>ref;
LensParams p{t,y,rho,1/q,s,true};AdaptiveConfig c;c.gradient_policy=GradientPolicy::None;c.with_jacobian=false;c.tol.mu_rtol=1e-6;c.tol.mu_atol=1e-8;AdaptiveWorkspace w;
auto r=epoch_adaptive(p,u,c,w);std::cout<<std::setprecision(17)<<id<<" "<<profile<<" "<<r.mu<<" "<<r.estimated_abs_error_mu<<" "<<r.value_converged<<" "<<(int)r.stop<<" "<<r.stats.node_evaluations<<std::endl;}}
