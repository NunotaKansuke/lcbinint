#include "lcbinint/magnification/holonomic/quartic_fast_count.hpp"
#include "quartic_reference.hpp"
#include <random>
#include <iostream>
#include <chrono>
int main(){std::mt19937_64 rng(2601);std::uniform_real_distribution<double>d(-1,1);int bad=0,fast=0,unresolved=0;std::vector<std::array<double,5>> ps;
for(int i=0;i<200000;++i){std::array<double,5> p;double scale=std::pow(10.,int(rng()%601)-300);for(auto& x:p)x=d(rng)*scale;if(i%10==0)p[rng()%5]=0.;if(i%20==0)p={scale,0.,-2*scale,0.,scale};ps.push_back(p);auto direct=lcbinint::holonomic::invariant_quartic_count(p); auto ref=lcbinint::holonomic_baseline::certify_quartic(p);if(direct>=0 && ref.certified && direct!=ref.root_count)++bad; if(direct>=0)++fast; if(!ref.certified)++unresolved;}
std::cout<<"conditions "<<ps.size()<<" mismatches "<<bad<<" fast "<<fast<<" ref_unresolved "<<unresolved<<'\n';volatile long sum=0;
for(int order=0;order<2;++order)for(int k=0;k<2;++k){bool newer=(order+k)%2;auto start=std::chrono::steady_clock::now();for(int rep=0;rep<3;++rep)for(auto p:ps){if(newer)sum+=lcbinint::holonomic::boundary_quartic_count(p);else sum+=lcbinint::holonomic_baseline::certify_quartic(p).root_count;}std::cout<<(newer?"after ":"before ")<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';}return bad!=0;}
