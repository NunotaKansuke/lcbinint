#include "quartic_binary_trial.hpp"

#include <random>
#include <iostream>
#include <chrono>
int main(){std::mt19937_64 rng(2601);std::uniform_real_distribution<double>d(-1,1);int bad=0;std::vector<std::array<double,5>> ps;
for(int i=0;i<200000;++i){std::array<double,5> p;double scale=std::pow(10.,int(rng()%601)-300);for(auto& x:p)x=d(rng)*scale;if(i%10==0)p[rng()%5]=0.;if(i%20==0)p={scale,0.,-2*scale,0.,scale};ps.push_back(p);auto a=lcbinint::holonomic::certify_quartic<true>(p);auto b=lcbinint::holonomic::certify_quartic(p);if(a.certified!=b.certified||a.root_count!=b.root_count||a.precision_tier!=b.precision_tier||a.reciprocal!=b.reciprocal||a.chart_singular!=b.chart_singular||a.chart_coeffs!=b.chart_coeffs)++bad;}
std::cout<<"conditions "<<ps.size()<<" mismatches "<<bad<<'\n';volatile long sum=0;
for(int order=0;order<2;++order)for(int k=0;k<2;++k){bool newer=(order+k)%2;auto start=std::chrono::steady_clock::now();for(int rep=0;rep<3;++rep)for(auto p:ps){if(newer)sum+=lcbinint::holonomic::certify_quartic<true>(p).root_count;else sum+=lcbinint::holonomic::certify_quartic(p).root_count;}std::cout<<(newer?"after ":"before ")<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';}return bad!=0;}
