#pragma once
#include <memory>
#include <mutex>
#include "gradient_quadrature.hpp"
#include "../nested_fejer2.hpp"
namespace lcbinint::algebraic_boundary::triple::fused_detail {
// Confluent rule on the SAME nested nodes: each physical sample supplies a
// value and a spatial slope. Values and parameter tangents use identical
// weights. Mode 0/1/2 denotes 1 / sqrt(1-x*x) / (1-x*x) integration weight.
struct RadialRule {
 std::vector<double> x,value,derivative;
 struct Witness {double x;std::vector<double> value,derivative;};
 std::vector<Witness> witnesses;
 std::vector<std::vector<double>> interpolation,interpolation_derivative;
 RadialRule(int level,int mode) {
  x=fejer_rule(level).x;const int n=int(x.size());value.resize(n);derivative.resize(n);
  const shape_gradient_detail::GaussRule integration(std::max(8,2*n+2));
  std::vector<long double> sums(n),slopes(n),diagonal(n);
  for(int j=0;j<n;++j)for(int k=0;k<n;++k)if(j!=k)diagonal[j]+=1.L/(x[j]-x[k]);
  const auto basis=[&](long double t,int j) {
   long double l=1;for(int k=0;k<n;++k)if(k!=j)l*=(t-x[k])/(x[j]-x[k]);
   const long double d=t-x[j];return std::array<long double,2>{(1-2*d*diagonal[j])*l*l,d*l*l};
  };
  // sqrt-weight moments use exact Chebyshev-II quadrature; the other
  // weights are polynomials, integrated exactly by the Gaussian rule.
  if(mode==1) {
   const int m=2*n+3;const long double pi=acosl(-1.L);
   for(int k=1;k<m;++k){const long double a=pi*k/m,t=cosl(a),w=pi/m*sinl(a)*sinl(a);for(int j=0;j<n;++j){const auto b=basis(t,j);sums[j]+=w*b[0];slopes[j]+=w*b[1];}}
  } else for(std::size_t k=0;k<integration.x.size();++k) {
   const long double t=integration.x[k],w=integration.w[k]*(mode==2?(1-t)*(1+t):1);
   for(int j=0;j<n;++j){const auto b=basis(t,j);sums[j]+=w*b[0];slopes[j]+=w*b[1];}
  }
  for(int j=0;j<n;++j){value[j]=double((sums[j]+sums[n-1-j])*.5L);derivative[j]=double((slopes[j]-slopes[n-1-j])*.5L);}
  for(double t:shape_gradient_detail::gauss_rule(8).x){Witness w;w.x=t;w.value.resize(n);w.derivative.resize(n);for(int j=0;j<n;++j){auto b=basis(t,j);w.value[j]=double(b[0]);w.derivative[j]=double(b[1]);}witnesses.push_back(std::move(w));}
  // Interpolation uses the previous nested rule's values AND spatial slopes.
  if(level>1) {
   const auto& coarse=fejer_rule(level-1).x;const int nc=int(coarse.size());
   interpolation.resize(n);interpolation_derivative.resize(n);
   for(int k=0;k<n;++k)if(k%2==0) {
    interpolation[k].resize(nc);interpolation_derivative[k].resize(nc);
    for(int j=0;j<nc;++j) {
     long double l=1,d=0;for(int a=0;a<nc;++a)if(a!=j){l*=((long double)x[k]-coarse[a])/(coarse[j]-coarse[a]);d+=1.L/(coarse[j]-coarse[a]);}
     const long double delta=(long double)x[k]-coarse[j];interpolation[k][j]=double((1-2*delta*d)*l*l);interpolation_derivative[k][j]=double(delta*l*l);
    }
   }
  }
 }
};
inline const RadialRule& radial_rule(int level,int mode) {
 if(level<1||level>8||mode<0||mode>2)throw std::invalid_argument("fused confluent rule");
 // Named static slots avoid anonymous-static aliasing across diagnostic DSOs.
 static std::array<std::array<std::unique_ptr<RadialRule>,3>,8> rules;
 // Production calls may use independent threads. Initialization is guarded;
 // references never move after construction.
 static std::mutex mutex;std::lock_guard<std::mutex> guard(mutex);
 auto& rule=rules[level-1][mode];if(!rule)rule=std::make_unique<RadialRule>(level,mode);return *rule;
}
}
