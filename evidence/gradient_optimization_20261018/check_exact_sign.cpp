#include "lcbinint/magnification/algebraic_boundary/tangent/geometry.hpp"
#include <cassert>
#include <iostream>
using namespace lcbinint::algebraic_boundary::tangent::geometry_experiment;
using namespace exact_detail;
int independent_sign(P p,const R& x,int side){
 int order=0;
 for(;;){R value=0;for(auto i=p.rbegin();i!=p.rend();++i)value=value*x+R(*i);
  if(value!=0){int sign=value>0?1:-1;return side<0&&(order&1)?-sign:sign;}
  if(p.size()==1)return 0;
  P d(p.size()-1);for(size_t k=1;k<p.size();++k)d[k-1]=p[k]*k;p=std::move(d);++order;
 }
}
int main(){int tested=0;
 for(int root=-3;root<=3;++root)for(int order=0;order<8;++order)for(int leading:{-7,3}){
  P p{leading};for(int j=0;j<order;++j){P next(p.size()+1);for(size_t k=0;k<p.size();++k){next[k]-=root*p[k];next[k+1]+=p[k];}p=std::move(next);}
  for(int i=-20;i<=20;++i)for(int side:{-1,1}){R x=R(i)/4;const Z n=boost::multiprecision::numerator(x),d=boost::multiprecision::denominator(x);
   assert(sign_at(p,n,d,side)==independent_sign(p,x,side));++tested;}
 }
 std::cout<<tested<<" exact rational signs, including repeated roots and both one-sided limits: passed\n";
}
