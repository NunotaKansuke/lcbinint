#pragma once
#include <boost/multiprecision/cpp_int.hpp>
namespace lcbinint::algebraic_boundary::tangent::geometry_experiment {
// Rare fallback: an exact Sturm chain of the polynomial at the original dyadic
// input. Interval uncertainty never becomes a guessed sign. Primitive integer
// pseudo-remainders avoid the repeated rational gcd cost of Euclidean division.
namespace exact_detail {
using Z=boost::multiprecision::cpp_int;
using R=boost::multiprecision::number<boost::multiprecision::cpp_rational::backend_type,boost::multiprecision::et_off>;
using P=std::vector<Z>;
inline Z absolute(Z x){return x<0?-x:x;}
inline Z gcd(Z a,Z b){a=absolute(a);b=absolute(b);while(b!=0){Z r=a%b;a=std::move(b);b=std::move(r);}return a;}
inline void primitive(P&p){while(p.size()>1&&p.back()==0)p.pop_back();Z g=0;for(const auto&c:p)g=gcd(g,c);if(g>1)for(auto&c:p)c/=g;}
inline P derivative(const P&p){if(p.size()<2)return {0};P d(p.size()-1);for(size_t i=1;i<p.size();++i)d[i-1]=p[i]*i;primitive(d);return d;}
inline int sign_at(P p,const R&x,int side){
 const Z numerator=boost::multiprecision::numerator(x),denominator=boost::multiprecision::denominator(x);
 int order=0;
 for(;;){
  Z value=p.back(),power=denominator;
  for(int i=int(p.size())-2;i>=0;--i){value=value*numerator+p[i]*power;power*=denominator;}
  if(value!=0){int sign=value>0?1:-1;return side<0&&(order&1)?-sign:sign;}
  if(p.size()==1)return 0;p=derivative(p);++order;
 }
}
struct Chain {
 std::vector<P> p;
 explicit Chain(const PrimaryFrame&pf){
  auto f=positive_detail::derivative(positive_detail::polynomial<R>(pf));
  Z denominator=1;for(int i=0;i<=f.degree;++i){Z d=boost::multiprecision::denominator(f.c[i]);if(d>denominator)denominator=d;}
  P first(f.degree+1);for(int i=0;i<=f.degree;++i){Z numerator=boost::multiprecision::numerator(f.c[i]),d=boost::multiprecision::denominator(f.c[i]);first[i]=numerator*(denominator/d);}primitive(first);
  p.push_back(first);p.push_back(derivative(first));
  while(p.back().size()>1){P r=p[p.size()-2];const auto&b=p.back();const int sign=b.back()>0?1:-1;Z pivot=absolute(b.back());
   while(r.size()>=b.size()&&!(r.size()==1&&r[0]==0)){size_t n=r.size()-1,k=r.size()-b.size();Z lead=r.back();for(size_t i=0;i<n;++i){r[i]*=pivot;if(i>=k)r[i]-=sign*lead*b[i-k];}r.pop_back();if(r.empty())r.push_back(0);primitive(r);}
   if(r.size()==1&&r[0]==0)break;for(auto&c:r)c=-c;p.push_back(std::move(r));
  }
 }
 int variation(const R&x,int side)const{int v=0,last=0;for(const auto&f:p){int s=sign_at(f,x,side);if(s){if(last&&last!=s)++v;last=s;}}return v;}
};
}
inline bool exact_stationary_roots(const PrimaryFrame&pf,double lo,double hi,std::vector<Q>&roots){
 using namespace exact_detail;Chain chain(pf);
 struct Bracket{R lo,hi;int a,b,depth;};R l=R(lo)*R(lo),h=R(hi)*R(hi);
 std::vector<Bracket> work{{l,h,chain.variation(l,1),chain.variation(h,-1),0}};
 std::vector<Q> found;int visits=0;
 while(!work.empty()&&++visits<=4096){auto b=std::move(work.back());work.pop_back();int n=b.a-b.b;if(n<0)return false;if(!n)continue;
  const R mid=(b.lo+b.hi)/2;long double x=mid.convert_to<long double>();
  if(n==1&&(b.hi-b.lo).convert_to<long double>()<1e-21L*std::max(1.L,std::fabs(x))){found.push_back(Q(x));continue;}
  if(b.depth>=256)return false;
  const int left=chain.variation(mid,-1),right=chain.variation(mid,1);
  if(left<right||left>b.a||right<b.b)return false;
  if(left>right){if(left-right!=1)return false;found.push_back(Q(x));}
  work.push_back({mid,b.hi,right,b.b,b.depth+1});work.push_back({b.lo,mid,b.a,left,b.depth+1});
 }
 if(!work.empty())return false;roots=std::move(found);return true;
}
}
