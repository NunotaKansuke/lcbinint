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
inline Z gcd(Z a,Z b){return boost::multiprecision::gcd(absolute(a),absolute(b));}
inline void primitive(P&p){while(p.size()>1&&p.back()==0)p.pop_back();Z g=0;for(const auto&c:p)g=gcd(g,c);if(g>1)for(auto&c:p)c/=g;}
inline P derivative(const P&p){if(p.size()<2)return {0};P d(p.size()-1);for(size_t i=1;i<p.size();++i)d[i-1]=p[i]*i;primitive(d);return d;}
// Most signs are nonzero. Read the primitive polynomial in place; construct
// a derivative only for the exact one-sided-zero case.
inline int value_sign_at(const P& p,const Z& numerator,const Z& denominator){
 Z value=p.back(),power=denominator;
 for(int i=int(p.size())-2;i>=0;--i){value=value*numerator+p[i]*power;power*=denominator;}
 return value==0?0:value>0?1:-1;
}
inline int sign_at(const P& p,const Z& numerator,const Z& denominator,int side){
 const int value=value_sign_at(p,numerator,denominator);
 if(value)return value;
 if(p.size()==1)return 0;
 const int sign=sign_at(derivative(p),numerator,denominator,side);
 return side<0?-sign:sign;
}
struct Chain {
 std::vector<P> p;
 explicit Chain(const PrimaryFrame&pf){
  auto f=positive_detail::derivative(positive_detail::polynomial<R>(pf));
  Z denominator=1;for(int i=0;i<=f.degree;++i){Z d=boost::multiprecision::denominator(f.c[i]);if(d>denominator)denominator=d;}
  P first(f.degree+1);for(int i=0;i<=f.degree;++i){Z numerator=boost::multiprecision::numerator(f.c[i]),d=boost::multiprecision::denominator(f.c[i]);first[i]=numerator*(denominator/d);}primitive(first);
  p.push_back(first);p.push_back(derivative(first));
  while(p.back().size()>1){P r=p[p.size()-2];const auto&b=p.back();const int sign=b.back()>0?1:-1;Z pivot=absolute(b.back());
   while(r.size()>=b.size()&&!(r.size()==1&&r[0]==0)){size_t n=r.size()-1,k=r.size()-b.size();Z lead=r.back();for(size_t i=0;i<n;++i){r[i]*=pivot;if(i>=k)r[i]-=sign*lead*b[i-k];}r.pop_back();if(r.empty())r.push_back(0);while(r.size()>1&&r.back()==0)r.pop_back();}
   // Every pseudo-division pivot is positive. Intermediate content removal
   // only rescales the same remainder by a positive integer, so normalize
   // once after the full remainder, preserving every signed Sturm variation.
   primitive(r);
   if(r.size()==1&&r[0]==0)break;for(auto&c:r)c=-c;p.push_back(std::move(r));
  }
 }
 int variation(const R&x,int side)const{const Z numerator=boost::multiprecision::numerator(x),denominator=boost::multiprecision::denominator(x);int v=0,last=0;for(const auto&f:p){int s=sign_at(f,numerator,denominator,side);if(s){if(last&&last!=s)++v;last=s;}}return v;}
};
}
inline bool exact_stationary_roots(const exact_detail::Chain& chain,double lo,double hi,std::vector<Q>&roots){
 using namespace exact_detail;
 struct Bracket{R lo,hi;int a,b,depth;};R l=R(lo)*R(lo),h=R(hi)*R(hi);
 std::vector<Bracket> work{{l,h,chain.variation(l,1),chain.variation(h,-1),0}};
 std::vector<Q> found;int visits=0;
 while(!work.empty()&&++visits<=4096){auto b=std::move(work.back());work.pop_back();int n=b.a-b.b;if(n<0)return false;if(!n)continue;
  // Sturm has already proved uniqueness on this bracket. Opposite nonzero
  // endpoint signs also prove an odd root, so exact polynomial signs locate
  // that same root without evaluating the remaining Sturm polynomials.
  // Even roots and endpoint zeros retain the general variation path below.
  if(n==1){
   const auto sign=[&](const R&x){return value_sign_at(chain.p.front(),
       boost::multiprecision::numerator(x),boost::multiprecision::denominator(x));};
   int left=sign(b.lo),right=sign(b.hi);
   if(left&&right&&left!=right){
    bool done=false;
    while(b.depth<=256){
     const R mid=(b.lo+b.hi)/2;const long double x=mid.convert_to<long double>();
     if((b.hi-b.lo).convert_to<long double>()<1e-21L*std::max(1.L,std::fabs(x))){found.push_back(Q(x));done=true;break;}
     if(b.depth>=256)break;
     const int middle=sign(mid);
     if(!middle){found.push_back(Q(x));done=true;break;}
     if(middle==left)b.lo=mid;else b.hi=mid;
     ++b.depth;
    }
    if(!done)return false;
    continue;
   }
  }
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
