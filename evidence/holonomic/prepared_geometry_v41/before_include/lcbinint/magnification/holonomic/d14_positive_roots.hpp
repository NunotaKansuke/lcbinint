#pragma once
#include <chrono>
#include "d14_real_sturm.hpp"
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::holonomic {
enum class PositiveRootAssurance { Incomplete, LegacyValidated, PositiveRealCertified };
struct D14RealBracket {
 D14Real lo{},hi{},estimate{};
 // qf endpoints retain the outward enclosure when conversion to twofold rounds.
 __float128 v_lo=0,v_hi=0;
 int distinct_count=1; bool simple_verified=false;
};
struct PositiveD14Stats {
 int chain_tier=-1,total_count=0,count_queries=0,sign_queries=0,subdivisions=0;
 int warm_attempts=0,warm_direct_complete=0,repaired_intervals=0,legacy_backend_calls=0;
 double coefficient_ms=0,chain_ms=0,isolation_ms=0,refine_ms=0,variation_ms=0,event_classification_ms=0;
 int largest_root_cluster=0;
 const char* reason="Unattempted";
};
struct PositiveD14Result {
 std::array<D14RealBracket,14> roots{};unsigned root_count=0;
 PositiveRootAssurance assurance=PositiveRootAssurance::Incomplete;
 PositiveD14Stats stats{};int domain_exponent=0;
};
struct PositiveD14Cache { bool valid=false; PositiveD14Result result{}; PrimaryFrame anchor{}; };
namespace positive_detail {
using RootClock=std::chrono::steady_clock;
inline double elapsed(RootClock::time_point t){return std::chrono::duration<double,std::milli>(RootClock::now()-t).count();}
template<class I> Interval<__float128> bounds(I a){
 if constexpr(std::is_same_v<I,Ball>){auto hi=Interval<__float128>(a.c.hi),lo=Interval<__float128>(a.c.lo);
  auto v=hi+lo;return {down(v.lo-(__float128)a.r),up(v.hi+(__float128)a.r)};}
 else return {(__float128)a.lo,(__float128)a.hi};
}
struct StageTimer { double* sum; RootClock::time_point start=RootClock::now();
 void stop(){if(sum){*sum+=elapsed(start);sum=nullptr;}}
 ~StageTimer(){stop();}
};
struct XBracket{__float128 lo=0,hi=1;int n=0,vl=0,vh=0;};
template<class I> bool solve_tier(const PrimaryFrame& pf,int e,PositiveD14Result& out,const PositiveD14Cache* cache){
 auto t=RootClock::now();auto p=polynomial<I>(pf);out.stats.coefficient_ms+=elapsed(t);
 Chain<I> chain;t=RootClock::now();bool ok=chain.build(p,e);out.stats.chain_ms+=elapsed(t);
 if(!ok){out.stats.reason="ChainPivotUncertain";return false;}
 auto variation=[&](__float128 x,int side){++out.stats.count_queries;StageTimer timer{&out.stats.variation_ms};return chain.variation(x,side);};
 int vl=variation(0,1),vh=variation(1,-1);if(vl<0||vh<0||vl<vh){out.stats.reason="DomainSignUncertain";return false;}
 const auto dp=derivative(chain.s[0]);
 int n=vl-vh;out.stats.total_count=n;out.stats.largest_root_cluster=n;if(n>14){out.stats.reason="CountInvalid";return false;}
 std::array<XBracket,14> isolated{};int count=0;
 auto sign=[&](__float128 x){++out.stats.sign_queries;return eval(chain.s[0],point<I>(x)).sign();};
 StageTimer isolation_timer{&out.stats.isolation_ms};
 if(cache&&cache->valid){
  ++out.stats.warm_attempts;
  // Predictor only; sign changes plus current global count certify completeness.
  for(unsigned i=0;i<std::min(14u,cache->result.root_count);++i){
   __float128 x=scalbnq(d14_to_qf(cache->result.roots[i].estimate),-e);
   auto a=bounds(eval(chain.s[0],point<I>(x))),b=bounds(eval(dp,point<I>(x)));
   __float128 dx=0;if(b.lo>0||b.hi<0)dx=d14_to_qf(d14_from_qf((a.lo+a.hi)/2)/d14_from_qf((b.lo+b.hi)/2));
   __float128 pred=x-dx,width=fmaxq(fabsq(dx)*2,scalbnq(fmaxq(fabsq(x),1),-40));
   bool found=false;
   for(int j=0;j<12;++j){__float128 l=fmaxq(0,pred-width),h=fminq(1,pred+width);
    int sl=sign(l),sh=sign(h);
    if(sl*sh==-1&&sl!=2&&sh!=2){isolated[count++]={l,h,1,0,0};found=true;break;}
    width*=2;
   } (void)found;
  }
  std::sort(isolated.begin(),isolated.begin()+count,[](auto a,auto b){return a.lo<b.lo;});
  bool disjoint=true;
  for(int i=1;i<count;++i)if(isolated[i-1].hi>=isolated[i].lo)disjoint=false;
  if(count==n&&disjoint)++out.stats.warm_direct_complete;
 }
 if(!out.stats.warm_direct_complete){
  // Merge overlapping proposals into unions, count them, then isolate only
  // those unions and uncovered gaps. No candidate is assumed count-one.
  std::array<XBracket,256> stack{};int top=0;
  if(count){
   int unions=0;
   for(int i=0;i<count;++i){
    if(unions&&isolated[i].lo<=isolated[unions-1].hi)
     isolated[unions-1].hi=fmaxq(isolated[unions-1].hi,isolated[i].hi);
    else isolated[unions++]=isolated[i];
   }
   __float128 end=0;int vend=vl;
   for(int i=0;i<unions;++i){auto b=isolated[i];
    int l=variation(b.lo,1),h=variation(b.hi,-1);
    if(l<0||h<0||l<h||vend<l){out.stats.reason="RepairSignUncertain";return false;}
    if(b.lo>end)stack[top++]={end,b.lo,vend-l,vend,l};
    stack[top++]={b.lo,b.hi,l-h,l,h};
    ++out.stats.repaired_intervals;end=b.hi;vend=h;
   }
   if(end<1)stack[top++]={end,1,vend-vh,vend,vh};
  }else stack[top++]={0,1,n,vl,vh};
  count=0;
  while(top){auto b=stack[--top];if(b.n<0||b.n>n||count+b.n>14){out.stats.reason="CountInvalid";return false;}if(!b.n)continue;
   if(b.n==1){isolated[count++]=b;continue;}
   if(++out.stats.subdivisions>4096||top>250){out.stats.reason="IsolationBudget";return false;}
   __float128 m=(b.lo+b.hi)/2;if(!(m>b.lo&&m<b.hi)){out.stats.reason="RepresentationLimited";return false;}
   int vm=variation(m,1),sm=sign(m);
   if(vm<0||sm==2){out.stats.reason="SplitSignUncertain";return false;}
   if(sm==0){int left=variation(m,-1);if(left<0){out.stats.reason="AtomSideUncertain";return false;}
    isolated[count++]={m,m,1,0,0};
    stack[top++]={b.lo,m,b.vl-left,b.vl,left};stack[top++]={m,b.hi,vm-b.vh,vm,b.vh};
   }else{stack[top++]={b.lo,m,b.vl-vm,b.vl,vm};stack[top++]={m,b.hi,vm-b.vh,vm,b.vh};}
  }
 }
 isolation_timer.stop();
 if(count!=n){out.stats.reason="CountMismatch";return false;}
 std::sort(isolated.begin(),isolated.begin()+count,[](auto a,auto b){return a.lo<b.lo;});
 StageTimer refine_timer{&out.stats.refine_ms};
 for(int i=0;i<count;++i){auto b=isolated[i];bool simple=false;
  for(int iteration=0;iteration<256&&b.lo!=b.hi;++iteration){
   __float128 mid=(b.lo+b.hi)/2;
   if(b.hi-b.lo<=scalbnq(fmaxq(fabsq(mid),FLT128_MIN),-49))break;
   // Interval Newton contraction; derivative range includes all x in I.
   auto pp=chain.s[0];Interval<__float128> dr=bounds(dp.c[dp.degree]);
   Interval<__float128> xi(b.lo,b.hi);
   for(int k=dp.degree-1;k>=0;--k)dr=dr*xi+bounds(dp.c[k]);
   auto pm=bounds(eval(pp,point<I>(mid)));
   if(dr.lo>0||dr.hi<0){simple=true;
    Interval<__float128> inv(down(1/dr.hi),up(1/dr.lo));auto step=pm*inv;
    __float128 l=fmaxq(b.lo,down(mid-step.hi)),h=fminq(b.hi,up(mid-step.lo));
    if(l>h){out.stats.reason="NewtonEmpty";return false;}
    if(h-l<(b.hi-b.lo)*0.75Q){b.lo=l;b.hi=h;continue;}
   }
   int sl=sign(b.lo),sm=sign(mid),sh=sign(b.hi);
   if(sm==0){b.lo=b.hi=mid;break;}
   if(sm!=2&&sl!=2&&sl*sm==-1)b.hi=mid;
   else if(sm!=2&&sh!=2&&sm*sh==-1)b.lo=mid;
   else {int a=variation(b.lo,1),m=variation(mid,1),h=variation(b.hi,-1);
    if(a<0||m<0||h<0){out.stats.reason="RefineSignUncertain";return false;}
    if(a-m==1)b.hi=mid;else if(m-h==1)b.lo=mid;else {out.stats.reason="RefineCountMismatch";return false;}
   }
  }
  __float128 mid=(b.lo+b.hi)/2;
  if(b.hi-b.lo>scalbnq(fmaxq(fabsq(mid),FLT128_MIN),-49)){out.stats.reason="RefineBudget";return false;}
  auto& r=out.roots[i];r.v_lo=scalbnq(b.lo,e);r.v_hi=scalbnq(b.hi,e);
  r.lo=d14_from_qf(r.v_lo);r.hi=d14_from_qf(r.v_hi);r.estimate=d14_from_qf(scalbnq(mid,e));r.simple_verified=simple;
 }
 refine_timer.stop();out.root_count=count;out.assurance=PositiveRootAssurance::PositiveRealCertified;out.stats.reason="Certified";return true;
}
} // detail
inline PositiveD14Result positive_d14_roots(const PrimaryFrame& pf,double rmax,const PositiveD14Cache* cache=nullptr){
 using namespace positive_detail;PositiveD14Result out;
 if(!environment_ok()||!std::isfinite(rmax)||rmax<=0){out.stats.reason="ArithmeticEnvironment";return out;}
 if(pf.Y==0||pf.a==0||pf.m0==1){out.stats.reason="MultiplicityUnresolved";return out;}
 int e;frexpq((__float128)rmax*rmax,&e);out.domain_exponent=e;
 out.stats.chain_tier=0;if(solve_tier<Interval<double>>(pf,e,out,cache))return out;
 out.stats.chain_tier=1;if(solve_tier<Ball>(pf,e,out,cache))return out;
 out.stats.chain_tier=2;if(solve_tier<Interval<__float128>>(pf,e,out,cache))return out;
 out.root_count=0;return out;
}
} // namespace

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
