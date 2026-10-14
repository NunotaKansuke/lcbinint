#include <iostream>
#include <cstdlib>
#include "lcbinint/magnification/holonomic/d14_positive_roots.hpp"
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::positive_detail;
void check(bool x,const char* m){if(!x){std::cerr<<m<<'\n';std::exit(1);}}
template<class I> void count_tests(){
 Poly<I> p; p.degree=2;p.c[0]=I(-1);p.c[2]=I(1);Chain<I> c;
 check(c.build(p),"build quadratic");check(c.variation(0)-c.variation(2,-1)==1,"one positive");
 check(c.variation(-2)-c.variation(2,-1)==2,"two roots");
 // Rounded cancellation is allowed to remain unresolved, never count -1 - -1.
 int l=c.variation(-1,1),h=c.variation(1,-1);
 check(l<0||h<0||l-h==0,"open endpoint accounting");
 Poly<I> z;z.degree=1;z.c[1]=I(1);Chain<I> zero;
 check(zero.build(z),"build exact zero atom");
 check(zero.variation(0,-1)-zero.variation(0,1)==1,"one sided zero atom");
 // Negative pivot and exact multiple root: positive pseudo-remainder convention.
 p.degree=3;p.c={};p.c[0]=I(2);p.c[1]=I(-3);p.c[3]=I(1);
 bool built=c.build(p);
 if(built)check(c.variation(-3)-c.variation(3,-1)==2,"multiple distinct count");
 // Near-real complex pair, exact degree drop, and exact zero polynomial.
 Poly<I> near;near.degree=2;near.c[0]=I(std::ldexp(1.0,-40));near.c[2]=I(1);
 check(c.build(near),"build near-real pair");
 check(c.variation(-1)>=0 && c.variation(1,-1)>=0 &&
       c.variation(-1)-c.variation(1,-1)==0,"near-real not real");
 Poly<I> drop;drop.degree=4;drop.c[1]=I(1);
 check(c.build(drop)&&c.s[0].degree==1,"exact degree drop");
 Poly<I> empty;check(!c.build(empty),"zero polynomial unresolved");
 // Interval arithmetic may reject a gcd; it must not declare uncertain zero.
 Poly<I> no;no.degree=2;no.c[0]=I(1);no.c[2]=I(1);
 check(c.build(no),"build complex pair");check(c.variation(0)-c.variation(3,-1)==0,"no positive roots");
}
int main(){
 check(environment_ok(),"arithmetic environment");
 count_tests<Interval<double>>();count_tests<Ball>();count_tests<Interval<__float128>>();
 Interval<double> a(-1,1);check(a.sign()==2,"uncertain != exact zero");
 auto pf=PrimaryFrame::from(LensParams{0.3,0.2,0.01,2,1,true});
 auto r=positive_d14_roots(pf,3);check(r.assurance==PositiveRootAssurance::PositiveRealCertified,"physical count certified");
 check(r.root_count==6,"physical regression count six");
 for(unsigned i=0;i<r.root_count;++i){check(r.roots[i].v_lo>0&&r.roots[i].v_hi<16,"domain");
  if(i)check(r.roots[i-1].v_hi<r.roots[i].v_lo,"disjoint brackets");}
 PositiveD14Cache cache;cache.valid=true;cache.result=r;
 auto warm=positive_d14_roots(pf,3,&cache);check(warm.root_count==r.root_count,"warm count");
 check(warm.stats.warm_direct_complete==1,"warm current count plus sign brackets");
 cache.result.root_count=0;auto stale=positive_d14_roots(pf,3,&cache);
 check(stale.root_count==6,"empty stale cache cannot hide birth");
 cache.result=r;for(auto& b:cache.result.roots)b.estimate=r.roots[0].estimate;
 auto dup=positive_d14_roots(pf,3,&cache);check(dup.root_count==6,"duplicate predictors repaired");
 auto moved_pf=pf;moved_pf.X+=0.001;
 auto moved=positive_d14_roots(moved_pf,3,&cache);
 auto fresh=positive_d14_roots(moved_pf,3);
 check(moved.assurance==fresh.assurance && moved.root_count==fresh.root_count,"moving cache count");
 check(dup.stats.repaired_intervals>0,"duplicate proposal unions repaired");
 const int rounding=std::fegetround();std::fesetround(FE_UPWARD);
 auto invalid_env=positive_d14_roots(pf,3);std::fesetround(rounding);
 check(invalid_env.assurance==PositiveRootAssurance::Incomplete,"unsupported rounding rejected");
 pf.Y=0;auto axis=positive_d14_roots(pf,3);check(axis.assurance==PositiveRootAssurance::Incomplete,"axis never false certified");
 std::cout<<"positive D14 count/enclosure/warm tests passed\n";
}
