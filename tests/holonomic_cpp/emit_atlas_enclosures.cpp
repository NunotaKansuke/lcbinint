// Independent Fraction/SymPy audit consumes these exact hexadecimal bounds.
#include <cstdio>
#include "lcbinint/magnification/holonomic/boundary_bernstein.hpp"
using namespace lcbinint::holonomic;using namespace atlas_detail;
void number(Q x){char b[128];quadmath_snprintf(b,sizeof b,"%+-#.28Qa",x);std::printf(" %s",b);}
template<class I> void emit(const PrimaryFrame& pf,const Box& box,int id,int tier){
 auto p=boundary<I>(pf,box);
 Box actual=box;
 if(id>=100){auto halves=split_physical(p,box,0);p=halves.first;actual.r1=(box.r0+box.r1)/2;}
 for(int i=0;i<=6;++i)for(int j=0;j<=4;++j){auto b=bounds(p.c[i][j]);std::printf("%d %d %d %d %d",id,tier,box.chart,i,j);for(Q x:{(Q)pf.a,(Q)pf.m0,(Q)pf.X,(Q)pf.Y,(Q)pf.rho,actual.r0,actual.r1,actual.s0,actual.s1,b.lo,b.hi})number(x);std::puts("");}
}
int main(){for(int k=0;k<8;++k){PrimaryFrame pf{};pf.a=.25+k*.125;pf.m0=.375;pf.X=-.3+k*.07;pf.Y=k==0?0:.11*k;pf.rho=k==7?1e-7:.0125;
 for(int chart=0;chart<2;++chart){Box b{.125Q,.75Q,-1,1,chart,0};if(k>=4){b.r0=.731Q;b.r1=b.r0+1e-9Q;b.s0=-.312Q;b.s1=b.s0+1e-8Q;}
 emit<Interval<double>>(pf,b,k,0);emit<Ball>(pf,b,k,1);emit<IQ>(pf,b,k,2);emit<Interval<double>>(pf,b,k+100,0);emit<Ball>(pf,b,k+100,1);emit<IQ>(pf,b,k+100,2);}}}
