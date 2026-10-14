#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <iostream>
using namespace lcbinint::holonomic;
int main(){int checks=0,failures=0;
auto check=[&](bool ok){++checks;if(!ok)++failures;};
for(int lev=2;lev<=8;++lev){const auto& r=linear_radial_rule(lev);int n=r.x.size();
 for(int d=0;d<n;++d){long double q=0;for(int k=0;k<n;++k)q+=r.w[k]*powl(r.x[k],d);
 long double exact=d%2?0:4.L/((d+1.L)*(d+3.L));check(fabsl(q-exact)<3e-13);}
 for(double w:r.w)check(w>0);
}
for(int d=0;d<=12;++d){AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=p.split_ld=true;w.panels.push_back(p);
 AdaptiveConfig c;c.weighted_radial=c.split_ld_radial=true;c.tol.mu_atol=1e-10;c.tol.mu_rtol=1e-8;
 auto result=adaptive_detail::integrate(w,c,[&](double R,double J,int,const AdaptiveSample*){AdaptiveSample s;
 double a=sqrt(R*(1-R))*pow(R,d)*J,b=R*(1-R)*pow(R,d)*J;
 s.uniform_radial_value=a;s.value[0]=a+b;s.ld_radial_value=b;return s;});
 double exact=exp(lgamma(d+1.5)+lgamma(1.5)-lgamma(d+3.))+1./((d+2.)*(d+3.));
 check(result.value_converged);check(fabs(result.mu-exact)<=c.tol.budget(0,exact));
}
// Opposite channel remainders must not cancel out of the error ledger.
{AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=p.split_ld=true;p.level=3;
 for(int k=1;k<8;++k){double x=fejer_node(3,k),z=sqrt(1-x*x);AdaptiveSample a;
 a.uniform_radial_value=z*exp(6*x);a.ld_radial_value=-exp(6*x);a.value[0]=0;
 a.R=(1+x)/2;p.samples[k*32]=w.samples.size();w.samples.push_back(a);}
 AdaptiveConfig c;c.weighted_radial=c.split_ld_radial=true;
 adaptive_detail::estimate<1>(p,w,1,c);
 check(p.radial[0]>1e-3);check(!p.value_resolved);
 // The angular/geometry ledger dominates both weighted measures.
 for(auto& a:w.samples){a.inner[0]=1;a.geometry[0]=2;}
 adaptive_detail::estimate<1>(p,w,1,c);
 long double envelope=0;const auto& u=radial_rule(3,true);const auto& l=linear_radial_rule(3);
 for(int k=0;k<7;++k)envelope+=std::max(u.w[k],l.w[k]/sqrt(1-u.x[k]*u.x[k]));
 check(fabs(p.inner[0]-envelope)<1e-13);check(fabs(p.geometry[0]-2*envelope)<1e-13);
}
std::cout<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;}
