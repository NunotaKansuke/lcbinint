#include "lcbinint/magnification/algebraic_boundary/triple/annular_area_bound.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 Geometry p{1,0,.001,0,0,1,2,.3};
 for(double center:{(std::sqrt(5.)+1)/2,(std::sqrt(5.)-1)/2}) {
  const double lo=center-.0002,hi=center+.0002,fraction=annular_support_fraction(lo,hi,p);
  if(!(fraction>0 && fraction<.01))throw std::runtime_error("nontrivial annulus bound");
  for(int j=0;j<=10000;++j) {
   const long double r=lo+(hi-lo)*static_cast<long double>(j)/10000;
   const long double a=std::abs(r-1/r),c=(a*a+1-static_cast<long double>(p.rho)*p.rho)/(2*a);
   const long double exact=c>=1?0:c<=-1?1:std::acos(c)/std::acos(-1.L);
   if(exact>fraction)throw std::runtime_error("single lens angular coverage");
  }
 }
 if(annular_support_fraction(-.1,.1,p)!=1)throw std::runtime_error("pole fallback");
 p.rho=10;if(annular_support_fraction(1.5,1.6,p)!=1)throw std::runtime_error("full coverage");
 p.rho=.001;if(annular_support_fraction(3.,3.00001,p)!=0)throw std::runtime_error("empty coverage");
 // A rounded m1+m2==1 can conceal a slightly negative exact complement.
 // The absolute-mass Lipschitz bound remains conservative and finite.
 p.m1=std::nextafter(.5,1.);p.m2=.5;p.a=1;p.bx=2;p.by=0;p.rho=2e-10;
 const long double R=1e-10L,m0=1.L-p.m1-p.m2;
 const long double center=R-m0/R-p.m1/(R-1)-p.m2/(R-2);
 p.X=double(center);p.Y=0;p.validate();
 // On this fixed circle, the image of an angular arc of half-width h
 // stays inside the source by the angular Lipschitz inequality. Thus h/pi
 // is an independent analytic lower bound on its angular coverage.
 const long double L=(1+std::abs(m0)/(R*R)+p.m1/((1-R)*(1-R))+
                      p.m2/((2-R)*(2-R)))*(1+1e-12L);
 const long double h=p.rho/(2*R*L);
 if(std::abs(center-p.X)>=p.rho*.25)throw std::runtime_error("edge reference center");
 const double edge=annular_support_fraction(double(R-1e-14L),double(R+1e-14L),p);
 if(!(edge>=h/std::acos(-1.L) && edge<=1))throw std::runtime_error("rounded mass complement");
 std::cout<<"PASS annular coverage against analytic single lens\n";
}
