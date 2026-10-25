#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
std::array<double,8> pack(const Geometry& p){return {p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};}
int main(){std::array<Complex,3> lenses{Complex(.3,-.2),Complex(.8,.9),Complex(-.7,.6)};
 Complex source(.2,.4);double rho=.05,m1=.2,m2=.08;
 auto frame=primary_frame<true>(lenses,source,rho,m1,m2);
 for(int j=0;j<11;++j)for(double h:{1e-5,3e-6}){
  auto lp=lenses,lm=lenses;auto sp=source,sm=source;double rp=rho,rm=rho,a=m1,b=m1,c=m2,d=m2;
  if(j==0){sp+=h;sm-=h;}else if(j==1){sp+=Complex(0,h);sm-=Complex(0,h);}
  else if(j==2){rp+=h;rm-=h;}else if(j==3){a+=h;b-=h;}else if(j==4){c+=h;d-=h;}
  else {int lens=(j-5)/2;Complex delta=(j-5)%2?Complex(0,h):Complex(h,0);lp[lens]+=delta;lm[lens]-=delta;}
  auto p=pack(primary_frame(lp,sp,rp,a,c).geometry),m=pack(primary_frame(lm,sm,rm,b,d).geometry);
  for(int i=0;i<8;++i){double fd=(p[i]-m[i])/(2*h);if(std::abs(fd-frame.jacobian[i][j])>1e-8*(1+std::abs(fd)))throw std::runtime_error("frame Jacobian");}
 }
 // Original and canonical lens mappings transform covariantly.
 Complex z(.6,.3),f=z;auto masses=frame.geometry.masses();for(int j=0;j<3;++j)f-=masses[j]/std::conj(z-lenses[j]);
 Complex rotation=std::conj(lenses[1]-lenses[0])/std::abs(lenses[1]-lenses[0]);
 Complex zc=rotation*(z-lenses[0]);double phi=evaluate(std::abs(zc),std::arg(zc),frame.geometry).phi;
 if(std::abs(phi-(1-std::norm(f-source)/(rho*rho)))>1e-10)throw std::runtime_error("frame map identity");
 std::cout<<"PASS: physical frame map and 8x11 Jacobian, two step sizes\n";
}
