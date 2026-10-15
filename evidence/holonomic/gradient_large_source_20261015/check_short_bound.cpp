#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "fused_k.hpp"
#include <random>
#include <iostream>
using namespace lcbinint::holonomic_fused;
int main(){std::mt19937_64 rng(20916);int finite=0,rejected=0;double worst=0;
 auto uniform=[&](){return double(rng()%1000000)/1000000;};
 const auto& hi=gc2_rule_fixed<256>();const auto& lo=gc2_rule_fixed<3>();
 for(int k=0;k<10000;++k){double m=8*(uniform()-.5),s=std::pow(10.,-4+4*uniform()),v=s*s,b0=std::pow(10.,-3+4*uniform()),b2=std::pow(10.,-2+4*uniform()),rho=std::pow(10.,-2+3*uniform());DeflatedQuad q{-std::pow(10.,-2+4*uniform()),.1*(uniform()-.5),-std::pow(10.,-2+4*uniform())};
  double bound=fused_gradient::short_angular_value_bound(q,m,s,v,b0,b2,rho);if(!std::isfinite(bound)){++rejected;continue;}++finite;
  auto sum=[&](const auto&rule){long double out=0;for(size_t j=0;j<rule.x.size();++j){long double t=(long double)m+(long double)s*rule.x[j],S=-((long double)q.d0+t*((long double)q.d1+t*q.p4)),A=1+t*t,B=(long double)b0*b0+(long double)b2*t*t;if(!(S>0&&B>0)){std::cerr<<"nonpositive certified factor\n";std::exit(2);}out+=(long double)rule.w[j]*sqrtl(S)/(A*sqrtl(A)*sqrtl(B));}return out*(2*(long double)v/rho);};
  long double a=sum(lo),b=sum(hi),error=fabsl(a-b),roundoff=128*std::numeric_limits<double>::epsilon()*std::max(fabsl(a),fabsl(b));if(error>(long double)bound+roundoff){std::cerr<<"bound failed "<<k<<" "<<bound<<" "<<(double)error<<"\n";return 1;}
 }
 auto pf=PrimaryFrame::from({-.04211916871535185,-.023021201678620647,.5,1000.,.6,true});double R=1.6095516093978897,m=-.014217861161051171,v=.18079566508308553;auto q=deflate_mv(boundary_quartic(R,pf).p,m,v);double bad=fused_gradient::short_angular_value_bound(q,m,std::sqrt(v),v,R-pf.a,(R+pf.a)*(R+pf.a),pf.rho);
 if(std::isfinite(bad)&&bad<1e-4){std::cerr<<"alias case not excluded\n";return 3;}
 std::cout<<finite<<" finite bounds passed; "<<rejected<<" inconclusive; alias shortcut excluded\n";return finite>1000?0:4;}
