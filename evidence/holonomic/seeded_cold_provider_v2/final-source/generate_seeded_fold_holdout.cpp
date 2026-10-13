// Offline validation inputs only: critical curves are never built at runtime.
#include <fstream>
#include <iomanip>
#include <complex>
#include "lcbinint/magnification/holonomic/point_images.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc<2)return 2;double input_phase=argc>2?std::stod(argv[2]):.47;double depth=argc>3?std::stod(argv[3]):.98;std::ofstream out(argv[1]);out<<std::setprecision(17);int id=1000;
 for(auto geometry:std::array<std::array<double,2>,10>{{{1e-5,.7},{1e-5,1.05},{1e-3,.95},{1e-3,1.3},{.1,.7},{.1,1.2},{1,.9},{1,2},{3,.8},{3,1.4}}}){
  double q=geometry[0],a=geometry[1],m0=1/(1+q),m1=1-m0;
  double phase=input_phase;Cplx<double> e{cos(phase),sin(phase)};
  std::vector<Cplx<double>> pc{{-m0*a*a,0},{2*m0*a,0},e*Cplx<double>{a*a,0}-Cplx<double>{1,0},e*Cplx<double>{-2*a,0},e};
  auto roots=pimg_detail::complex_aberth<double>(pc,300,2e-15);
  for(auto root:roots){
   std::complex<double> z{root.re,root.im};auto k=m0/(z*z)+m1/((z-a)*(z-a));
   if(std::abs(k-std::polar(1.,phase))>1e-10)return 1;
   auto c=z-m0/std::conj(z)-m1/(std::conj(z)-a);
   double rho=.01*sqrt(std::min(m0,m1));
   auto normal=std::polar(1.,.5*M_PI-.5*phase);
   for(double side:{-1.,1.}){
    auto source=c+side*depth*rho*normal;
    out<<id++<<' '<<source.real()-m1*a<<' '<<source.imag()<<' '<<rho<<' '<<q<<' '<<a<<'\n';
   }
  }
 }
 // On-axis cusp/contact and tiny central sources, separately from the folds.
 for(auto p:std::array<std::array<double,5>,8>{{{0,0,1e-4,1e-5,1.},{0,0,.02,.1,1.},{0,0,.02,1.,.7},{.02,0,.03,.001,1.},{.4,0,.03,1.,2.},{-.4,0,.03,1.,2.},{.1,.05,.05,3.,1.1},{-.1,-.05,.05,3.,1.1}}})
  out<<id++<<' '<<p[0]<<' '<<p[1]<<' '<<p[2]<<' '<<p[3]<<' '<<p[4]<<'\n';
}
