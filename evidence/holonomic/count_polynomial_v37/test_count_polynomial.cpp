#include <iostream>
#include "lcbinint/magnification/holonomic/radial_count_polynomial.hpp"
#include "lcbinint/magnification/holonomic/quartic_fast_count.hpp"
using namespace lcbinint::holonomic;
int main(){int checks=0,fail=0;
 for(double q:{1e-8,1e-4,.1,1.})for(double a:{.6,1.,2.})for(double rho:{1e-5,.1}){
 LensParams lens{.3,-.2,rho,q,a,true};auto pf=PrimaryFrame::from(lens);RadialCountPolynomial poly(pf);
 for(double R:{.01,.2,.5,.9,1.,1.4,2.,4.}){
 auto p=poly.evaluate(R);auto f=local_fold_quantities<__float128>((__float128)R,(__float128)0,pf);
 std::array<double,5> ref{(double)f.P,(double)f.Pt,(double)(f.Ptt/2),(double)(f.Psss/6),(double)(f.Pssss/24)};
 double scale=0;for(double v:ref)scale=std::max(scale,std::fabs(v));
 for(int k=0;k<5;++k){++checks;if(std::fabs(p[k]-ref[k])>1e-13*std::max(1e-300,scale)){++fail;std::cerr<<"coefficient error "<<q<<' '<<a<<' '<<R<<' '<<k<<'\n';}}
 auto cert=certify_quartic(ref);const int count=invariant_quartic_count(p);
 if(count>=0 && cert.certified){++checks;if(count!=cert.root_count){++fail;std::cerr<<"count disagreement\n";}}
 }
 }
 std::cout<<checks<<" checks "<<fail<<" failures\n";return fail?1:0;
}
