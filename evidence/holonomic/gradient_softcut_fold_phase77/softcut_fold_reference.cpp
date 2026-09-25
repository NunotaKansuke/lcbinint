#include "adaptive_reference.hpp"
#include <iomanip>
#include <iostream>
using namespace lcbinint::holonomic;
static void move_param(LensParams& p,int j,double h){if(j==0)p.xs+=h;else p.ys+=h;}
int main(){
  LensParams p; p.xs=-0.07045642231021985; p.ys=-0.14887100926364968;
  p.rho=0.07530925479707781; p.q=2.1433640368724767e-8;
  p.a=3.035303804422572; p.barycentric=false;
  std::cout<<std::setprecision(17)
    <<"u\taxis\tstep_fraction\tn\th\tplus_status\tminus_status\tplus\tminus\tfd\n";
  for(double u:{0.0,.5})for(int j:{0,1})
    for(double frac:{1e-3,3e-4,1e-4})for(int n:{128,256}){
      const double h=frac*p.rho;LensParams plus=p,minus=p;
      move_param(plus,j,h);move_param(minus,j,-h);
      auto tp=classify_cells(PrimaryFrame::from(plus));
      auto tm=classify_cells(PrimaryFrame::from(minus));
      const double vp=reference(plus,u,tp,n),vm=reference(minus,u,tm,n);
      const double fd=(vp-vm)/(2*h);
      std::cout<<u<<'\t'<<(j==0?"X":"Y")<<'\t'<<frac<<'\t'<<n<<'\t'<<h<<'\t'
        <<(tp.status==Status::OK)<<'\t'<<(tm.status==Status::OK)<<'\t'
        <<vp<<'\t'<<vm<<'\t'<<fd<<'\n';
    }
}
