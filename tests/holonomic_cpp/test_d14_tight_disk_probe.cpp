// Standalone research certificate regression. Not part of production routing.
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "../../benchmarks/holonomic/d14_tight_disk_probe.hpp"
#include <iostream>
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
int main(){
 const LensParams ps[]={
 {1.1409565850100225,-4.0472265932659397,0.00017266345397718498,1.0/0.56332240832139935,0.2310772011499552,true},
 {-0.012715162713256456,0.30367111943384689,0.64030113599253125,1.0/0.0001084095167119894,0.47185016883470127,true}};
 int failures=0;
 for(int a=-3;a<=3;++a)for(int lo=-4;lo<=4;++lo)for(int hi=lo;hi<=4;++hi){
  auto b=D14RoucheInterval(qf(lo),qf(hi));auto product=point_times_interval(qf(a),b);
  qf exact_lo=std::min(qf(a)*lo,qf(a)*hi),exact_hi=std::max(qf(a)*lo,qf(a)*hi);
  if(!product.valid()||product.lo>exact_lo||product.hi<exact_hi)++failures;
 }

 for(int k=0;k<2;++k){D14EventContractCapture cap;auto pf=PrimaryFrame::from(ps[k]);
  {D14EventContractCaptureScope scope(cap);(void)classify_cells(pf,nullptr,nullptr,true);}
  bool found=false;
  for(const auto& c:cap.candidates)if(c.stage=="qf_warm"){
   found=true;auto cert=tight_certificate(pf,c.roots);std::cout<<k<<" qf_warm "<<cert.certified<<' '<<cert.isolated_disks<<'\n';
   if(cert.certified!=(k==0))++failures;
   auto point_cert=tight_certificate(pf,c.roots,false,true);
   if(point_cert.certified!=cert.certified)++failures;
  }
  if(!found)++failures;
  auto duplicate=cap.oracle_roots;duplicate[0]=duplicate[1];
  if(tight_certificate(pf,duplicate).certified)++failures;
  if(tight_certificate(pf,duplicate,false,true).certified)++failures;
  auto poly=positive_detail::polynomial<D14RoucheInterval>(pf);
  for(auto center:cap.oracle_roots){
   auto reference=d14_rouche_shift(poly,center),candidate=point_interval_shift(poly,center);
   for(int j=0;j<15;++j)for(int component=0;component<2;++component){
    auto a=component?reference[j].im:reference[j].re,b=component?candidate[j].im:candidate[j].re;
    if(!b.valid()||b.lo<a.lo||b.hi>a.hi)++failures;
   }
  }
 }
 std::cout<<"failures "<<failures<<'\n';return failures?1:0;
}
