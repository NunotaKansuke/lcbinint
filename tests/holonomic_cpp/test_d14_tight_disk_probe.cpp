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
 for(int k=0;k<2;++k){D14EventContractCapture cap;auto pf=PrimaryFrame::from(ps[k]);
  {D14EventContractCaptureScope scope(cap);(void)classify_cells(pf,nullptr,nullptr,true);}
  bool found=false;
  for(const auto& c:cap.candidates)if(c.stage=="qf_warm"){
   found=true;auto cert=tight_certificate(pf,c.roots);std::cout<<k<<" qf_warm "<<cert.certified<<' '<<cert.isolated_disks<<'\n';
   if(cert.certified!=(k==0))++failures;
  }
  if(!found)++failures;
  auto duplicate=cap.oracle_roots;duplicate[0]=duplicate[1];
  if(tight_certificate(pf,duplicate).certified)++failures;
 }
 std::cout<<"failures "<<failures<<'\n';return failures?1:0;
}
