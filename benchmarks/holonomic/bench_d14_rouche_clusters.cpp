// Classify qf-warm candidates rejected by the outward-rounded D14 Rouche
// certificate.  Research-only; it does not alter solver acceptance.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"

using namespace lcbinint::holonomic;

struct Row { int case_id=0,configuration_id=0,d_bin_index=0,epoch_index=0;
    std::string profile;double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,reference=0; };
static bool read(const std::string& line,Row& r){if(line.empty()||line[0]=='#')return false;
    std::istringstream in(line);return bool(in>>r.case_id>>r.configuration_id>>r.profile
      >>r.d_bin_index>>r.epoch_index>>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.X>>r.reference);}
static bool same(const Row&a,const Row&b){return a.case_id==b.case_id&&
    a.configuration_id==b.configuration_id&&a.profile==b.profile&&a.d_bin_index==b.d_bin_index;}

int main(int argc,char**argv){if(argc<3){std::cerr<<"usage: bench_d14_rouche_clusters INPUT OUTPUT\n";return 2;}
 std::ifstream input(argv[1]);std::ofstream out(argv[2]);if(!input||!out)return 2;
 std::vector<Row> rows;std::string line;while(std::getline(input,line)){Row r;if(read(line,r))rows.push_back(r);}
 std::stable_sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){if(a.case_id!=b.case_id)return a.case_id<b.case_id;
  if(a.configuration_id!=b.configuration_id)return a.configuration_id<b.configuration_id;
  if(a.profile!=b.profile)return a.profile<b.profile;if(a.d_bin_index!=b.d_bin_index)return a.d_bin_index<b.d_bin_index;
  return a.epoch_index<b.epoch_index;});
 out<<std::setprecision(17)<<"case_id configuration_id profile d_bin_index epoch_index lane isolated_disks "
   "component_id component_size unresolved_roots positive_real physical_real real_soft complex_positive other "
   "min_pair_separation max_newton_correction cluster_certificate cluster_radius cluster_ratio crosses_real_axis\n";
 for(bool warm:{false,true}){std::vector<Cplx<__float128>> previous;
  for(size_t k=0;k<rows.size();++k){if(k==0||!same(rows[k-1],rows[k]))previous.clear();const auto&r=rows[k];
   LensParams p{r.time,r.y,r.rho,1.0/r.q,r.s,true};auto pf=PrimaryFrame::from(p);D14EventContractCapture capture;
   std::vector<Cplx<__float128>> current;{D14EventContractCaptureScope scope(capture);
    classify_cells(pf,warm&&!previous.empty()?&previous:nullptr,&current,true);}
   for(const auto& candidate:capture.candidates){if(candidate.stage!="qf_warm"||candidate.converged||!candidate.finite)continue;
    auto certificate=re_detail::d14_rouche_certificate(pf,candidate.roots,true);if(certificate.certified)continue;
    std::array<int,14> parent{};std::iota(parent.begin(),parent.end(),0);
    auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
    auto unite=[&](int a,int b){a=find(a);b=find(b);if(a!=b)parent[b]=a;};
    // Every unresolved singleton is joined to its closest candidate.  This
    // reports the smallest local cluster that blocks one-root isolation.
    for(int i=0;i<14;++i)if(!certificate.isolated[i]){int nearest=-1;__float128 distance=HUGE_VALQ;
      for(int j=0;j<14;++j)if(i!=j){auto d=cabs(certificate.center[i]-certificate.center[j]);
        if(d<distance){distance=d;nearest=j;}}if(nearest>=0)unite(i,nearest);}
    std::map<int,std::vector<int>> groups;for(int i=0;i<14;++i)groups[find(i)].push_back(i);int component=0;
    auto sc=re_detail::d14_struct_build((__float128)pf.a,(__float128)pf.m0,(__float128)pf.X,(__float128)pf.Y,(__float128)pf.rho);
    for(const auto& [_,indices]:groups){int unresolved=0,pos=0,physical=0,realsoft=0,complexpos=0,other=0;
      __float128 minsep=HUGE_VALQ,maxcorr=0;for(int i:indices){unresolved+=!certificate.isolated[i];const auto&z=candidate.roots[i];
       Cplx<__float128> value,derivative;re_detail::d14_struct_eval(sc,z,value,derivative);
       if(cabs(derivative)>0)maxcorr=fmaxq(maxcorr,cabs(value)/cabs(derivative));
       for(int j:indices)if(i<j)minsep=fminq(minsep,cabs(z-candidate.roots[j]));
       const bool real=fabsq(z.im)<=(__float128)1e-8*((__float128)1+fabsq(z.re));
       if(real&&z.re>0){++pos;double R=std::sqrt((double)z.re);auto probe=re_detail::probe_double_root(R,pf);
         if(probe.physically_real)++physical;else ++realsoft;}
       else if(!real&&z.re>0)++complexpos;else ++other;}
      if(indices.size()==1)minsep=0;
      const auto cluster=re_detail::d14_rouche_cluster_certificate(pf,candidate.roots,indices);
      const bool crosses_real=cluster.certified&&
          fabsq(cluster.center.im)<=cluster.radius;
      out<<r.case_id<<' '<<r.configuration_id<<' '<<r.profile<<' '<<r.d_bin_index<<' '<<r.epoch_index<<' '
       <<(warm?"warm":"cold")<<' '<<certificate.isolated_disks<<' '<<component++<<' '<<indices.size()<<' '
       <<unresolved<<' '<<pos<<' '<<physical<<' '<<realsoft<<' '<<complexpos<<' '<<other<<' '
       <<(double)minsep<<' '<<(double)maxcorr<<' '<<cluster.certified<<' '
       <<(double)cluster.radius<<' '<<(double)cluster.ratio<<' '<<crosses_real<<'\n';}
   }
   if(warm)previous=std::move(current);
  }
 }
}
