#include "lcbinint/magnification/algebraic_boundary/triple/patterson_stream.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){
 int physical_fields=0;
 for(int i=0;i<720;++i){Geometry p{.1+.02*std::sin(i*.41),.15,.015+.3*(1+std::sin(i*.27)),.2,.08,1.,-.4,.6};const double R=.1+1.8*(i+.5)/720,theta=i*.618;shape_gradient_detail::Ray ray(p,R);double ea,eb,curvature;const auto x=ray.sample<true,true>(theta,ea),y=ray.sample<true,true>(theta,eb,&curvature);const auto check=[&](double a,double b){require(std::isfinite(a)&&std::abs(a-b)<=2e-13*(1+std::abs(b)),"real physical kernel versus curvature reference");++physical_fields;};check(x.phi,y.phi);check(x.dR,y.dR);check(x.dtheta,y.dtheta);check(ray.angular_slope(theta),y.dtheta);check(ea,eb);for(int j=0;j<8;++j)check(x.dp[j],y.dp[j]);}
 const int degrees[]={1,5,11,23,47};int moments=0;
 for(int level=1;level<=5;++level){const auto& q=patterson_rule(level);for(int power=0;power<=degrees[level-1];++power){long double sum=0;for(std::size_t k=0;k<q.x.size();++k)sum+=(long double)q.w[k]*std::pow((long double)q.x[k],power);const long double exact=power%2?0:2.L/(power+1);require(std::abs(sum-exact)<2e-14L,"Patterson analytic moment");++moments;}
  if(level>1){const auto& old=patterson_rule(level-1);for(std::size_t j=0;j<old.x.size();++j)require(q.x[2*j+1]==old.x[j],"bitwise nested radial nodes");}
 }
 int embedded_moments=0;
 for(int level=3;level<=5;++level){const auto& q=patterson_rule(level);const auto& e=embedded_patterson_detail::embedded_radial_rule(level);for(double w:e.weights)require(w>0,"positive embedded Patterson weights");for(int d=0;d<int(e.slots.size());++d){double sum=0;for(std::size_t j=0;j<e.slots.size();++j)sum+=e.weights[j]*std::pow(q.x[e.slots[j]],d);require(std::abs(sum-(d%2?0:2./(d+1)))<2e-13,"embedded Patterson polynomial moment");++embedded_moments;}}
 packet_cache_detail::DyadicPacketCache<double> packets;unsigned evaluations=0;
 for(unsigned level=1;level<=8;++level){packets.prepare(level);for(unsigned k=1;k<(1u<<level);++k){const auto index=packet_cache_detail::slot(level,k);const double coordinate=double(k)/(1u<<level);if(!packets.contains(index)){packets[index]=coordinate;++evaluations;}require(packets[index]==coordinate,"nested packet identity");}packets.finish(level);require(evaluations==(1u<<level)-1,"one packet evaluation per new dyadic node");}
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};const double a=.89,b=.90;
 const auto panel=[&](const Geometry& geometry){PattersonPanel<true> out(geometry,a,b,false,false,{},nullptr,0,1e-8,true,false,0,true);auto topology=angular_topology((a+b)*.5,geometry);require(topology.completed,"fixed band topology");out.set_certified_band(topology,a,b);return out;};
 auto g=panel(p);require(g.refine(3,2),"first order coarse panel");const auto coarse=g.work().evaluations;require(g.refine(5,2),"first order retained refinement");const auto fine=g.work().evaluations;require(fine>coarse,"refinement adds tangent work");require(g.refine(5,2)&&g.work().evaluations==fine,"same request reuses complete packets");auto fresh=panel(p);require(fresh.refine(5,2),"fresh fine panel");require(fresh.work().evaluations==fine,"incremental refinement computes each packet once");
 double Geometry::* fields[]={&Geometry::X,&Geometry::Y,&Geometry::rho,&Geometry::m1,&Geometry::m2,&Geometry::a,&Geometry::bx,&Geometry::by};
 for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}){auto plus=p,minus=p;plus.*fields[j]+=h;minus.*fields[j]-=h;RadialPanel<> vp(plus,a,b),vm(minus,a,b);require(vp.refine(7,2)&&vm.refine(7,2),"independent value panels");const double fd=(vp.result().f0-vm.result().f0)/(2*h),actual=g.result().df0[j];if(std::abs(actual-fd)>2e-7+2e-5*std::abs(fd)){std::cerr<<j<<" "<<h<<" "<<actual<<" "<<fd<<"\n";return 1;}}
 PattersonPanel<true> ld(p,a,b,false,false,{},nullptr,0,1e-8,false,false,1,true);auto ld_band=angular_topology((a+b)*.5,p);require(ld_band.completed,"LD fixed band topology");ld.set_certified_band(ld_band,a,b);require(ld.refine(5,4),"LD fixed band transport");
 for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}){auto plus=p,minus=p;plus.*fields[j]+=h;minus.*fields[j]-=h;RadialPanel<> vp(plus,a,b),vm(minus,a,b);require(vp.refine(7,7)&&vm.refine(7,7),"independent LD value panels");const double fd=(vp.result().fh-vm.result().fh)/(2*h),actual=ld.result().dfh[j];require(std::abs(actual-fd)<=2e-7+2e-5*std::abs(fd),"LD first order versus independent scalar FD");}
 int root_components=0;double root_worst=0;
 for(double r:{.45,.65,.85,1.05,1.25})for(double angle:{-2.,-1.,0.,1.,2.}){const auto pair=shape_gradient_detail::radial_complex_pair(r,angle,.25,1.5,p,true);if(!pair||!pair->material)continue;
  for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}){auto plus=p,minus=p;plus.*fields[j]+=h;minus.*fields[j]-=h;const auto hi=shape_gradient_detail::radial_complex_pair(r,angle,.25,1.5,plus),lo=shape_gradient_detail::radial_complex_pair(r,angle,.25,1.5,minus);if(!hi||!lo||std::abs(hi->center-pair->center)>.01||std::abs(lo->center-pair->center)>.01)continue;
   for(auto comparison:{std::pair<double,double>{pair->center_motion[j],(hi->center-lo->center)/(2*h)},std::pair<double,double>{pair->scale_motion[j],(hi->scale-lo->scale)/(2*h)}}){root_worst=std::max(root_worst,std::abs(comparison.first-comparison.second)/(1+std::abs(comparison.second)));++root_components;}
  }
 }
 require(root_components>=32&&root_worst<2e-6,"holomorphic stationary-root transport FD");
 std::cout<<embedded_moments<<" embedded moments, "<<physical_fields<<" physical fields, "<<moments<<" moments, retained first-order packets, 32 independent U/LD scalar FD checks, "<<root_components<<" stationary-root FD components\n";
}
