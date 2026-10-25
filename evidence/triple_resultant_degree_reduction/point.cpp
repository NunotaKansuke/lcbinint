#include "lcbinint/types.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
using namespace lcbinint;
namespace tr=lcbinint::algebraic_boundary::triple;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
 if(argc!=6)return 2;
 std::string engine=argv[1];int bins=std::stoi(argv[2]);double limb=std::stod(argv[3]),tol=std::stod(argv[4]);int repeats=std::stoi(argv[5]);
 SourcePosition s;double rho;std::cin>>s.x>>s.y>>rho;
 std::array<Complex,3> z;std::array<double,3> masses;
 for(int i=0;i<3;++i){double x,y;std::cin>>x>>y>>masses[i];z[i]={x,y};}
 auto p=tr::primary_frame(z,{s.x,s.y},rho,masses[1],masses[2]).geometry;
 tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=tol;
 tr::RadialEventPlan plan;tr::PreparedRegularSupport prepared;double prep=0;
 if(engine=="prepared"){auto t=Clock::now();plan=tr::radial_event_plan(p,opt.event_box_budget);prep=std::chrono::duration<double>(Clock::now()-t).count();}
 std::cout<<std::setprecision(17);
 auto num=[](double v){if(std::isfinite(v))std::cout<<v;else std::cout<<"null";};
 for(int i=0;i<repeats;++i){auto t=Clock::now();double mu,err;bool ok;std::string stop;int method=-1,domains=1,rn=0,re=0,rs=0;size_t nodes=0,quad=0,boxes=0;int qb=0,qr=0,dr=0,events=0;size_t continued=0,global=0;
 tr::RadialEventPlan regular;double fast_support_seconds=0;
 if(engine=="algebraic"){auto begin=Clock::now();regular=prepared.plan(p,tol);fast_support_seconds=std::chrono::duration<double>(Clock::now()-begin).count();}
 const char* support_route="";std::size_t bounded_components=0;double bounded_value=0,bounded_error=0;
 {auto r=tr::integrate(p,limb,opt,engine=="prepared"?&plan:regular.completed?&regular:nullptr);
 std::cerr<<"diagnostic "<<tol<<" "<<limb<<" "<<r.refinements<<" "<<r.h_splits<<" "<<r.normalized.angular_error<<" "<<r.normalized.endpoint_error<<" "<<r.error_estimate-r.normalized.angular_error-r.normalized.endpoint_error<<"\n";rn=r.support.resultant_nodes;re=r.support.resultant_events;rs=r.support.resultant_solves;domains=r.component_domains;support_route=r.support.reason;bounded_components=r.support.excluded_images.size();bounded_value=r.support.bounded_component_value;bounded_error=r.support.bounded_component_error;mu=r.magnification;err=r.error_estimate;ok=r.converged&&r.support_complete;stop=r.reason;nodes=r.value_evaluations;quad=r.quad_samples;boxes=r.support.boxes;qb=r.support.quad_boxes;qr=r.support.quad_refinements;dr=r.support.double_refinements;events=int(r.support.events.size());continued=r.continued_radii;global=r.global_radii;}
 double dt=std::chrono::duration<double>(Clock::now()-t).count();std::cout<<"{\"repeat\":"<<i<<",\"seconds\":"<<dt<<",\"preparation_seconds\":"<<prep<<",\"mu\":";num(mu);std::cout<<",\"error_estimate\":";num(err);std::cout<<",\"converged\":"<<(ok?"true":"false")<<",\"method\":"<<method<<",\"resultant_nodes\":"<<rn<<",\"resultant_events\":"<<re<<",\"resultant_solves\":"<<rs<<",\"component_domains\":"<<domains<<",\"stop\":\""<<stop<<"\",\"nodes\":"<<nodes<<",\"quad\":"<<quad<<",\"boxes\":"<<boxes<<",\"event_quad_boxes\":"<<qb<<",\"event_quad_refinements\":"<<qr<<",\"event_double_refinements\":"<<dr<<",\"events\":"<<events<<",\"continued_radii\":"<<continued<<",\"global_radii\":"<<global<<",\"support_route\":\""<<support_route<<"\",\"regular_stop\":\""<<regular.reason<<"\",\"fast_support_seconds\":"<<fast_support_seconds<<",\"bounded_components\":"<<bounded_components<<",\"bounded_component_value\":"<<bounded_value<<",\"bounded_component_error\":"<<bounded_error<<"}"<<std::endl;
 }
}
