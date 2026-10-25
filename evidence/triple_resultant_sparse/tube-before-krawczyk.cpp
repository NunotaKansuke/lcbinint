#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#include "../triple_source_certificate/physical_box.hpp"
#include <iostream>
#include <chrono>
using namespace lcbinint::algebraic_boundary::triple;
using I=surface_detail::I;using Box=event_detail::Box;
using Clock=std::chrono::steady_clock;
event_detail::Jet<I> convert(physical_box_experiment::Jet<I> f){event_detail::Jet<I> g;g.v=f.v;g.t=f.t;g.r=f.r;g.rt=f.rt;g.tt=f.tt;return g;}
inline bool source_exclusion(const Box& box,const Geometry& p,double source_delta,const event_detail::PhysicalExclusionData* prepared=nullptr) {
    std::optional<event_detail::PhysicalExclusionData> local;
    if(!prepared){local.emplace(p);prepared=&*local;}
    const double eps=std::numeric_limits<double>::epsilon();
    const double r=box.rlo+(box.rhi-box.rlo)*.5,t=box.tlo+(box.thi-box.tlo)*.5;
    const double sign=box.reciprocal?-1:1,den=1+t*t;
    const Complex z=sign*r*Complex((1-t*t)/den,2*t/den),source(p.X,p.Y);
    const double chord=std::max(2*std::abs(box.tlo-t)/std::hypot(1.,box.tlo),
        2*std::abs(box.thi-t)/std::hypot(1.,box.thi))/std::hypot(1.,t);
    const double delta=std::hypot(.5*(box.rhi-box.rlo),std::sqrt(box.rhi)*std::sqrt(r)*chord)+64*eps*(1+box.rhi);
    const auto& lenses=prepared->lenses;const auto& masses=prepared->masses;
    std::array<double,3> image_distance{};
    for(int j=0;j<3;++j) {
        image_distance[j]=std::abs(z-lenses[j]);
        const double distance=image_distance[j]+delta+64*eps*(1+image_distance[j]);
        if(!(distance>0))continue;
        double remainder=prepared->source_distance[j]+source_delta+distance+p.rho;bool valid=true;
        for(int k=0;k<3;++k)if(k!=j) {
            const double lower=prepared->lens_distance[j][k]-distance-
                64*eps*(1+prepared->lens_distance[j][k]);
            if(!(lower>0)){valid=false;break;}
            remainder+=(masses[k]+4*eps)/lower;
        }
        if(valid && (masses[j]-4*eps)/distance>remainder*(1+128*eps)+128*eps)return true;
    }
    Complex g=z-source,U{};double error=delta+source_delta,U_error=0,U_condition=0,condition=1+std::abs(z)+prepared->source_norm;
    for(int j=0;j<3;++j) {
        const double distance=image_distance[j],lower=distance-delta-64*eps*(1+distance);
        if(!(lower>0))return false;
        const auto inv=1./std::conj(z-lenses[j]);
        g-=masses[j]*inv;
        if(j){U+=masses[j]*std::conj(lenses[j])*inv*inv;
            U_error+=2*(masses[j]+4*eps)*prepared->lens_norm[j]*delta/(lower*lower*lower)+4*eps*prepared->lens_norm[j]/(lower*lower);
            U_condition+=(masses[j]+4*eps)*std::abs(lenses[j])/(lower*lower);}
        error+=(masses[j]+4*eps)*delta/(lower*distance)+4*eps/lower;
        condition+=masses[j]/distance;
    }
    error+=128*eps*condition;U_error+=128*eps*(1+U_condition);
    const double norm=std::abs(g),margin=128*eps*(1+norm+p.rho+error);
    if(norm-error>p.rho+margin || norm+error<p.rho-margin)return true;
    // g_theta=i(z-conj(z)*kappa)=i(g+source-U).
    // Re(conj(g)*i*g)=0 exactly. The primary lens has l_0=0,
    // so it cancels from U instead of inflating a shear error bound.
    const Complex tangent=Complex(0,1)*(source-U);
    const double tangent_error=U_error+source_delta+256*eps*(1+prepared->source_norm+std::abs(U));
    const double product=std::real(std::conj(g)*tangent);
    const double product_error=norm*tangent_error+std::abs(tangent)*error+error*tangent_error+
        256*eps*(1+norm*std::abs(tangent));
    return std::abs(product)>product_error;
}

int main(){std::string name;
while(std::cin>>name){double x,y,rho;std::array<Complex,3> lenses;std::array<double,3> m;std::cin>>x>>y>>rho;for(int j=0;j<3;++j){double a,b;std::cin>>a>>b>>m[j];lenses[j]={a,b};}
auto p=primary_frame(lenses,{x,y},rho,m[1],m[2]).geometry;PreparedRegularSupport prep;auto plan=prep.plan(p,.001);if(!plan.completed){std::cerr<<name<<" no plan\n";return 1;}
for(double rel:{.001,.01,.05}){
auto start=Clock::now();double delta=rho*rel;I uncertainty(-delta,delta);std::vector<Box> tubes;int unique=0;
for(auto e:plan.events){
 auto fm=physical_box_experiment::Ray<I>(I(e.radius),p,uncertainty).event_jet(I(e.coordinate),e.reciprocal);if(!fm)continue;
 auto fc=physical_detail::Ray<double>(e.radius,p).event_jet(e.coordinate,e.reciprocal);if(!fc)continue;
 double det=fc->r*fc->tt-fc->t*fc->rt;if(det==0)continue;
 I dr=(I(fc->tt)*fm->v-I(fc->t)*fm->t)*I(1/det),dt=(-I(fc->rt)*fm->v+I(fc->r)*fm->t)*I(1/det);
 for(double scale:{1.5,2.,4.,8.}){double rw=scale*event_detail::magnitude(dr)+1e-10*(1+e.radius),tw=scale*event_detail::magnitude(dt)+1e-10;
 Box b{e.radius-rw,e.radius+rw,e.coordinate-tw,e.coordinate+tw,e.reciprocal};
 auto f=physical_box_experiment::Ray<I>(I(b.rlo,b.rhi),p,uncertainty).event_jet(I(b.tlo,b.thi),b.reciprocal);if(!f)continue;
 auto proof=event_detail::proof_from_jets(b,convert(*f),convert(*fm));if(proof.kind==2){tubes.push_back(b);++unique;break;}
 }
}
std::vector<Box> coverage=tubes;
for(auto b:tubes)if(b.tlo>0 || b.thi<0){double lo=interval_detail::up(-1/b.tlo),hi=interval_detail::down(-1/b.thi);if(hi>lo)coverage.push_back({b.rlo,b.rhi,lo,hi,!b.reciprocal});}
Geometry envelope=p;envelope.X=std::copysign(interval_detail::up(std::abs(p.X)+delta),p.X);envelope.Y=std::copysign(interval_detail::up(std::abs(p.Y)+delta),p.Y);RadialEventPlan domain;event_detail::radial_domain(envelope,domain);
struct Cell{Box box;int depth;};std::vector<Cell> pending;int nodes=0,unresolved=0;event_detail::PhysicalExclusionData exclusion_data(p);
if(unique==int(plan.events.size()))pending={{{domain.rmin,domain.rmax,-1,1,false},0},{{domain.rmin,domain.rmax,-1,1,true},0}};
while(!pending.empty()){
 auto c=pending.back();pending.pop_back();auto b=c.box;if(nodes>=32768){++unresolved;continue;}++nodes;
 bool removed=false;
 for(auto tube:coverage)if(tube.reciprocal==b.reciprocal){double rl=std::max(b.rlo,tube.rlo),rh=std::min(b.rhi,tube.rhi),tl=std::max(b.tlo,tube.tlo),th=std::min(b.thi,tube.thi);if(rh>rl && th>tl){
 for(auto child:std::array<Box,4>{{{b.rlo,rl,b.tlo,b.thi,b.reciprocal},{rh,b.rhi,b.tlo,b.thi,b.reciprocal},{rl,rh,b.tlo,tl,b.reciprocal},{rl,rh,th,b.thi,b.reciprocal}}})if(child.rhi>child.rlo&&child.thi>child.tlo)pending.push_back({child,c.depth});removed=true;break;}}
 if(removed)continue;
 if(source_exclusion(b,p,2*delta,&exclusion_data))continue;
 auto f=physical_box_experiment::Ray<I>(I(b.rlo,b.rhi),p,uncertainty).angular(I(b.tlo,b.thi),b.reciprocal);
 if(f && (std::abs(f->v.sign())==1 || std::abs(f->t.sign())==1))continue;
 if(c.depth>=28){++unresolved;continue;}
 if((b.rhi-b.rlo)/(domain.rmax-domain.rmin)>(b.thi-b.tlo)/2){double mid=b.rlo+(b.rhi-b.rlo)*.5;pending.push_back({{mid,b.rhi,b.tlo,b.thi,b.reciprocal},c.depth+1});pending.push_back({{b.rlo,mid,b.tlo,b.thi,b.reciprocal},c.depth+1});}
 else {double mid=b.tlo+(b.thi-b.tlo)*.5;pending.push_back({{b.rlo,b.rhi,mid,b.thi,b.reciprocal},c.depth+1});pending.push_back({{b.rlo,b.rhi,b.tlo,mid,b.reciprocal},c.depth+1});}
}
double ms=std::chrono::duration<double>(Clock::now()-start).count()*1000;
std::cout<<"{\"case\":\""<<name<<"\",\"relative_halfwidth\":"<<rel<<",\"events\":"<<plan.events.size()<<",\"unique_tubes\":"<<unique<<",\"nodes\":"<<nodes<<",\"unresolved\":"<<unresolved<<",\"complete\":"<<(unique==int(plan.events.size())&&unresolved==0)<<",\"build_ms\":"<<ms<<"}\n";
}
}
}
