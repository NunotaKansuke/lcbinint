#include "lcbinint/magnification/algebraic_boundary/triple/radius_terms.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
void check(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main() {
    Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
    double Geometry::* members[]={&Geometry::X,&Geometry::Y,&Geometry::rho,
        &Geometry::m1,&Geometry::m2,&Geometry::a,&Geometry::bx,&Geometry::by};
    // Compare the cached value-only quad ray against the original three-lens
    // evaluator; retain the latter as an independent promoted expression.
    for(double R:{.2,.82,.93,1.04,1.19,2.3}) {
        const local_precision::ValueRay ray(R,p);
        for(int k=0;k<40;++k) {
            const local_precision::Q theta=local_precision::Q(-1.5)+local_precision::Q(k)*local_precision::Q(.153);
            const auto a=ray.evaluate<true>(theta);
            const auto b=local_precision::evaluate<false,true>(local_precision::Q(R),theta,p);
            check(fabsq(a.phi-b.phi)<local_precision::Q(1e-27)*(1+fabsq(b.phi)),"cached quad value parity");
            check(fabsq(a.dtheta-b.dtheta)<local_precision::Q(1e-26)*(1+fabsq(b.dtheta)),"cached quad tangent parity");
        }
    }
    // The optimized derivative ray must agree with the original promoted
    // physical equations, including all mass and position derivatives.
    for(double R:{.2,.82,.93,1.04,1.19,2.3}) {
        const local_precision::GradientRay ray(R,p);
        for(int k=0;k<80;++k) {
            const local_precision::Q theta=local_precision::Q(-3)+local_precision::Q(k)*local_precision::Q(.079);
            const auto a=ray.evaluate(theta);
            const auto b=local_precision::evaluate<true>(local_precision::Q(R),theta,p);
            const auto close=[](auto x,auto y){return fabsq(x-y)<local_precision::Q(1e-26)*(1+fabsq(y));};
            check(close(a.phi,b.phi) && close(a.dtheta,b.dtheta) && close(a.dR,b.dR),"cached quad gradient geometry parity");
            for(int j=0;j<8;++j)check(close(a.dp[j],b.dp[j]),"cached quad parameter parity");
        }
    }
    int occupied=0;
    for(double R:{.82,.89,.93,1.04,1.115,1.19}) {
        RadiusWorkspace<> value;
        RadiusWorkspace<true> gradient;
        check(value.initialize(R,p) && gradient.initialize(R,p),"topology");
        for(auto bracket:value.topology.roots) {
            const double theta=bracket.lo+(bracket.hi-bracket.lo)*.5;
            local_precision::Q rational,angular;
            check(local_precision::endpoint(R,theta,p,rational),"rational quad endpoint");
            check(local_precision::refine_endpoint(theta,p,angular,[&](local_precision::Q angle){
                return local_precision::evaluate<false,true>(local_precision::Q(R),angle,p);
            }),"independent angular quad endpoint");
            check(fabsq(rational-angular)<local_precision::Q(1e-24)*(1+fabsq(angular)),"quad endpoint parity");
        }
        check(value.refine(4) && gradient.refine(4),"coarse integral");
        auto evaluations=value.value_evaluations;
        check(value.refine(7) && gradient.refine(7),"fine integral");
        check(value.value_evaluations>=evaluations,"monotone work");
        check(value.gradient_evaluations==0,"value path no gradient work");
        check(value.reused_samples>0 || value.topology.arcs.empty(),"nested sample reuse");
        check(value.value_evaluations==value.topology.arcs.size()*127,"each node evaluated once");
        check(value.result().f0==gradient.result().f0 && value.result().fh==gradient.result().fh,"value gradient parity");
        if(value.topology.arcs.empty())continue;
        ++occupied;
        // Independent composite midpoint quadrature on the angular arcs.
        double reference=0;
        for(auto arc:value.topology.arcs) {
            const int n=65536;const double h=(arc.hi-arc.lo)/n;
            for(int j=0;j<n;++j)
                reference+=R*h*std::sqrt(std::max(0.,evaluate(R,arc.lo+(j+.5)*h,p).phi));
        }
        check(std::abs(reference-value.result().fh)<2e-7,"independent limb integral");
        for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}) {
            auto plus=p,minus=p;plus.*members[j]+=h;minus.*members[j]-=h;
            RadiusWorkspace<> vp,vm;
            check(vp.initialize(R,plus) && vm.initialize(R,minus),"perturbed topology");
            check(vp.refine(7) && vm.refine(7),"perturbed integral");
            const double df0=(vp.result().f0-vm.result().f0)/(2*h);
            const double dfh=(vp.result().fh-vm.result().fh)/(2*h);
            if(std::abs(df0-gradient.result().df0[j])>3e-4*(1+std::abs(df0)) ||
               std::abs(dfh-gradient.result().dfh[j])>3e-4*(1+std::abs(dfh))) {
                std::cerr<<R<<" "<<j<<" "<<h<<" df0 "<<df0<<" "<<gradient.result().df0[j]
                         <<" dfh "<<dfh<<" "<<gradient.result().dfh[j]<<"\n";
                return 1;
            }
        }
        auto count=value.value_evaluations;
        check(value.refine(6) && value.value_evaluations==count,"lower refinement reuses result");
    }
    // Exercise the loose value-only fixed-radius evaluator and per-radius
    // angular budget against the independent physical arc quadrature above.
    // Tightening a ceiling must retain all existing samples and improve the
    // estimate without executing parameter-derivative code.
    for(double R:{.82,.89,.93,1.04,1.115,1.19}) {
        RadiusWorkspace<> local,reference;
        check(local.initialize(R,p,{},false,1e-6),"local topology");
        check(reference.initialize(R,p) && reference.refine(9),"local reference");
        check(local.refine_local(5,5e-6,0),"local budget");
        const auto work=local.value_evaluations;
        check(local.refine_local(8,5e-7,0),"tighter local budget");
        check(local.value_evaluations>=work && local.gradient_evaluations==0,"local retained value work");
        const auto v=local.result(),r=reference.result();
        check(std::abs(v.fh-r.fh)<=4*(v.angular_error+r.angular_error+v.endpoint_error)+1e-10,"local integral accuracy");
        check(std::abs(v.f0-r.f0)<1e-13,"local uniform component parity");
        const auto retained=local.value_evaluations;
        check(local.refine_local(4,5e-6,0) && local.value_evaluations==retained,"local lower ceiling reuse");
    }
    // Repair keeps a separate nested sample ledger. Repeating a repair must
    // reuse samples and must not add the previous roundoff ledger twice.
    for(double R:{.82,.89,.93,1.04,1.115,1.19}) {
        RadiusWorkspace<> local,reference;
        check(local.initialize(R,p,{},false,1e-6) && local.refine(4),"repair initial radius");
        check(reference.initialize(R,p) && reference.refine(10),"repair reference radius");
        check(local.repair(1e-8),"budgeted repair");
        const auto first=local.result();const auto work=local.value_evaluations;
        check(local.repair(1e-8),"repeated repair");
        check(local.value_evaluations==work && local.gradient_evaluations==0,"repair sample reuse");
        check(local.result().endpoint_error==first.endpoint_error,"repair ledger counted once");
        const auto ref=reference.result();
        check(std::abs(first.fh-ref.fh)<=4*(first.angular_error+first.endpoint_error+ref.angular_error)+1e-10,"repaired independent arc rule parity");
    }
    check(occupied>=2,"occupied cases");
    // Full-circle geometry: fixed artificial endpoints have zero derivative.
    Geometry full{.01,.02,5.,.2,.08,1.,-.4,.6};
    RadiusWorkspace<true> circle;
    check(circle.initialize(1.2,full) && circle.refine(7),"full circle");
    check(circle.topology.roots.empty() && circle.topology.arcs.size()==1,"full circle classification");
    check(std::abs(circle.result().f0-1.2*2*std::acos(-1.))<1e-14,"full circle area term");
    for(double d:circle.result().df0)check(d==0,"fixed full-circle endpoints");
    std::cout<<"PASS: retained nested angular integration and 8 derivatives at two steps; occupied="<<occupied<<"\n";
}
