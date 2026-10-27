#include "lcbinint/magnification/algebraic_boundary/triple/radial_panel.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
void check(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main() {
    // Embedded rules reuse existing samples, retain positive weights and
    // integrate their polynomial moments without invoking lens arithmetic.
    for(int level:{3,4,5}) {
        const auto& embedded=panel_detail::embedded_radial_rule(level);
        const auto& fine=lcbinint::algebraic_boundary::fejer_rule(level);
        for(double weight:embedded.weights)check(weight>0,"positive embedded radial weights");
        for(std::size_t degree=0;degree<=embedded.slots.size();++degree) {
            long double integral=0;
            for(std::size_t j=0;j<embedded.slots.size();++j)
                integral+=(long double)embedded.weights[j]*std::pow((long double)fine.x[embedded.slots[j]],int(degree));
            const long double exact=degree%2?0:2.L/(degree+1);
            check(std::abs(integral-exact)<5e-13L,"embedded radial polynomial moments");
            if(degree==embedded.slots.size())continue;
            for(int side=0;side<2;++side) {
                long double interpolated=0;
                for(std::size_t j=0;j<embedded.slots.size();++j)
                    interpolated+=(long double)embedded.interpolation[side][j]*
                        std::pow((long double)fine.x[embedded.slots[j]],int(degree));
                check(std::abs(interpolated-std::pow((long double)fine.x[embedded.missing[side]],int(degree)))<5e-12L,
                    "embedded omitted-node interpolation witness");
            }
        }
    }
    Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
    // Interior panel: no moving radial boundary is being differentiated.
    RadialPanel<> v(p,.89,.90);
    RadialPanel<true> g(p,.89,.90);
    check(v.refine(4,5) && g.refine(4,5),"coarse panel");
    // Error control must bound the requested profile combination, retaining
    // the same component quadratures and safety factor. Estimating the
    // physical profile directly can reduce the separate triangle bound.
    RadialPanel<> half(p,.89,.90,false,false,{},nullptr,0,0,false,false,.5);
    check(half.refine(4,5),"profile-weighted panel");
    check(half.result().f0==v.result().f0 && half.result().fh==v.result().fh,"profile weights preserve component values");
    check(half.result().angular_error<=.5*v.result().angular_error+1e-14*v.result().angular_error,
          "physical-profile quadrature triangle bound");
    check(std::abs(half.result().endpoint_error-.5*v.result().endpoint_error)<1e-14*(1+v.result().endpoint_error),
          "profile-weighted endpoint error");
    RadialPanel<true> gradient_half(p,.89,.90,false,false,{},nullptr,0,0,false,false,.5);
    check(gradient_half.refine(4,5),"gradient profile projection");
    for(int j=0;j<8;++j) {
        check(gradient_half.result().df0[j]==g.result().df0[j] &&
              gradient_half.result().dfh[j]==g.result().dfh[j],"gradient profile preserves values");
        check(gradient_half.result().gradient_error[j]<=.5*g.result().gradient_error[j]+1e-14,
              "projected gradient error respects triangle bound");
    }
    auto count=v.value_evaluations();
    check(v.refine(6,7) && g.refine(6,7),"panel refinement");
    check(v.radial_evaluations==63,"each radial node solved once");
    check(v.reused_radii>0 && v.value_evaluations()>count,"radial sample retention");
    check(v.result().f0==g.result().f0 && v.result().fh==g.result().fh,"panel value gradient parity");
    const double u=.5;
    auto value=normalize(v.result(),p,u);
    auto gradient=normalize(g.result(),p,u);
    double Geometry::* members[]={&Geometry::X,&Geometry::Y,&Geometry::rho,
        &Geometry::m1,&Geometry::m2,&Geometry::a,&Geometry::bx,&Geometry::by};
    for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}) {
        auto plus=p,minus=p;plus.*members[j]+=h;minus.*members[j]-=h;
        RadialPanel<> vp(plus,.89,.90),vm(minus,.89,.90);
        if(!vp.refine(6,7) || !vm.refine(6,7)) {std::cerr<<j<<" "<<h<<" "<<vp.reason<<" "<<vm.reason<<"\n";return 1;}
        double fd=(normalize(vp.result(),plus,u).f0-normalize(vm.result(),minus,u).f0)/(2*h);
        if(std::abs(fd-gradient.df0[j])>3e-4*(1+std::abs(fd))) {
            std::cerr<<j<<" "<<fd<<" "<<gradient.df0[j]<<"\n";return 1;
        }
    }
    // Differentiate both physical geometry and moving panel coordinates.
    // This checks the R derivative and map-Jacobian derivative independently
    // against finite differences of the value-only mapped integral.
    const std::array<double,8> left_motion{.03,-.02,.1,.04,-.03,.02,.01,-.01};
    const std::array<double,8> right_motion{-.01,.01,.2,-.02,.01,.03,-.01,.02};
    std::array<double,8> slope;
    for(int j=0;j<8;++j)slope[j]=(right_motion[j]-left_motion[j])/.01;
    RadialPanel<true> moving(p,.89,.90,true,true);
    moving.set_radial_motion(.89,left_motion,slope);
    check(moving.refine(6,7),"moving panel refinement");
    const auto moving_gradient=normalize(moving.result(),p,u);
    for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}) {
        auto plus=p,minus=p;plus.*members[j]+=h;minus.*members[j]-=h;
        RadialPanel<> vp(plus,.89+h*left_motion[j],.90+h*right_motion[j],true,true);
        RadialPanel<> vm(minus,.89-h*left_motion[j],.90-h*right_motion[j],true,true);
        check(vp.refine(6,7) && vm.refine(6,7),"moving value reference");
        const double fd=(normalize(vp.result(),plus,u).f0-normalize(vm.result(),minus,u).f0)/(2*h);
        check(std::abs(fd-moving_gradient.df0[j])<3e-4*(1+std::abs(fd)),"moving coordinate derivative");
    }
    // Uniform isolated components use the same weighted radial nodes as
    // value integration. Check both material motion and its map Jacobian.
    RadialPanel<true> weighted_moving(p,.89,.90,true,true,{},nullptr,0,0,true,true,0);
    weighted_moving.set_radial_motion(.89,left_motion,slope);
    check(weighted_moving.refine(6,2),"weighted moving panel");
    const auto weighted_gradient=normalize(weighted_moving.result(),p,0);
    const auto weighted_radii=weighted_moving.radial_evaluations;
    check(weighted_moving.refine(5,2) && weighted_moving.radial_evaluations==weighted_radii,
          "weighted gradients retain samples");
    for(int j=0;j<8;++j)for(double h:{1e-5,3e-6}) {
        auto plus=p,minus=p;plus.*members[j]+=h;minus.*members[j]-=h;
        RadialPanel<> vp(plus,.89+h*left_motion[j],.90+h*right_motion[j],true,true,{},nullptr,0,0,true,true,0);
        RadialPanel<> vm(minus,.89-h*left_motion[j],.90-h*right_motion[j],true,true,{},nullptr,0,0,true,true,0);
        check(vp.refine(6,2) && vm.refine(6,2),"weighted value reference");
        const double fd=(normalize(vp.result(),plus,0).f0-normalize(vm.result(),minus,0).f0)/(2*h);
        check(std::abs(fd-weighted_gradient.df0[j])<3e-4*(1+std::abs(fd)),"weighted moving derivative");
    }
    // Independent midpoint-in-radius integration using separately generated
    // angular topologies; tests the shared radial mapping and quadrature.
    double f0=0,fh=0;const int n=256;
    for(int k=0;k<n;++k) {
        double R=.89+.01*(k+.5)/n;
        RadiusWorkspace<> r;
        check(r.initialize(R,p) && r.refine(7),"radial midpoint reference");
        f0+=r.result().f0*.01/n;fh+=r.result().fh*.01/n;
    }
    check(std::abs(f0-v.result().f0)<2e-8 && std::abs(fh-v.result().fh)<2e-8,"radial midpoint parity");
    // Angular-only refinement reuses the same radial workspaces.
    const auto old_radii=v.radial_evaluations;
    check(v.refine(6,8) && v.radial_evaluations==old_radii,"angular-only refinement retention");
    // Preserve the failed smooth-panel candidate as a regression: this panel
    // crosses a 2->4-root event. It must not be declared tolerance-converged.
    RadialPanel<> contact(p,.89,.93);
    check(contact.refine(6,7),"contact panel candidate");
    check(contact.result().angular_error>1e-6,"contact panel reports unresolved radial error");
    auto cache=std::make_shared<SharedRadiusCache<false>>();
    RadialPanel<> owned(p,.89,.90,false,false,{},cache);
    auto changed=p;changed.X+=.01;bool rejected=false;
    try {RadialPanel<> stale(changed,.89,.90,false,false,{},cache);}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"shared cache rejects different geometry");
    ImageExclusion ball{{1.,0.},.1};AngularOptions masked;masked.inclusion=&ball;rejected=false;
    try {RadialPanel<> stale_mask(p,.89,.90,false,false,masked,cache);}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"shared cache rejects different component mask");
    std::cout<<"PASS: partial radial panel, normalized value="<<value.f0
             <<", 8 derivatives at two steps, retained radii="<<v.radial_evaluations<<"\n";
}
