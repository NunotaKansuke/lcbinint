#include "lcbinint/magnification/algebraic_boundary/triple/radial_panel.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
void check(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main() {
    Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
    // Interior panel: no moving radial boundary is being differentiated.
    RadialPanel<> v(p,.89,.90);
    RadialPanel<true> g(p,.89,.90);
    check(v.refine(4,5) && g.refine(4,5),"coarse panel");
    // Error control must bound the requested profile combination, retaining
    // the same component quadratures and safety factor. At u=1/2 every
    // contribution is exactly half the unweighted sum on this fixed map.
    RadialPanel<> half(p,.89,.90,false,false,{},nullptr,0,0,false,false,.5);
    check(half.refine(4,5),"profile-weighted panel");
    check(half.result().f0==v.result().f0 && half.result().fh==v.result().fh,"profile weights preserve component values");
    check(std::abs(half.result().angular_error-.5*v.result().angular_error)<1e-14*v.result().angular_error,
          "profile-weighted quadrature error");
    check(std::abs(half.result().endpoint_error-.5*v.result().endpoint_error)<1e-14*(1+v.result().endpoint_error),
          "profile-weighted endpoint error");
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
