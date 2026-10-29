#pragma once
#include "integrator.hpp"
#include "frame.hpp"
#include "regular_support.hpp"
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple {
struct PolePartitionProof {
    ImageExclusion region;
    double radius_lo=0,radius_hi=0,magnification_upper=0;
};
// Isolate m/conj(delta), not its singular Jacobian. For every source in
// the disk the resulting fixed-point map contracts a pole-centred ball.
// A separate annular inequality excludes every other preimage throughout
// the larger partition region. The region may contain the lens pole;
// it is a partition boundary, not an image-space Newton ball.
inline std::optional<PolePartitionProof> prove_pole_partition(
        const Geometry& p,int pole,double region_radius) {
    using B=interval_detail::Interval<double>;
    using C=coefficient_detail::C<B>;
    if(pole<0 || pole>2 || !(region_radius>0) || !interval_detail::environment_ok())return std::nullopt;
    const auto inv=[](B x){return B(interval_detail::down(1/x.hi),interval_detail::up(1/x.lo));};
    const auto norm=[](C z){
        const auto square=[](B x){const double a=std::min(std::abs(x.lo),std::abs(x.hi)),b=std::max(std::abs(x.lo),std::abs(x.hi));return B(x.lo<=0 && x.hi>=0?0:interval_detail::down(a*a),interval_detail::up(b*b));};
        const B n=square(z.re)+square(z.im);
        return B(interval_detail::down(std::sqrt(std::max(0.,n.lo))),interval_detail::up(std::sqrt(n.hi)));
    };
    const auto l=p.lenses();const std::array<B,3> m{B(1)-B(p.m1)-B(p.m2),B(p.m1),B(p.m2)};
    for(auto a:m)if(!(a.lo>0) || !a.valid())return std::nullopt;
    C H{B(l[pole].real())-B(p.X),B(l[pole].imag())-B(p.Y)};
    std::array<B,3> distances{};
    for(int j=0;j<3;++j)if(j!=pole) {
        const C d{B(l[pole].real())-B(l[j].real()),B(l[pole].imag())-B(l[j].imag())};
        distances[j]=norm(d);if(!(distances[j].lo>region_radius))return std::nullopt;
        const B n=d.re*d.re+d.im*d.im;if(!(n.lo>0))return std::nullopt;
        H=H-C{m[j]}*d*C{inv(n)};
    }
    const B h=norm(H),rho(p.rho),R(region_radius);
    const auto variation=[&](B r) {
        B e=r;for(int j=0;j<3;++j)if(j!=pole)e=e+m[j]*r*inv(distances[j]*(distances[j]-r));return e;
    };
    const B gap=h-rho-variation(R);if(!(gap.lo>0))return std::nullopt;
    const B core=B(2)*m[pole]*inv(gap);
    if(!core.valid() || !(core.hi<region_radius))return std::nullopt;
    const B c(core.hi),e=variation(c),den=h-rho-e;
    if(!(den.lo>0))return std::nullopt;
    B slope(1);for(int j=0;j<3;++j)if(j!=pole){const B d=distances[j]-c;slope=slope+m[j]*inv(d*d);}
    const B q=m[pole]*slope*inv(den*den),upper=m[pole]*inv(den);
    if(!q.valid() || !(q.hi<1) || !(upper.hi<c.lo))return std::nullopt;
    // No source preimage in core <= |delta| <= region_radius.
    if(!((h-rho-variation(R)-m[pole]*inv(c)).lo>0))return std::nullopt;
    const B lower=m[pole]*inv(h+rho+e),u(upper.hi);
    B shear=m[pole]*inv(u*u);
    for(int j=0;j<3;++j)if(j!=pole){const B d=distances[j]-u;shear=shear-m[j]*inv(d*d);}
    if(!(lower.lo>0) || !(upper.hi>lower.lo) || !(shear.lo>1))return std::nullopt;
    const B bound=inv(shear*shear-B(1));if(!bound.valid())return std::nullopt;
    return PolePartitionProof{{l[pole],region_radius},lower.lo,upper.hi,bound.hi};
}
struct PolePartitionResult:IntegrationDiagnostics {std::array<double,11> physical_gradient{};};
template<bool Gradient> inline PolePartitionResult integrate_pole_partition(
        const std::array<Complex,3>& lenses,Complex source,double rho,
        const std::array<double,3>& masses,double limb,const IntegrationOptions& requested) {
    PolePartitionResult out;
    int primary=int(std::max_element(masses.begin(),masses.end())-masses.begin());
    const std::array<int,3> order{primary,(primary+1)%3,(primary+2)%3};
    const auto frame=primary_frame<Gradient>({lenses[order[0]],lenses[order[1]],lenses[order[2]]},source,rho,masses[order[1]],masses[order[2]]);
    const auto& p=frame.geometry;const auto ls=p.lenses();
    PartialComponents components;components.identity={p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};
    std::array<PolePartitionProof,2> proofs;
    for(int j=1;j<3;++j) {
        double distance=INFINITY;for(int k=0;k<3;++k)if(k!=j)distance=std::min(distance,std::abs(ls[j]-ls[k]));
        const auto proof=prove_pole_partition(p,j,.001*distance);
        if(!proof){out.reason="pole_partition_unproved";return out;}
        proofs[j-1]=*proof;components.images.push_back(proof->region);
    }
    if(!(std::abs(components.images[0].center-components.images[1].center)>
         components.images[0].radius+components.images[1].radius)){out.reason="pole_partition_overlap";return out;}
    auto support=radial_event_plan(p,std::min(requested.event_box_budget,65536),0.,&components);
    if(!support.completed){out.reason=support.reason;return out;}
    support.excluded_images=components.images;support.component_images.clear();
    auto options=requested;options.relative_tolerance*=.5;options.absolute_tolerance*=.5;
    options.complete_partition_member=true;options.separate_components=false;
    const auto main=integrate_coupled<Gradient>(p,limb,options,&support);
    if(!main.converged || !main.support_complete || (Gradient && !main.gradient_valid)){out.reason=main.reason;return out;}
    out.magnification=main.magnification;out.error_estimate=main.error_estimate;
    out.value_evaluations=main.value_evaluations;out.parameter_derivative_evaluations=main.parameter_derivative_evaluations;
    if constexpr(Gradient)out.physical_gradient=unpermute_physical_gradient(frame.pullback(main.normalized.df0),order);
    for(int j=1;j<3;++j) {
        const std::array<int,3> local_order{order[j],order[0],order[3-j]};
        const auto local=primary_frame<Gradient>({lenses[local_order[0]],lenses[local_order[1]],lenses[local_order[2]]},source,rho,masses[local_order[1]],masses[local_order[2]]);
        const auto proof=prove_pole_partition(local.geometry,0,proofs[j-1].region.radius);
        if(!proof){out.reason="local_pole_partition_unproved";return out;}
        RadialEventPlan plan;const auto& a=local.geometry;
        plan.identity={a.X,a.Y,a.rho,a.m1,a.m2,a.a,a.bx,a.by};plan.completed=true;
        plan.unisolated_folds=true;plan.rmin=proof->radius_lo;plan.rmax=proof->radius_hi;
        plan.included_image=ImageExclusion{{0,0},proofs[j-1].region.radius};
        options.absolute_tolerance=(requested.absolute_tolerance+requested.relative_tolerance)/64;
        const auto part=integrate_coupled<Gradient>(a,limb,options,&plan);
        if(!part.converged || !part.support_complete || (Gradient && !part.gradient_valid)){out.reason=part.reason;return out;}
        out.magnification+=part.magnification;
        // Also cover the different rounded physical chart representations.
        // Every image value and IFT was evaluated; none is replaced by a bound.
        out.error_estimate+=std::max(part.error_estimate,proofs[j-1].magnification_upper+std::abs(part.magnification));
        out.value_evaluations+=part.value_evaluations;out.parameter_derivative_evaluations+=part.parameter_derivative_evaluations;
        if constexpr(Gradient){const auto g=unpermute_physical_gradient(local.pullback(part.normalized.df0),local_order);for(int k=0;k<11;++k)out.physical_gradient[k]+=g[k];}
    }
    out.converged=std::isfinite(out.magnification) && out.magnification>0 &&
        out.error_estimate<=requested.absolute_tolerance+requested.relative_tolerance*std::abs(out.magnification);
    out.support_complete=out.converged;
    if constexpr(Gradient)out.gradient_valid=out.converged && std::all_of(out.physical_gradient.begin(),out.physical_gradient.end(),[](double x){return std::isfinite(x);});
    out.reason=out.converged?"complete_pole_partition":"pole_partition_error_budget";
    return out;
}
}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
