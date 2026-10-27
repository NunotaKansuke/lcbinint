#pragma once
#include "root_pair.hpp"
#include "root_pair_reverse.hpp"
#include "critical_pair.hpp"
#include "radius_terms.hpp"
#include "gradient_angular_chart.hpp"
#include "quartic_angular_geometry.hpp"
#include "global_angular_stream.hpp"
#include "dyadic_packet_cache.hpp"
#ifdef LCBININT_TRIPLE_COMPACT_CRITICAL_TRANSPORT
#include "compact_critical_transport.hpp"
#endif
#ifdef LCBININT_TRIPLE_COFACTOR_CORE
#include "cofactor_core_adjoint.hpp"
#endif
#ifdef LCBININT_TRIPLE_ENDPOINT_COFACTOR
#include "endpoint_cofactor_transport.hpp"
#endif
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
template<int N> struct PairIntegral {double flux=0,error=0;std::array<double,N> gradient{};};
// Regular first-order transport in the same deflated coordinates as the
// binary tangent kernel. No mixed radial/parameter jet is required.
template<int N,class Work> std::optional<PairIntegral<N>> ordinary_arc(
 const root_pair_detail::Arc<N>& model,double gate,Work& work,double& error){
 LCBININT_COUPLED_SCOPE(pair);
 struct Point {root_pair_detail::Jet<N> field;};packet_cache_detail::DyadicPacketCache<Point> points;
 const int minimum=model.disk_resolved(4)?3:4;
 double previous=0;
 for(int level=2;level<=8;++level){const int n=1<<level;const auto& rule=angular_rule_extended(level);points.prepare(level);PairIntegral<N> total;
  for(int k=1;k<n;++k){const unsigned slot=packet_cache_detail::slot(level,k);auto& point=points[slot];if(!points.contains(slot)){if(!model.density(rule.x[k-1],point.field))return std::nullopt;++work.evaluations;}
   total.flux+=rule.norm_w[k-1]*point.field.value;for(int j=0;j<N;++j)total.gradient[j]+=rule.norm_w[k-1]*point.field.dp[j];}
  points.finish(level);
  if(level>=minimum&&std::abs(total.flux-previous)<=gate*std::abs(total.flux)){error=std::abs(total.flux-previous);return total;}previous=total.flux;
 }
 return std::nullopt;
}
// Re-express the same critical pair in a new frozen half-angle chart.
// This carries its eight RHS and deflated quartic without another root solve.
template<int N> bool move_pair(const root_pair_detail::Arc<N>& original,double basis,
        double R,const Geometry& p,const std::array<double,N>& motion,
        std::optional<root_pair_detail::Coefficients<N>>* coefficients,
        const root_pair_detail::RadialCache<N>* radial,root_pair_detail::Arc<N>& out) {
    using J=root_pair_detail::Jet<N>;const double pi=std::acos(-1.);
    const double angle=std::remainder(basis-original.center,2*pi);
    if(std::abs(angle)<1e-14){out=original;out.center=basis;return true;}
    if(!out.configure(R,p,basis,motion,coefficients,radial))return false;
    const double kappa=std::tan(angle*.5),f=1+kappa*kappa;
    const J a=J(1)+J(kappa)*original.pair.mean;
    const J den=a*a-J(kappa*kappa)*original.pair.square,inverse=root_pair_detail::inverse(den);
    out.pair.mean=((original.pair.mean-J(kappa))*a-J(kappa)*original.pair.square)*inverse;
    out.pair.square=original.pair.square*J(f*f)*inverse*inverse;
    const J prefactor=den*J(1/(f*f*f));
    std::array<double,5> power{};power[0]=1;for(int j=1;j<5;++j)power[j]=power[j-1]*kappa;
    for(int i=0;i<5;++i) {
        J q;
        for(int j=0;j<5;++j) {
            double weight=0;for(int a=std::max(0,i-(4-j));a<=std::min(i,j);++a)
                weight+=root_pair_detail::binomial(j,a)*root_pair_detail::binomial(4-j,i-a)*power[j-a]*power[i-a]*((i-a)%2?-1:1);
            q.value+=weight*original.pair.quotient[j].value;
            for(int k=0;k<N;++k)q.dp[k]+=weight*original.pair.quotient[j].dp[k];
        }
        out.pair.quotient[i]=q*prefactor;
    }
    out.half_width=std::sqrt(std::abs(out.pair.square.value));out.pair.valid=true;
    return out.half_width>0 && std::isfinite(out.half_width);
}
// Only critical geometry selects this route. Ordinary arcs retain their
// incumbent kernel. A local failure retains the discovered minima for the
// incumbent repair, including its existing quadrature and precision policy.
#ifdef LCBININT_TRIPLE_REVERSE_ANGULAR
template<int N,class Ray,class Work> std::optional<PairIntegral<N>> critical_arc_reverse(
        const Ray& ray,double R,const Geometry& p,double lo,double hi,bool physical_ends,
        const std::array<double,N>& motion,Work& work,
        std::optional<root_pair_detail::Coefficients<N>>& coefficients,
        std::unique_ptr<root_pair_detail::RadialCache<N>>* shared_radial,
        std::optional<AngularChart>& retained,int quadrature_order=8,const std::array<double,N>* lower_motion=nullptr,const std::array<double,N>* upper_motion=nullptr,const AngularStationarity* stationary=nullptr,const typename Ray::Core* midpoint=nullptr,std::optional<root_pair_detail::Coefficients<0>>* scalar_coefficients=nullptr,double value_gate=1e-6) {
    static_assert(N==8);if(!lower_motion||!upper_motion)return std::nullopt;
    std::optional<root_pair_detail::Coefficients<0>> owned_scalar;if(!scalar_coefficients)scalar_coefficients=&owned_scalar;
    LCBININT_COUPLED_SCOPE(chart);
    {LCBININT_CHART_SCOPE(discovery);
     std::optional<std::array<root_pair_detail::Jet<0>,7>> scalar_harmonics;
     if(lower_motion&&upper_motion&&midpoint&&midpoint->phi>0&&midpoint->roundoff<1e-7*midpoint->phi&&scalar_coefficients&&!*scalar_coefficients&&shared_radial){if(!*shared_radial)*shared_radial=std::make_unique<root_pair_detail::RadialCache<N>>(p);scalar_harmonics=(*shared_radial)->scalar_harmonics(R);}
     const auto hint=lower_motion&&upper_motion?quartic_arc_geometry(ray,R,p,lo,hi,work,midpoint,scalar_coefficients,scalar_harmonics?&*scalar_harmonics:nullptr):std::nullopt;if(hint){if(hint->minima.empty()){retained.reset();return std::nullopt;}retained=*hint;}else
     retained=complex_arc_geometry(ray,R,p,lo,hi,physical_ends,work,16,12,false,stationary);
    }
    if(!retained || !retained->valid)return std::nullopt;
    {LCBININT_CHART_SCOPE(coefficients);if(shared_radial && !*shared_radial)*shared_radial=std::make_unique<root_pair_detail::RadialCache<N>>(p);}
    const auto* radial=shared_radial?shared_radial->get():nullptr;
    PairIntegral<N> result;const double pi=std::acos(-1.);
    for(std::size_t j=0;j<retained->minima.size();++j) {
        auto& minimum=retained->minima[j];root_pair_detail::Arc<0> canonical;
        std::optional<root_pair_detail::Arc<0>> model;{LCBININT_CHART_SCOPE(coefficients);if(!canonical.configure(R,p,minimum.center,{},scalar_coefficients,nullptr))return std::nullopt;
        model=root_pair_detail::complex_model(canonical,R,p,minimum.center,minimum.scale,minimum.phi,minimum.error,**scalar_coefficients,work);}
        if(!model){if(retained->algebraic_hint)retained=complex_arc_geometry(ray,R,p,lo,hi,physical_ends,work,16,12,false);return std::nullopt;}
        const double lower=j?(retained->minima[j-1].center+minimum.center)*.5:lo;
        const double upper=j+1<retained->minima.size()?(minimum.center+retained->minima[j+1].center)*.5:hi;
        const int pieces=std::max(1,int(std::ceil((upper-lower)/(pi*.5))));
        for(int piece=0;piece<pieces;++piece) {
            const double a=lower+(upper-lower)*piece/pieces,b=lower+(upper-lower)*(piece+1)/pieces;
            const double c=minimum.center+2*pi*std::round(((a+b)*.5-minimum.center)/(2*pi));
            double basis=c;
            if(std::max(std::abs(a-c),std::abs(b-c))>pi*.75)basis=(std::min(a,c)+std::max(b,c))*.5;
            root_pair_detail::Arc<0> active;
            {LCBININT_CHART_SCOPE(coefficients);if(!move_pair<0>(*model,basis,R,p,{},scalar_coefficients,nullptr,active))return std::nullopt;}
            const double tl=std::tan((a-basis)*.5),th=std::tan((b-basis)*.5);
            const double m=active.pair.mean.value,h=active.half_width;
            const double ulo=std::asinh((tl-m)/h),uhi=std::asinh((th-m)/h);
            const bool moving=true;
            if(!(uhi>ulo) || !std::isfinite(uhi-ulo) || uhi-ulo>64)return std::nullopt;
            const int cells=std::max(1,int(std::ceil((uhi-ulo)/
            1.
            )));
            LCBININT_CHART_SCOPE(moments);
            std::array<double,16> piece_moments{};
            const ShiftedHyperbolicNodes shifted_nodes(gauss_rule(quadrature_order),cells>=3?(uhi-ulo)/(2*cells):0);
            bool global_completed=false;
#ifdef LCBININT_TRIPLE_GLOBAL_ANGULAR_STREAM
            if(moving){double global_error=0;const auto global=global_angular_moments(active,ulo,uhi,physical_ends&&j==0&&piece==0,physical_ends&&j+1==retained->minima.size()&&piece+1==pieces,work,value_gate,global_error);if(global){result.error+=global_error;piece_moments=*global;global_completed=true;}}
#endif
            for(int cell=0;!global_completed&&cell<cells;) {
                int next=cell+1;
#if defined(LCBININT_TRIPLE_ANALYTIC_ANGULAR_CELLS) || defined(LCBININT_TRIPLE_BERNSTEIN_ANGULAR_CELLS)
                const bool touches_left=physical_ends&&j==0&&piece==0&&cell==0;
                const bool touches_right=physical_ends&&j+1==retained->minima.size()&&piece+1==pieces&&cell+2==cells;
                if(moving&&!touches_left&&!touches_right&&cell+2<=cells){const double middle=ulo+(uhi-ulo)*(cell+1)/cells,half=(uhi-ulo)/cells;
                    const double center_t=active.pair.mean.value+active.half_width*std::sinh(middle),radius_t=active.half_width*std::exp(std::abs(middle))*std::expm1(
#ifdef LCBININT_TRIPLE_BERNSTEIN_ANGULAR_CELLS
                    2.125*half
#else
                    3*half
#endif
                    );
                    if(active.analytic_disk(center_t,radius_t))next=cell+2;
                }
#endif
                const double fraction_lo=double(cell)/cells,fraction_width=double(next-cell)/cells;cell=next;
                const FoldRadialMap map{ulo+(uhi-ulo)*fraction_lo,ulo+(uhi-ulo)*(fraction_lo+fraction_width),
                    physical_ends&&j==0&&piece==0&&fraction_lo==0,
                    physical_ends&&j+1==retained->minima.size()&&piece+1==pieces&&cell==cells};
                const auto& rule=gauss_rule(map.left || map.right ?
#ifdef LCBININT_TRIPLE_COMPACT_CRITICAL_ENDS
                    std::max(8,quadrature_order)
#else
                    std::max(
                    16
                    ,2*quadrature_order)
#endif
 :
#ifdef LCBININT_TRIPLE_BERNSTEIN_ANGULAR_CELLS
                    (fraction_width>1.5/cells?10:quadrature_order)
#else
                    quadrature_order
#endif
                );
                if(moving){
                    // Integrate the local coefficient sensitivities with the
                    // flux. Pull back the shared coefficient packet once per
                    // cell instead of evaluating eight polynomial RHS/node.
                    std::array<double,16> moments{};const double span=uhi-ulo;
                    const FoldRadialMap normalized{0,1,map.left,map.right};
                    const double v=active.pair.square.value,h=active.half_width;
                    const auto& q=active.pair.quotient;
                    const bool shifted=cells>=3&&!map.left&&!map.right&&fraction_width==1./cells&&rule.x.size()<=128;
                    const auto center=shifted?ShiftedHyperbolicNodes::evaluate(ulo+span*(fraction_lo+fraction_width*.5)):std::array<double,2>{};
                    for(std::size_t k=0;k<rule.x.size();++k){const auto node=normalized(rule.x[k]);const double alpha=fraction_lo+fraction_width*node[0],u=ulo+span*alpha;double sh,ch;
                        if(shifted){const auto& q=shifted_nodes.offsets[k];sh=center[0]*q[1]+center[1]*q[0];ch=center[1]*q[1]+center[0]*q[0];}else
                        if(!map.left&&!map.right&&std::abs(u)<=64){const double e=std::expm1(std::abs(u)),inverse=1/(1+e);sh=std::copysign(.5*e*(1+inverse),u);ch=1+.5*e*(e*inverse);}else
                        {sh=std::sinh(u);ch=std::cosh(u);}const double t=active.pair.mean.value+h*sh,one=1+t*t;
                        double Q=q[4].value;for(int d=3;d>=0;--d)Q=Q*t+q[d].value;
                        const double Qt=((4*q[4].value*t+3*q[3].value)*t+2*q[2].value)*t+q[1].value;
                        std::array<double,2> pole{},inverse_pole{};double denominator_slope=6*t/one;
                        for(int d=0;d<2;++d){const auto& p=active.poles[d];pole[d]=(p[2].value*t+p[1].value)*t+p[0].value;inverse_pole[d]=1/pole[d];denominator_slope+=(2*p[2].value*t+p[1].value)*inverse_pole[d];}
                        const double ratio=Q/(one*one*one*pole[0]*pole[1]);if(!(ratio>0)||!std::isfinite(ratio))return std::nullopt;
                        const double density=2*active.inverse_rho*(-v)*ch*ch*std::sqrt(ratio),weight=rule.w[k]*span*fraction_width*node[1]*density;

                        const double inverse_q=std::abs(Q)>=std::numeric_limits<double>::min()?1/Q:0;
                        const double slope=.5*((inverse_q?Qt*inverse_q:Qt/Q)-denominator_slope),coefficient_weight=inverse_q?.5*weight*inverse_q:.5*weight/Q,uslope=
                        2*sh/ch+slope*h*ch;
                        moments[0]+=weight;moments[1]+=weight*slope;moments[2]+=weight*(1/v-slope*sh/(2*h));
                        double power=1;for(int d=0;d<5;++d){moments[3+d]+=
                        coefficient_weight*power;
                        power*=t;}
                        for(int d=0;d<2;++d){double power=1;for(int e=0;e<3;++e){moments[8+3*d+e]-=.5*weight*power*inverse_pole[d];power*=t;}}
                        moments[14]+=weight*(uslope*(1-alpha)-1/span);moments[15]+=weight*(uslope*alpha+1/span);++work.evaluations;
                    }
                    for(int k=0;k<16;++k)piece_moments[k]+=moments[k];continue;
                }
            }
            const double shl=std::sinh(ulo),chl=std::cosh(ulo),shh=std::sinh(uhi),chh=std::cosh(uhi),v=active.pair.square.value;
            const double am=piece_moments[1]-piece_moments[14]/(h*chl)-piece_moments[15]/(h*chh);
            const double av=piece_moments[2]-piece_moments[14]*shl/(2*v*chl)-piece_moments[15]*shh/(2*v*chh);
            std::array<double,5> aq;for(int d=0;d<5;++d)aq[d]=piece_moments[3+d];const auto ap=root_pair_detail::factorization_adjoint(active,am,av,aq);if(!ap)return std::nullopt;
            if(!coefficients)coefficients.emplace(R,p,motion,radial);std::array<std::array<double,3>,2> pole_adjoint;for(int d=0;d<2;++d)for(int e=0;e<3;++e)pole_adjoint[d][e]=piece_moments[8+3*d+e];
            const auto canonical=root_pair_detail::coefficient_adjoint(*coefficients,basis,*ap,pole_adjoint,-piece_moments[0]*active.inverse_rho);
            result.flux+=piece_moments[0];for(int k=0;k<N;++k){double partial=canonical[k];
             if(j==0&&piece==0)partial+=piece_moments[14]*(1+tl*tl)/(2*h*chl)*(*lower_motion)[k];
             if(j+1==retained->minima.size()&&piece+1==pieces)partial+=piece_moments[15]*(1+th*th)/(2*h*chh)*(*upper_motion)[k];
             if(!std::isfinite(partial))return std::nullopt;result.gradient[k]+=partial;
            }

        }
    }
    result.error=std::max(result.error,1e-6*std::abs(result.flux));return result;
}
#endif
template<int N,class Ray,class Work> std::optional<PairIntegral<N>> critical_arc(
        const Ray& ray,double R,const Geometry& p,double lo,double hi,bool physical_ends,
        const std::array<double,N>& motion,Work& work,
        std::optional<root_pair_detail::Coefficients<N>>& coefficients,
        std::unique_ptr<root_pair_detail::RadialCache<N>>* shared_radial,
        std::optional<AngularChart>& retained,int quadrature_order=8,const std::array<double,N>* lower_motion=nullptr,const std::array<double,N>* upper_motion=nullptr,const AngularStationarity* stationary=nullptr,const typename Ray::Core* midpoint=nullptr,std::optional<root_pair_detail::Coefficients<0>>* scalar_coefficients=nullptr,double value_gate=1e-6) {
#ifdef LCBININT_TRIPLE_REVERSE_ANGULAR
    if constexpr(N==8)if(lower_motion&&upper_motion){const auto reverse=critical_arc_reverse<N>(ray,R,p,lo,hi,physical_ends,motion,work,coefficients,shared_radial,retained,quadrature_order,lower_motion,upper_motion,stationary,midpoint,scalar_coefficients,value_gate);if(reverse)return reverse;}
#endif
    LCBININT_COUPLED_SCOPE(chart);
#if defined(LCBININT_TRIPLE_GLOBAL_ANGULAR_STREAM) || defined(LCBININT_TRIPLE_COFACTOR_CORE) || defined(LCBININT_TRIPLE_COMPACT_CRITICAL_TRANSPORT) || defined(LCBININT_TRIPLE_HYBRID_COMPACT_ANGLE) || defined(LCBININT_TRIPLE_ENDPOINT_COFACTOR) || defined(LCBININT_TRIPLE_REVERSE_ANGULAR)
    constexpr bool physical_transport=false;
#else
    // The LD intensity vanishes at each physical angular end. Integrate
    // its derivative at fixed angle; internal piece boundaries cancel.
    // This avoids differentiating an arbitrary quadrature coordinate map.
    constexpr bool physical_transport=true;
#endif
    {LCBININT_CHART_SCOPE(discovery);
     std::optional<std::array<root_pair_detail::Jet<0>,7>> scalar_harmonics;
     if(lower_motion&&upper_motion&&midpoint&&midpoint->phi>0&&midpoint->roundoff<1e-7*midpoint->phi&&scalar_coefficients&&!*scalar_coefficients&&shared_radial){if(!*shared_radial)*shared_radial=std::make_unique<root_pair_detail::RadialCache<N>>(p);scalar_harmonics=(*shared_radial)->scalar_harmonics(R);}
     const auto hint=lower_motion&&upper_motion?quartic_arc_geometry(ray,R,p,lo,hi,work,midpoint,scalar_coefficients,scalar_harmonics?&*scalar_harmonics:nullptr):std::nullopt;if(hint){if(hint->minima.empty()){retained.reset();return std::nullopt;}retained=*hint;}else
     retained=complex_arc_geometry(ray,R,p,lo,hi,physical_ends,work,16,12,false,stationary);
    }
    if(!retained || !retained->valid)return std::nullopt;
    {LCBININT_CHART_SCOPE(coefficients);if(shared_radial && !*shared_radial)*shared_radial=std::make_unique<root_pair_detail::RadialCache<N>>(p);}
    const auto* radial=shared_radial?shared_radial->get():nullptr;
    PairIntegral<N> result;const double pi=std::acos(-1.);
    for(std::size_t j=0;j<retained->minima.size();++j) {
        auto& minimum=retained->minima[j];root_pair_detail::Arc<N> canonical;
        std::optional<root_pair_detail::Arc<N>> model;{LCBININT_CHART_SCOPE(coefficients);if(!canonical.configure(R,p,minimum.center,motion,&coefficients,radial))return std::nullopt;
        model=root_pair_detail::complex_model(canonical,R,p,minimum.center,minimum.scale,minimum.phi,minimum.error,*coefficients,work);}
        if(!model){if(retained->algebraic_hint)retained=complex_arc_geometry(ray,R,p,lo,hi,physical_ends,work,16,12,false);return std::nullopt;}
        const double lower=j?(retained->minima[j-1].center+minimum.center)*.5:lo;
        const double upper=j+1<retained->minima.size()?(minimum.center+retained->minima[j+1].center)*.5:hi;
        const int pieces=std::max(1,int(std::ceil((upper-lower)/(pi*.5))));
        for(int piece=0;piece<pieces;++piece) {
            const double a=lower+(upper-lower)*piece/pieces,b=lower+(upper-lower)*(piece+1)/pieces;
            const double c=minimum.center+2*pi*std::round(((a+b)*.5-minimum.center)/(2*pi));
            double basis=c;
            if(std::max(std::abs(a-c),std::abs(b-c))>pi*.75)basis=(std::min(a,c)+std::max(b,c))*.5;
            root_pair_detail::Arc<N> active;
            {LCBININT_CHART_SCOPE(coefficients);if(!move_pair<N>(*model,basis,R,p,motion,&coefficients,radial,active))return std::nullopt;}
            const double tl=std::tan((a-basis)*.5),th=std::tan((b-basis)*.5);
            const double m=active.pair.mean.value,h=active.half_width;
            const double ulo=std::asinh((tl-m)/h),uhi=std::asinh((th-m)/h);
            root_pair_detail::Jet<N> moving_lo(ulo),moving_hi(uhi);
            const bool moving=lower_motion&&upper_motion;
            if(moving&&!physical_transport){std::array<double,N> zero{};moving_lo=active.critical_coordinate(a,j==0&&piece==0?*lower_motion:zero);moving_hi=active.critical_coordinate(b,j+1==retained->minima.size()&&piece+1==pieces?*upper_motion:zero);}
            if(!(uhi>ulo) || !std::isfinite(uhi-ulo) || uhi-ulo>64)return std::nullopt;
#ifdef LCBININT_TRIPLE_COMPACT_CRITICAL_TRANSPORT
            if(moving)if(const auto compact=compact_critical_transport(active,moving_lo,moving_hi,physical_ends&&j==0&&piece==0,physical_ends&&j+1==retained->minima.size()&&piece+1==pieces,work,std::min(1e-6,value_gate*.1))){result.flux+=compact->flux;result.error+=compact->error;for(int k=0;k<N;++k)result.gradient[k]+=compact->gradient[k];continue;}
#endif
#ifdef LCBININT_TRIPLE_ENDPOINT_COFACTOR
            if(moving)if(const auto deflated=endpoint_cofactor_transport(active,moving_lo,moving_hi,physical_ends&&j==0&&piece==0,physical_ends&&j+1==retained->minima.size()&&piece+1==pieces,work,8)){result.flux+=deflated->flux;for(int k=0;k<N;++k)result.gradient[k]+=deflated->gradient[k];continue;}
#endif
            const int cells=std::max(1,int(std::ceil((uhi-ulo)/
            1.
            )));
            LCBININT_CHART_SCOPE(moments);
            std::array<double,16> piece_moments{};
            const ShiftedHyperbolicNodes shifted_nodes(gauss_rule(quadrature_order),cells>=3?(uhi-ulo)/(2*cells):0);
#ifdef LCBININT_TRIPLE_COFACTOR_CORE
            std::optional<root_pair_detail::CofactorCoreAdjoint<N>> analytic_core;
            if(moving&&uhi-ulo>4){analytic_core.emplace(active);if(!analytic_core->valid)analytic_core.reset();}
#endif
            bool global_completed=false;
#ifdef LCBININT_TRIPLE_GLOBAL_ANGULAR_STREAM
            if(moving){double global_error=0;const auto global=global_angular_moments(active,ulo,uhi,physical_ends&&j==0&&piece==0,physical_ends&&j+1==retained->minima.size()&&piece+1==pieces,work,value_gate,global_error);if(global){result.error+=global_error;piece_moments=*global;global_completed=true;}}
#endif
            for(int cell=0;!global_completed&&cell<cells;) {
                int next=cell+1;
#if defined(LCBININT_TRIPLE_COMPACT_CRITICAL_TRANSPORT) && defined(LCBININT_TRIPLE_HYBRID_COMPACT_ANGLE)
                if(moving){const double limit=std::acosh(16.),span=uhi-ulo,ua=ulo+span*double(cell)/cells;int end=cell;
                 if(std::abs(ua)<limit)while(end<cells&&std::abs(ulo+span*double(end+1)/cells)<limit)++end;
                 if(end>=cell+2){using J=root_pair_detail::Jet<N>;const J a=moving_lo+(moving_hi-moving_lo)*J(double(cell)/cells),b=moving_lo+(moving_hi-moving_lo)*J(double(end)/cells);
                  if(const auto compact=compact_critical_transport(active,a,b,physical_ends&&j==0&&piece==0&&cell==0,physical_ends&&j+1==retained->minima.size()&&piece+1==pieces&&end==cells,work,std::min(1e-6,value_gate*.1))){result.flux+=compact->flux;result.error+=compact->error;for(int k=0;k<N;++k)result.gradient[k]+=compact->gradient[k];cell=end;continue;}
                 }
                }
#endif
#ifdef LCBININT_TRIPLE_COFACTOR_CORE
                if(analytic_core){const double span=uhi-ulo,r=analytic_core->series.radius,m=active.pair.mean.value,h=active.half_width,fa=double(cell)/cells,ua=ulo+span*fa,ta=m+h*std::sinh(ua);int end=cell;
                 if(std::abs(ta-m)<.75*r){while(end<cells&&std::abs(h*std::sinh(ulo+span*double(end+1)/cells))<.75*r)++end;}
                 if(end>=cell+2){const double fb=double(end)/cells,ub=ulo+span*fb,tb=m+h*std::sinh(ub);if(const auto integral=analytic_core->integrate(ta,tb)){
                  const double density_a=analytic_core->boundary_density(ta),density_b=analytic_core->boundary_density(tb),sha=std::sinh(ua),cha=std::cosh(ua),shb=std::sinh(ub),chb=std::cosh(ub);
                  result.flux+=integral->value;result.error+=1e-8*std::abs(integral->value);
                  for(int k=0;k<N;++k){const double upa=(1-fa)*moving_lo.dp[k]+fa*moving_hi.dp[k],upb=(1-fb)*moving_lo.dp[k]+fb*moving_hi.dp[k],tp_a=active.pair.mean.dp[k]-sha*active.pair.square.dp[k]/(2*h)+h*cha*upa,tp_b=active.pair.mean.dp[k]-shb*active.pair.square.dp[k]/(2*h)+h*chb*upb;result.gradient[k]+=integral->dp[k]+density_b*tp_b-density_a*tp_a;}
                  ++work.evaluations;cell=end;continue;
                 }}
                }
#endif
#if defined(LCBININT_TRIPLE_ANALYTIC_ANGULAR_CELLS) || defined(LCBININT_TRIPLE_BERNSTEIN_ANGULAR_CELLS)
                const bool touches_left=physical_ends&&j==0&&piece==0&&cell==0;
                const bool touches_right=physical_ends&&j+1==retained->minima.size()&&piece+1==pieces&&cell+2==cells;
                if(moving&&!touches_left&&!touches_right&&cell+2<=cells){const double middle=ulo+(uhi-ulo)*(cell+1)/cells,half=(uhi-ulo)/cells;
                    const double center_t=active.pair.mean.value+active.half_width*std::sinh(middle),radius_t=active.half_width*std::exp(std::abs(middle))*std::expm1(
#ifdef LCBININT_TRIPLE_BERNSTEIN_ANGULAR_CELLS
                    2.125*half
#else
                    3*half
#endif
                    );
                    if(active.analytic_disk(center_t,radius_t))next=cell+2;
                }
#endif
                const double fraction_lo=double(cell)/cells,fraction_width=double(next-cell)/cells;cell=next;
                const FoldRadialMap map{ulo+(uhi-ulo)*fraction_lo,ulo+(uhi-ulo)*(fraction_lo+fraction_width),
                    physical_ends&&j==0&&piece==0&&fraction_lo==0,
                    physical_ends&&j+1==retained->minima.size()&&piece+1==pieces&&cell==cells};
                const auto& rule=gauss_rule(map.left || map.right ?
#ifdef LCBININT_TRIPLE_COMPACT_CRITICAL_ENDS
                    std::max(8,quadrature_order)
#else
                    std::max(
                    physical_transport&&moving?12:16
                    ,2*quadrature_order)
#endif
 :
#ifdef LCBININT_TRIPLE_BERNSTEIN_ANGULAR_CELLS
                    (fraction_width>1.5/cells?10:quadrature_order)
#else
                    quadrature_order
#endif
                );
                if(moving){
                    // Integrate the local coefficient sensitivities with the
                    // flux. Pull back the shared coefficient packet once per
                    // cell instead of evaluating eight polynomial RHS/node.
                    std::array<double,16> moments{};const double span=uhi-ulo;
                    const FoldRadialMap normalized{0,1,map.left,map.right};
                    const double v=active.pair.square.value,h=active.half_width;
                    const auto& q=active.pair.quotient;
                    const bool shifted=cells>=3&&!map.left&&!map.right&&fraction_width==1./cells&&rule.x.size()<=128;
                    const auto center=shifted?ShiftedHyperbolicNodes::evaluate(ulo+span*(fraction_lo+fraction_width*.5)):std::array<double,2>{};
                    for(std::size_t k=0;k<rule.x.size();++k){const auto node=normalized(rule.x[k]);const double alpha=fraction_lo+fraction_width*node[0],u=ulo+span*alpha;double sh,ch;
                        if(shifted){const auto& q=shifted_nodes.offsets[k];sh=center[0]*q[1]+center[1]*q[0];ch=center[1]*q[1]+center[0]*q[0];}else
                        if(!map.left&&!map.right&&std::abs(u)<=64){const double e=std::expm1(std::abs(u)),inverse=1/(1+e);sh=std::copysign(.5*e*(1+inverse),u);ch=1+.5*e*(e*inverse);}else
                        {sh=std::sinh(u);ch=std::cosh(u);}const double t=active.pair.mean.value+h*sh,one=1+t*t;
                        double Q=q[4].value;for(int d=3;d>=0;--d)Q=Q*t+q[d].value;
                        const double Qt=((4*q[4].value*t+3*q[3].value)*t+2*q[2].value)*t+q[1].value;
                        std::array<double,2> pole{},inverse_pole{};double denominator_slope=6*t/one;
                        for(int d=0;d<2;++d){const auto& p=active.poles[d];pole[d]=(p[2].value*t+p[1].value)*t+p[0].value;inverse_pole[d]=1/pole[d];denominator_slope+=(2*p[2].value*t+p[1].value)*inverse_pole[d];}
                        const double ratio=Q/(one*one*one*pole[0]*pole[1]);if(!(ratio>0)||!std::isfinite(ratio))return std::nullopt;
                        const double density=2*active.inverse_rho*(-v)*ch*ch*std::sqrt(ratio),weight=rule.w[k]*span*fraction_width*node[1]*density;

                        const double inverse_q=std::abs(Q)>=std::numeric_limits<double>::min()?1/Q:0;
                        const double slope=.5*((inverse_q?Qt*inverse_q:Qt/Q)-denominator_slope),coefficient_weight=inverse_q?.5*weight*inverse_q:.5*weight/Q,uslope=
                        2*sh/ch+slope*h*ch;
                        moments[0]+=weight;
                        if constexpr(physical_transport){
                            const double pair=(t-active.pair.mean.value)*(t-active.pair.mean.value)-v;
                            moments[1]-=weight*(t-active.pair.mean.value)/pair;
                            moments[2]-=.5*weight/pair;
                        }else{moments[1]+=weight*slope;moments[2]+=weight*(1/v-slope*sh/(2*h));}
                        double power=1;for(int d=0;d<5;++d){moments[3+d]+=
                        coefficient_weight*power;
                        power*=t;}
                        for(int d=0;d<2;++d){double power=1;for(int e=0;e<3;++e){moments[8+3*d+e]-=.5*weight*power*inverse_pole[d];power*=t;}}
                        if constexpr(!physical_transport){moments[14]+=weight*(uslope*(1-alpha)-1/span);moments[15]+=weight*(uslope*alpha+1/span);}++work.evaluations;
                    }
                    for(int k=0;k<16;++k)piece_moments[k]+=moments[k];continue;
                }
                for(std::size_t k=0;k<rule.x.size();++k) {
                    const auto u=map(rule.x[k]);const double delta=h*std::sinh(u[0]);
                    root_pair_detail::Jet<N> value;
                    if(!active.critical_density(delta,delta*delta-active.pair.square.value,value))return std::nullopt;
                    ++work.evaluations;const double weight=rule.w[k]*u[1];result.flux+=weight*value.value;
                    for(int i=0;i<N;++i)result.gradient[i]+=weight*value.dp[i];
                }
            }
            if(moving){const auto& q=active.pair.quotient;
                    result.flux+=piece_moments[0];for(int k=0;k<N;++k){double partial=-piece_moments[0]*active.rho.dp[k]*active.inverse_rho+piece_moments[1]*active.pair.mean.dp[k]+piece_moments[2]*active.pair.square.dp[k];
                        if constexpr(!physical_transport)partial+=piece_moments[14]*moving_lo.dp[k]+piece_moments[15]*moving_hi.dp[k];
                        for(int d=0;d<5;++d)partial+=piece_moments[3+d]*q[d].dp[k];for(int d=0;d<2;++d)for(int e=0;e<3;++e)partial+=piece_moments[8+3*d+e]*active.poles[d][e].dp[k];
                        if(!std::isfinite(partial))return std::nullopt;result.gradient[k]+=partial;
                    }
            }
        }
    }
    result.error=std::max(result.error,1e-6*std::abs(result.flux));return result;
}
} // namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail
