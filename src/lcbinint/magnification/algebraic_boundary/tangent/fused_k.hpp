#pragma once
#include "fused_motion.hpp"
#include "fused_geometry_screen.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/root_pair.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/angular_transport.hpp"

namespace lcbinint::algebraic_boundary::tangent::fused_gradient {
// Five RHS share the same root-pair Jacobian and singularity check.
inline std::array<RootPairDR,5> root_pair_tangents(const RootPair& pair,
    const std::array<double,5>& pc,const std::array<std::array<double,5>,5>& dp) {
    std::array<RootPairDR,5> out{};
    const auto J=eo_jacobian(pair,pc);
    const double det=J.E_m*J.O_v-J.E_v*J.O_m;
    if(std::fabs(det)<kRootPairDetFloor || !std::isfinite(det))return out;
    for(int j=0;j<5;++j) {
        const auto d=p_derivs(dp[j],pair.m);
        const double E=d.P+0.5*pair.v*d.P2+pair.v*pair.v*dp[j][4];
        const double O=d.P1+(pair.v/6.0)*d.P3;
        out[j]={-(J.O_v*E-J.E_v*O)/det,-(-J.O_m*E+J.E_m*O)/det,true};
    }
    return out;
}
// The 1/3 embedded difference can vanish for a nonconstant integrand.
// Certify only the cheap three-node shortcut with a Cauchy/Taylor tail bound.
// On |x|<=r each quadratic factor is p0*(1+e), |e|<=1/2.
// GC2(3) is exact through degree five, so its error is bounded by
// pi*M*r^-6/(1-r^-1), where M bounds the analytic integrand on that circle.
// This controls value quadrature only; it is not a gradient-error criterion.
inline double short_angular_value_bound(const DeflatedQuad& q,double m,double s,
    double v,double b0,double b2,double rho) {
    namespace screen=geometry_experiment;
    using I=screen::ScreenInterval;
    if(!positive_detail::environment_ok())return INFINITY;
    const I mm(m),ss(s),square_s=ss*ss,one(1),two(2);
    const I S0=-(I(q.d0)+mm*(I(q.d1)+mm*I(q.p4)));
    const I S1=-ss*(I(q.d1)+two*mm*I(q.p4)),S2=-square_s*I(q.p4);
    const I A0=one+mm*mm,A1=two*mm*ss,A2=square_s;
    const I B0=I(b0)*I(b0)+I(b2)*mm*mm,B1=two*I(b2)*mm*ss,B2=I(b2)*square_s;
    auto up=screen::screen_up,down=screen::screen_down;
    auto radius=[&](I p0,I p1,I p2){
        if(!(p0.lo>0)||!std::isfinite(p0.hi)||!std::isfinite(screen::screen_upper(p1))||!std::isfinite(screen::screen_upper(p2)))return 0.;
        const double a=up(screen::screen_upper(p2)/p0.lo),b=up(screen::screen_upper(p1)/p0.lo);
        const double discriminant=up(up(b*b)+up(2*a));
        const double denominator=up(b+up(std::sqrt(discriminant)));
        return denominator>0&&std::isfinite(denominator)?down(1/denominator):0.;
    };
    double r=std::min({radius(S0,S1,S2),radius(A0,A1,A2),radius(B0,B1,B2)});
    if(!(r>1&&std::isfinite(r)))return INFINITY;
    int exponent=0;std::frexp(r,&exponent);const int power=exponent-1;
    if(power<=0)return INFINITY;
    r=std::ldexp(1.,power);
    const double ah=down(.5*A0.lo),bh=down(.5*B0.lo);
    const double denominator=down(down(ah*down(std::sqrt(ah)))*down(std::sqrt(bh)));
    if(!(denominator>0))return INFINITY;
    const double M=up(up(std::sqrt(up(1.5*S0.hi)))/denominator);
    const double tail=up(up(std::ldexp(1.,-6*power))/down(1-1/r));
    const double prefactor=up(up(2*v)/rho);
    const double bound=up(prefactor*up(up(kAngularPi)*up(M*tail)));
    return std::isfinite(bound)?bound:INFINITY;
}
struct AngularPair { VKJacobian high{}, low{}; bool ok=false; int points=0; };
// Value and five tangents share each point evaluation, square root and
// deflated polynomial. Nested upgrades evaluate only the newly inserted nodes.
inline AngularPair angular_pair(double m,double v,const std::array<double,5>& dm,
    const std::array<double,5>& dv,double R,const PrimaryFrame& pf,
    const std::array<double,5>& pc,const std::array<std::array<double,5>,5>& dpc,
    bool reciprocal,double value_gate,double gradient_gate) {
    AngularPair out;
    if(!std::isfinite(m)||!std::isfinite(v)||!(v>kAngularVFloor))return out;
    const double s=std::sqrt(v);
    if(std::fabs(m)+s>kAngularTMax)return out;
    const auto q=deflate_mv(pc,m,v);
    const double rm=R-pf.a,rp=R+pf.a,b0=reciprocal?rp:rm,b2=reciprocal?rm*rm:rp*rp;
    std::array<double,5> dp4{},dd1{},dd0{};
    for(int j=0;j<5;++j){dp4[j]=dpc[j][4];dd1[j]=dpc[j][3]+2*(dm[j]*q.p4+m*dp4[j]);
        dd0[j]=dpc[j][2]+2*(dm[j]*q.d1+m*dd1[j])-(2*m*dm[j]-dv[j])*q.p4-(m*m-v)*dp4[j];}
    std::array<NestedAngularPoint,31> points;
    int previous_order=0;VKJacobian previous{};
    const double prefactor=2/pf.rho;
    auto stage=[&](auto order){
        constexpr int N=decltype(order)::value;
        const auto& rule=gc2_rule_fixed<N>();
        if(previous_order)for(int i=previous_order-1;i>=0;--i)points[2*i+1]=points[i];
        for(int i=0;i<N;++i)if(!previous_order||(i&1)==0){
            ++out.points;
            if(!gc2_nested_jac_point(q,m,s,b0,b2,rm,rp,rule.x[i],dm,dv,dp4,dd1,dd0,reciprocal,points[i]))return -1;}
        auto sum=[&](auto n,bool low){
            constexpr int M=decltype(n)::value;const auto& weights=gc2_rule_fixed<M>();
            double G=0;std::array<double,5> dG{};
            for(int i=0;i<M;++i){const auto& point=points[low?2*i+1:i];G+=weights.w[i]*point.value;
                for(int j=0;j<5;++j)dG[j]+=weights.w[i]*point.derivative[j];}
            VKJacobian result{};result.vK=prefactor*v*G;result.ok=std::isfinite(result.vK);
            for(int j=0;j<5;++j){result.dvK[j]=prefactor*(dv[j]*G+v*dG[j]);
                if(j==2)result.dvK[j]-=prefactor*v*G/pf.rho;
                result.ok=result.ok&&std::isfinite(result.dvK[j]);}return result;};
        out.high=sum(order,false);
        out.low=previous_order?previous:sum(std::integral_constant<int,(N-1)/2>{},true);
        if(!out.high.ok||!out.low.ok)return -1;
        bool accepted=2*std::fabs(out.high.vK-out.low.vK)<=value_gate*std::fabs(out.high.vK);
        for(int j=0;j<5;++j)accepted=accepted&&
            2*std::fabs(out.high.dvK[j]-out.low.dvK[j])<=gradient_gate*std::max(1e-30,std::fabs(out.high.dvK[j]));
        if constexpr(N==3)if(accepted) {
            const double bound=short_angular_value_bound(q,m,s,v,b0,b2,pf.rho);
            accepted=bound<=geometry_experiment::screen_down(value_gate*std::fabs(out.high.vK));
        }
        previous=out.high;previous_order=N;
        return accepted?1:0;
    };
    int state=stage(std::integral_constant<int,3>{});
    if(state==0)state=stage(std::integral_constant<int,7>{});
    if(state==0)state=stage(std::integral_constant<int,15>{});
    if(state==0)state=stage(std::integral_constant<int,31>{});
    out.ok=state==1;return out;
}
}
namespace lcbinint::algebraic_boundary::tangent::fused_gradient {
struct MaterialMap {
    long double a=0,b=0;
    std::array<double,5> da{},db{};
    bool valid=false,derivatives_complete=false;
    Motion at(double R,double lo,double jac) const {
        Motion result;
        const long double width=b-a,x=((long double)R+lo-a)/width;
        for(int j=0;j<5;++j){result.radius[j]=double((1-x)*da[j]+x*db[j]);
            result.jacobian[j]=double(jac*(db[j]-da[j])/width);}
        return result;
    }
};
template<class Event>
inline MaterialMap make_material_map(const Event& left,const Event& right,const PrimaryFrame& pf, bool left_fixed=false, bool right_fixed=false) {
    MaterialMap map;map.a=(long double)left.radius+left.radius_lo;
    map.b=(long double)right.radius+right.radius_lo;
    if(!(map.b>map.a))return map;
    auto derivative=[&](const Event& event,std::array<double,5>& output){
        // The endpoint derivative only needs the stationary angular location.
        // `pair_motion_valid` additionally requires differentiating the local
        // root-pair coordinates, which can fail at a degenerate or ill-conditioned fold.
        // In that case the retained fold seed is still enough: phi_theta=0
        // removes the angular-motion term from d(phi(R,theta,p))/dp.
        double m=0.0;
        bool reciprocal=false;
        if(event.pair_motion_valid) {
            m=event.pair_m;
            reciprocal=event.pair_reciprocal;
        } else if(event.t_seed_valid && std::isfinite(event.t_seed)) {
            m=event.t_seed;
        } else if(event.u_seed_valid && std::isfinite(event.u_seed)) {
            m=event.u_seed;
            reciprocal=true;
        } else {
            return false;
        }
        const double theta=reciprocal?2*std::atan(m)+kPi:2*std::atan(m);
        const RadiusOffsetScope offset(event.radius_lo);
        const MotionScope fixed(nullptr);
        double angular,slope;
        const auto sample=fused_precise_phi(event.radius,theta,pf,&angular,&slope);
        if(!std::isfinite(slope)||slope==0)return false;
        if(std::fabs(sample.phi)>1e-3 || std::fabs(angular)>1e-5*std::max(1.,std::fabs(event.radius*slope)))return false;
        for(int j=0;j<5;++j){output[j]=-sample.dP[j]/slope;if(!std::isfinite(output[j]))return false;}
        return true;
    };
    const bool left_valid=left_fixed || derivative(left,map.da);
    const bool right_valid=right_fixed || derivative(right,map.db);
    if(!left_valid)map.da.fill(0.0);
    if(!right_valid)map.db.fill(0.0);
    map.valid=true;
    map.derivatives_complete=left_valid&&right_valid;
    return map;
}
}
