#pragma once
#include "fused_motion.hpp"
#include "lcbinint/magnification/holonomic/root_pair.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"

namespace lcbinint::holonomic::fused_gradient {
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
struct AngularPair { VKJacobian high{}, low{}; bool ok=false; int points=0; };
// Value and five tangents share each point evaluation, square root and
// deflated polynomial. Nested upgrades evaluate only the newly inserted nodes.
inline AngularPair angular_pair(double m,double v,const std::array<double,5>& dm,
    const std::array<double,5>& dv,double R,const PrimaryFrame& pf,
    const std::array<double,5>& pc,const std::array<std::array<double,5>,5>& dpc,
    bool reciprocal,double value_gate,double gradient_gate) {
    AngularPair out;
    if(!std::isfinite(m)||!std::isfinite(v)||!(v>kHoloVFloor))return out;
    const double s=std::sqrt(v);
    if(std::fabs(m)+s>kHoloTMax)return out;
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
namespace lcbinint::holonomic::fused_gradient {
struct MaterialMap {
    long double a=0,b=0;
    std::array<double,5> da{},db{};
    bool valid=false;
    Motion at(double R,double lo,double jac) const {
        Motion result;
        const long double width=b-a,x=((long double)R+lo-a)/width;
        for(int j=0;j<5;++j){result.radius[j]=double((1-x)*da[j]+x*db[j]);
            result.jacobian[j]=double(jac*(db[j]-da[j])/width);}
        return result;
    }
};
template<class Event>
inline MaterialMap make_material_map(const Event& left,const Event& right,const PrimaryFrame& pf) {
    MaterialMap map;map.a=(long double)left.radius+left.radius_lo;
    map.b=(long double)right.radius+right.radius_lo;
    if(!(map.b>map.a))return map;
    auto derivative=[&](const Event& event,std::array<double,5>& output){
        if(!event.pair_motion_valid)return false;
        const double theta=event.pair_reciprocal?2*std::atan(event.pair_m)+kPi:2*std::atan(event.pair_m);
        const auto c=phi_detail::core(event.radius,theta,pf);
        const auto gradient=phi_detail::dP_from(c,pf);
        using namespace phi_detail;
        const C radial=(c.z+C{pf.m0,0}*c.inv_zb+C{c.m1,0}*(c.inv_zba*c.inv_zba)*c.zb)/C{event.radius,0};
        const double slope=-2*(c.g.re*radial.re+c.g.im*radial.im)*c.inv_r2;
        if(!std::isfinite(slope)||slope==0)return false;
        const double angular=phi_detail::dphi_dtheta_from(c,pf);
        if(std::fabs(c.phi)>1e-3 || std::fabs(angular)>1e-5*std::max(1.,std::fabs(event.radius*slope)))return false;
        for(int j=0;j<5;++j){output[j]=-gradient[j]/slope;if(!std::isfinite(output[j]))return false;}
        return true;
    };
    map.valid=derivative(left,map.da)&&derivative(right,map.db);return map;
}
}
