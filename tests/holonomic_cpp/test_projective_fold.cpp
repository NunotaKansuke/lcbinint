#include <iostream>
#include <cmath>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main() {
    int failures=0;
    auto check=[&](bool value,const char* why) {
        if(!value) { std::cerr<<"FAIL: "<<why<<'\n'; ++failures; }
    };
    check(adaptive_detail::projective_fold_contact_gate(
              (__float128)0,(__float128)0,(__float128)1)==
              adaptive_detail::ProjectiveFoldReject::DegenerateAngularContact,
          "high-order angular contact is rejected");
    check(adaptive_detail::projective_fold_contact_gate(
              (__float128)0,(__float128)1,(__float128)0)==
              adaptive_detail::ProjectiveFoldReject::NoRadialCrossing,
          "radial tangency is rejected");
    check(adaptive_detail::projective_fold_contact_gate(
              (__float128)1e-20,(__float128)1,(__float128)1)==
              adaptive_detail::ProjectiveFoldReject::ProjectiveContactUnresolved,
          "unresolved projective contact fails closed");

    // Verify the local reciprocal derivatives are the exact coefficient
    // reversal Q(u)=u^4 P(-1/u), not the original t-chart derivatives.
    const LensParams parity_params{0.21,-0.13,0.037,0.23,0.91,false};
    const auto parity_pf=PrimaryFrame::from(parity_params);
    const __float128 parity_R=(__float128)0.73;
    const auto t_local=local_fold_quantities<__float128>(parity_R,(__float128)0,
                                                         parity_pf,false);
    const auto u_local=local_fold_quantities<__float128>(parity_R,(__float128)0,
                                                         parity_pf,true);
    const __float128 qscale=fabsq(u_local.P)+fabsq(u_local.Pt)+
        fabsq(u_local.Ptt)+fabsq(u_local.Psss)+fabsq(u_local.Pssss)+1;
    const __float128 qtol=(__float128)64*FLT128_EPSILON*qscale;
    check(fabsq(u_local.P-t_local.Pssss/(__float128)24)<=qtol,
          "reciprocal Q(0) equals t-quartic leading coefficient p4");
    check(fabsq(u_local.Pt+t_local.Psss/(__float128)6)<=qtol,
          "reciprocal Q_u(0) equals -p3");
    check(fabsq(u_local.Ptt-t_local.Ptt)<=qtol,
          "reciprocal Q_uu(0) equals 2*p2");
    check(fabsq(u_local.Psss/(__float128)6+t_local.Pt)<=qtol,
          "reciprocal cubic coefficient equals -p1");
    check(fabsq(u_local.Pssss/(__float128)24-t_local.P)<=qtol,
          "reciprocal quartic coefficient equals p0");

    const LensParams caustic{0.12,0.0,0.02,0.5,1.0,false};
    const auto caustic_pf=PrimaryFrame::from(caustic);
    auto caustic_topology=classify_cells(caustic_pf,nullptr,nullptr,true);
    const auto original_events=caustic_topology.events.size();
    const auto original_cells=caustic_topology.cells.size();
    int chart_count=0,certified_count=0;
    for(const auto& event:caustic_topology.events) {
        if(event.kind!="chart_p4")continue;
        ++chart_count;
        const auto probe=adaptive_detail::probe_projective_p4_fold(
            event,caustic_topology.events,caustic_pf);
        if(probe.accepted()) {
            ++certified_count;
            check(std::isfinite(probe.radius_lo)&&std::isfinite(probe.uncertainty),
                  "accepted projective fold has split radius and finite uncertainty");
            check(probe.contact_relative<1e-28,
                  "accepted projective fold satisfies Q_u(0)=0");
            check(probe.angular_curvature_relative>1e-12,
                  "accepted projective fold has nondegenerate angular contact");
            check(probe.radial_crossing_relative>1e-12,
                  "accepted projective fold crosses in radius");
        }
    }
    check(chart_count==2,"caustic-cross control has two chart_p4 events");
    check(certified_count==2,"both theta=pi contacts are certified projective folds");
    adaptive_detail::certify_projective_p4_events(caustic_topology,caustic_pf);
    check(caustic_topology.events.size()==original_events&&
          caustic_topology.cells.size()==original_cells,
          "certification changes only event metadata, not topology structure");
    int promoted=0;
    for(const auto& event:caustic_topology.events)
        promoted+=event.projective_fold_certified&&event.kind=="chart_p4"&&
                  event.physically_real&&event.fold_u_seed_valid&&
                  event.fold_u_seed==0.0;
    check(promoted==2,"only certified chart events carry the reciprocal fold seed");

    // A generic chart crossing is the negative control: p4=0 alone must not
    // be enough to turn it into a fold or attach a radial fold map.
    const LensParams crossing{0.558659830418558,-0.15344854176328532,
        0.21691857933122094,0.00013864385201713654,
        0.15765128198672196,false};
    const auto crossing_pf=PrimaryFrame::from(crossing);
    const auto crossing_topology=classify_cells(crossing_pf,nullptr,nullptr,true);
    int crossing_charts=0,crossing_promotions=0;
    for(const auto& event:crossing_topology.events) {
        if(event.kind!="chart_p4")continue;
        ++crossing_charts;
        const auto probe=adaptive_detail::probe_projective_p4_fold(
            event,crossing_topology.events,crossing_pf);
        crossing_promotions+=probe.accepted();
        check(!probe.accepted()&&
              probe.reject==adaptive_detail::ProjectiveFoldReject::NoCoincidentD14Event,
              "rand006 chart crossing is rejected by the independent D14 coincidence gate");
        check(probe.contact_relative>1e-2,
              "rand006 chart crossing has nonzero reciprocal contact coefficient Q_u(0)");
    }
    check(crossing_charts>0,"negative control contains chart_p4 events");
    check(crossing_promotions==0,"pure chart crossing is never promoted");

    if(failures==0)std::cout<<"projective fold certificate tests passed\n";
    return failures?1:0;
}
