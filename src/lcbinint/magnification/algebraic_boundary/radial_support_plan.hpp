#pragma once

#include "radial_support_seeds.hpp"
#include "bounded_radial_roots.hpp"
#include "radial_support_plan_types.hpp"

namespace lcbinint::algebraic_boundary::exact_support_detail {

// Rees classification with the discriminant sign inherited from radial
// isolation. No angular root coordinates or quartic discriminant are needed.
inline int exact_angular_count(std::array<Dyadic,5> p, int discriminant_sign) {
    if(p[4].n == 0) {
        p = {p[4], -p[3], p[2], -p[1], p[0]};
        if(p[4].n == 0) return -1;
    }
    if(discriminant_sign < 0) return 2;
    if(discriminant_sign == 0) return -1;
    const auto a=p[4], b=p[3], c=p[2], d=p[1], e=p[0];
    const auto a2=a*a, b2=b*b, c2=c*c;
    const auto P=Dyadic(8)*a*c-Dyadic(3)*b2;
    if(P.n > 0) return 0;
    const auto D=Dyadic(64)*a2*a*e-Dyadic(16)*a2*c2+
        Dyadic(16)*a*b2*c-Dyadic(16)*a2*b*d-Dyadic(3)*b2*b2;
    return P.n < 0 && D.n < 0 ? 4 : 0;
}

inline RadialSupportPlan radial_plan(const PrimaryFrame& pf) {
    RadialSupportPlan out;
    // Reflection symmetry can give coincident radial events; keep the phase
    // enclosure route for exactly axial sources.
    if(pf.Y==0)return out;
    auto roots=radial_support_roots(pf);
    if(!roots.completed || roots.boxes.empty()) return out;
    // A one-variation box contains a simple root. Endpoint atoms need a
    // separate multiplicity argument and retain the physical-seed fallback.
    for(const auto& root:roots.boxes)
        if(root.lo == root.hi || root.lower_sign == 0 || root.upper_sign == 0) return out;
    const int power=radial_domain_power(pf);
    if(power < 0) return out;

    struct Cell {double lo,hi,witness,left_witness,right_witness; bool occupied;};
    std::vector<Cell> cells;
    auto sample=[&](double lo, double hi, int discriminant_sign) {
        if(!(hi > lo)) return false;
        const double R=std::sqrt(.5*(lo+hi));
        const __float128 square=(__float128)R*R;
        if(!(square > lo && square < hi) || R == pf.a) return false;
        int crossings=2;
        bool occupied=true;
        if(discriminant_sign > 0) {
            const auto p=angular_coefficients(R,pf);
            crossings=exact_angular_count(p,discriminant_sign);
            if(crossings < 0) return false;
            occupied=crossings > 0 || p[0].n > 0;
        }
        const auto interior=[&](double candidate) {
            const __float128 square=(__float128)candidate*candidate;
            return square>lo&&square<hi&&candidate!=pf.a;
        };
        double left_witness=std::nextafter(std::sqrt(lo),INFINITY);
        double right_witness=std::nextafter(std::sqrt(hi),0.0);
        if(lo==0)left_witness=R;
        if(!interior(left_witness))left_witness=R;
        if(!interior(right_witness))right_witness=R;
        cells.push_back({lo,hi,R,left_witness,right_witness,occupied});
        for(double witness:{left_witness,R,right_witness}) {
            if(!out.radii.empty()&&out.radii.back()==witness)continue;
            out.radii.push_back(witness);
            out.crossings.push_back(occupied && crossings == 0 ? -2 : crossings);
        }
        return true;
    };
    // Bernstein endpoint coefficients have the polynomial's exact sign.
    // A root-free gap inherits that sign throughout, including R*R.
    double left=0;
    for(const auto& root:roots.boxes) {
        if(!sample(left,root.lo,root.lower_sign)) return out;
        left=root.hi;
    }
    if(!sample(left,std::ldexp(1.,power),
               roots.boxes.back().upper_sign)) return out;

    for(std::size_t i=0;i<cells.size();++i) {
        if(!cells[i].occupied) continue;
        const auto begin=i;
        while(i+1 < cells.size() && cells[i+1].occupied) ++i;
        if(begin == 0 || i+1 == cells.size()) return out;
        out.brackets.push_back({cells[begin].witness,
            cells[i].witness, cells[begin-1].right_witness,
            cells[i+1].left_witness});
    }
    out.completed=!out.brackets.empty();
    return out;
}

} // namespace lcbinint::algebraic_boundary::exact_support_detail
