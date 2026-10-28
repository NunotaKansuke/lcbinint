#pragma once
#include "lcbinint/magnification/algebraic_boundary/triple/root_pair.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/physical_jet.hpp"
#include <optional>
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
template<int N> Jet<N> operator/(const Jet<N>&a,const Jet<N>&b){return a*inverse(b);}
}
namespace lcbinint::algebraic_boundary::triple::fold_leading_detail {
using D=root_pair_detail::Jet<8>;using J=physical_detail::Jet<D>;using C=coefficient_detail::C<J>;
struct Model{D radius,coordinate,amplitude;};
inline J constant(D x){return J(x,D(0),D(0));}
inline physical_detail::Jet<D> equation(D radius,D coordinate,const Geometry&p,bool chart){
 std::array<D,8> x{D(p.X),D(p.Y),D(p.rho),D(p.m1),D(p.m2),D(p.a),D(p.bx),D(p.by)};for(int k=0;k<8;++k)x[k].dp[k]=1;
 const J R(radius,D(1),D(0)),t(coordinate,D(0),D(1)),one(1),ts=physical_detail::square_jet(t),inv=physical_detail::reciprocal_jet(one+ts),sgn(chart?-1:1);
 const C unit{sgn*(one-ts)*inv,sgn*J(2)*t*inv};const C z=C{R}*unit;
 const J radial=R-(one-constant(x[3])-constant(x[4]))*physical_detail::reciprocal_jet(R);
 C g=C{radial}*unit-C{constant(x[0]),constant(x[1])};
 const std::array<C,2> lens{C{constant(x[5])},C{constant(x[6]),constant(x[7])}};
 for(int k=0;k<2;++k){const C d=coefficient_detail::conj(z-lens[k]);const J norm=physical_detail::square_jet(d.re)+physical_detail::square_jet(d.im);g=g-C{constant(x[3+k])}*coefficient_detail::conj(d)*C{physical_detail::reciprocal_jet(norm)};}
 return constant(x[2])*constant(x[2])-physical_detail::square_jet(g.re)-physical_detail::square_jet(g.im);
}
inline std::optional<Model> make(const Geometry&p,double r,double t,bool chart){
 auto f=equation(D(r),D(t),p,chart);if(!std::isfinite(f.r.value)||!std::isfinite(f.tt.value)||std::abs(f.r.value)<1e-13||std::abs(f.tt.value)<1e-13)return std::nullopt;
 D R(r),T(t);for(int k=0;k<8;++k){R.dp[k]=-f.v.dp[k]/f.r.value;T.dp[k]=-(f.t.dp[k]+f.rt.value*R.dp[k])/f.tt.value;}
 auto total=equation(R,T,p,chart);D ratio=D(-2)*total.r/total.tt;if(!std::isfinite(ratio.value)||ratio.value==0)return std::nullopt;if(ratio.value<0)ratio=-ratio;
 D amplitude=D(4)*root_pair_detail::root(ratio)/(D(1)+T*T);for(int k=0;k<8;++k)if(!std::isfinite(amplitude.dp[k]))return std::nullopt;return Model{R,T,amplitude};
}
}
