#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace lcbinint::algebraic_boundary::triple {
using Complex = std::complex<double>;
inline double complex_magnitude(Complex z) {
    const double squared=std::norm(z);
    if(squared>=32*std::numeric_limits<double>::min() && std::isfinite(squared))
        return std::sqrt(squared);
    return std::abs(z);
}
// The ordinary range needs one real reciprocal. Preserve scaled complex
// division for extreme distances where squaring or the reciprocal is unsafe.
inline Complex inverse_conjugate(Complex d) {
    const double n=std::norm(d);
    if(n>=32*std::numeric_limits<double>::min() && std::isfinite(n))return d*(1/n);
    return 1./std::conj(d);
}

// Fixed primary frame. Derivatives: X,Y,rho,m1,m2,a,bx,by.
struct Geometry {
    double X, Y, rho, m1, m2, a, bx, by;
    std::array<Complex, 3> lenses() const {
        return {Complex(0,0), Complex(a,0), Complex(bx,by)};
    }
    std::array<double, 3> masses() const { return {1-m1-m2,m1,m2}; }
    void validate() const {
        for (double x : {X,Y,rho,m1,m2,a,bx,by})
            if (!std::isfinite(x)) throw std::invalid_argument("nonfinite triple geometry");
        if (!(rho>0) || m1<0 || m2<0 || m1+m2>1)
            throw std::invalid_argument("invalid triple radius or masses");
    }
};

// Parameter derivatives are absent from the value instantiation.
template<bool WithGradient> struct Evaluation;
template<> struct Evaluation<false> { double phi,roundoff=0,angular_sensitivity=0; };
template<> struct Evaluation<true> {
    double phi, dtheta, dR;
    std::array<double,8> dp;
};

template<bool WithGradient=false,bool EstimateError=false>
inline Evaluation<WithGradient> evaluate(double R, double theta, const Geometry& p) {
    const Complex w(std::cos(theta),std::sin(theta)), z=R*w;
    const auto lenses=p.lenses(); const auto masses=p.masses();
    Complex g=z-Complex(p.X,p.Y);
    std::array<Complex,3> inverse{};
    for (int j=0;j<3;++j) {
        // Zero-mass poles are removable, but derivatives there are not defined.
        const auto den=std::conj(z-lenses[j]);
        if (std::norm(den)==0) {
            if constexpr (!WithGradient) { if (masses[j]==0) continue; }
            throw std::domain_error("triple evaluation at lens pole");
        }
        inverse[j]=inverse_conjugate(z-lenses[j]);
        g-=masses[j]*inverse[j];
    }
    const double invrho2=1/(p.rho*p.rho);
    Evaluation<WithGradient> out{};
    out.phi=1-std::norm(g)*invrho2;
    if constexpr(EstimateError && !WithGradient) {
        double deflection=0,shear_bound=0;
        for(int j=0;j<3;++j) {
            deflection+=masses[j]*std::abs(inverse[j]);
            shear_bound+=masses[j]*std::norm(inverse[j]);
        }
        const double eps=std::numeric_limits<double>::epsilon();
        const double delta=32*eps*(R+std::hypot(p.X,p.Y)+deflection+R*shear_bound);
        out.roundoff=(2*std::abs(g)*delta+delta*delta)*invrho2+
            16*eps*(1+std::norm(g)*invrho2);
        out.angular_sensitivity=2*std::abs(g)*R*(1+shear_bound)*invrho2;
    }
    if constexpr (WithGradient) {
        const auto derivative=[&](Complex dg) {
            return -2*std::real(std::conj(g)*dg)*invrho2;
        };
        Complex shear=0;
        for(int j=0;j<3;++j) shear+=masses[j]*inverse[j]*inverse[j];
        out.dtheta=derivative(Complex(0,1)*(z-std::conj(z)*shear));
        out.dR=derivative(w+std::conj(w)*shear);
        out.dp={derivative(-1.0),derivative(Complex(0,-1)),
            2*std::norm(g)*invrho2/p.rho,
            derivative(inverse[0]-inverse[1]),
            derivative(inverse[0]-inverse[2]),
            derivative(-masses[1]*inverse[1]*inverse[1]),
            derivative(-masses[2]*inverse[2]*inverse[2]),
            derivative(Complex(0,1)*masses[2]*inverse[2]*inverse[2])};
    }
    return out;
}

// Scalar-generic coefficient construction: the same operations are used for
// binary64 values and outward enclosures reconstructed from original inputs.
namespace coefficient_detail {
template<class T> struct C {
    T re{},im{};
    C()=default;
    C(T r,T i=T(0)):re(r),im(i) {}
};
template<class T> C<T> operator+(C<T> x,C<T> y) {return {x.re+y.re,x.im+y.im};}
template<class T> C<T> operator-(C<T> x,C<T> y) {return {x.re-y.re,x.im-y.im};}
template<class T> C<T> operator*(C<T> x,C<T> y) {
    return {x.re*y.re-x.im*y.im,x.re*y.im+x.im*y.re};
}
template<class T> C<T> conj(C<T> x) {return {x.re,-x.im};}
}
template<class T> struct BoundaryT {
    std::array<T,7> coefficients{};
    std::array<coefficient_detail::C<T>,4> numerator{};
    std::array<coefficient_detail::C<T>,3> denominator{};
};
using Boundary=BoundaryT<double>;

template<class T> inline BoundaryT<T> boundary_from_radius(T R,const Geometry& p) {
    using coefficient_detail::C;
    using coefficient_detail::conj;
    const T rho(p.rho);
    // With u on the unit circle and b the conjugate of the third lens,
    // D=R(R-a*u)(R-b*u). Collect N=(R*u-source)*D minus the three
    // deflection terms. The normalized masses sum to one exactly.
    const T a(p.a),m1(p.m1),m2(p.m2),r2=R*R,excess=r2-T(1);
    const C<T> b{T(p.bx),-T(p.by)},ab{a*b.re,a*b.im},source{T(p.X),T(p.Y)};
    BoundaryT<T> out;
    auto& d=out.denominator;
    d[0]=C<T>(R*r2);
    d[1]={-r2*(a+b.re),-r2*b.im};
    d[2]={R*ab.re,R*ab.im};
    auto& n=out.numerator;
    n[0]={-source.re*d[0].re,-source.im*d[0].re};
    n[1]=C<T>(r2*excess)-source*d[1];
    const T third_weight=excess+m2;
    n[2]=C<T>{-R*((excess+m1)*a+third_weight*b.re),-R*third_weight*b.im}-source*d[2];
    const T outer_weight=excess+m1+m2;
    n[3]={outer_weight*ab.re,outer_weight*ab.im};
    std::array<C<T>,4> h{};
    const T rho2=rho*rho;
    for(int k=0;k<=3;++k) {
        C<T> area{};
        for(int j=0;j+k<3;++j)area=area+d[j+k]*conj(d[j]);
        h[k]={rho2*area.re,rho2*area.im};
        for(int j=0;j+k<4;++j)h[k]=h[k]-n[j+k]*conj(n[j]);
    }
    // For t=tan(theta/2), harmonic k contributes the real part of
    // h[k]*(1+i*t)^(2*k)*(1+t*t)^(3-k), doubled for k>0.
    // Even powers use only real parts, odd powers only imaginary parts.
    // Opposite ends of the sextic share these sums and differences.
    const T even06=h[0].re+T(2)*h[2].re;
    const T delta06=T(2)*(h[1].re+h[3].re);
    const T even24=T(3)*h[0].re-T(10)*h[2].re;
    const T delta24=T(2)*(h[1].re-T(15)*h[3].re);
    const T odd15=-T(4)*(h[1].im+T(3)*h[3].im);
    const T delta15=T(8)*h[2].im;
    out.coefficients={even06+delta06,odd15-delta15,even24+delta24,
        T(8)*(T(5)*h[3].im-h[1].im),even24-delta24,odd15+delta15,even06-delta06};
    return out;
}
template<class T> inline BoundaryT<T> boundary_as(double radius,const Geometry& p) {
    return boundary_from_radius<T>(T(radius),p);
}
inline Boundary boundary(double R,const Geometry& p) {
    p.validate();
    if(!(R>0) || !std::isfinite(R)) throw std::invalid_argument("invalid triple radius");
    return boundary_as<double>(R,p);
}

inline std::array<double,7> reciprocal(const std::array<double,7>& p) {
    std::array<double,7> out{};
    for(int j=0;j<7;++j) out[j]=(j%2?-1:1)*p[6-j];
    return out;
}
} // namespace lcbinint::algebraic_boundary::triple
