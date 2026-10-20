#pragma once
#include <complex>
#include "lcbinint/magnification/algebraic_boundary/exact_radial_support.hpp"
#include "lcbinint/magnification/algebraic_boundary/radius_terms.hpp"
namespace lcbinint::algebraic_boundary::exact_support_detail {
    inline std::array<Dyadic, 5> angular_coefficients(double radius, const lcbinint::algebraic_boundary::PrimaryFrame& pf) {
        const Dyadic r(radius), a(pf.a), m(pf.m0), X(pf.X), Y(pf.Y), rho(pf.rho), one(1), two(2);
        auto r2=r*r, n0r=-X*r2, n0i=-Y*r2, n1r=r*(r2-one+a*X), n1i=r*a*Y, n2r=a*(m-r2);
        auto c0r=n0r+n1r+n2r, c0i=n0i+n1i, c1r=two*n0i, c1i=two*(n2r-n0r), c2r=n1r-n0r-n2r, c2i=n1i-n0i;
        auto k=rho*rho*r2, bm=(r-a)*(r-a), bp=(r+a)*(r+a);
        return {
            k*bm-(c0r*c0r+c0i*c0i), -two*(c0r*c1r+c0i*c1i), k*(bm+bp)-(c1r*c1r+c1i*c1i)-two*(c0r*c2r+c0i*c2i), -two*(c1r*c2r+c1i*c2i), k*bp-(c2r*c2r+c2i*c2i)
        };
    }
    inline BernsteinResult<4> angular_roots(const std::array<Dyadic, 5>& p) {
        std::array<Dyadic, 5> shifted {
        };
        for(int j=0; j<=4; ++j) {
            int choose=1;
            for(int k=j; k<=4; ++k) {
                if(k>j)choose=choose*k/(k-j);
                auto term=p[k];
                term.n*=choose;
                if((k-j)&1)term.n=-term.n;
                shifted[j]=shifted[j]+term;
            }
            shifted[j].e+=j;
        }
        int base=0;
        for(auto& c:shifted)if(c.n!=0)base=std::min(base, c.e);
        constexpr int den=12, binomial[5]= {
            1, 4, 6, 4, 1
        };
        BernsteinBox<4> box;
        box.lo=-1;
        box.hi=1;
        for(int i=0; i<=4; ++i) {
            int choose=1;
            for(int k=0; k<=std::min(i, 4); ++k) {
                if(k)choose=choose*(i-k+1)/k;
                Integer term=shifted[k].n;
                term<<=shifted[k].e-base;
                box.b[i]+=term*(den/binomial[k])*choose;
            }
        }
        return isolate_bernstein(std::move(box), 256);
    }
    struct SeedResult {
        bool completed=false;
        std::vector<std::array<double, 2>> seeds;
        int bands=0;
    };
    inline SeedResult seeds(const lcbinint::algebraic_boundary::PrimaryFrame& pf) {
        using namespace lcbinint::algebraic_boundary;
        SeedResult out;
        auto roots=isolate(pf);
        if(!roots.completed)return out;
        const int power = radial_domain_power(pf);
        if (power < 0) return out;
        double vmax = std::ldexp(1., power);
        double left=0;
        auto sample=[&](double low, double high) {
            if(!(high>low))return true;
            double v=.5*(low+high), R=std::sqrt(v);
            __float128 exact_square=(__float128)R*R;
            if(!(exact_square>low&&exact_square<high)||R==pf.a)return false;
            auto p=angular_coefficients(R, pf);
            for(bool reciprocal: {
                false, true
            }) {
                auto pc=p;
                if(reciprocal)pc= {
                    p[4], -p[3], p[2], -p[1], p[0]
                };
                auto angles=angular_roots(pc);
                if(!angles.completed)return false;
                double start=-1;
                auto witness=[&](double lo, double hi) {
                    if(!(hi>lo))return true;
                    double t=.5*(lo+hi);
                    if(!(t>lo&&t<hi))return false;
                    Dyadic exact(t), value=pc[4];
                    for(int k=3; k>=0; --k)value=value*exact+pc[k];
                    if(value.n<=0)return true;
                    double re=R*(1-t*t)/(1+t*t), im=2*R*t/(1+t*t);
                    if(reciprocal) {
                        re=-re;
                        im=-im;
                    }
                    const std::complex<double> z(re, im);
                const double d0 = std::abs(z), d1 = std::abs(z-pf.a);
                const double eps = std::numeric_limits<double>::epsilon();
                const double coordinate_error = 64*eps*(1+R+pf.a+
                    std::abs(pf.X)+std::abs(pf.Y));
                if (!(d0 > coordinate_error && d1 > coordinate_error)) return false;
                const double map_error = coordinate_error*(1+
                    pf.m0/(d0*(d0-coordinate_error))+
                    (1-pf.m0)/(d1*(d1-coordinate_error)));
                const auto w = z-pf.m0/std::conj(z)-
                    (1-pf.m0)/(std::conj(z)-pf.a);
                const double distance = std::abs(w-std::complex<double>(pf.X, pf.Y));
                const double arithmetic_error = 4096*eps*(1+std::abs(z)+
                    std::abs(w)+pf.m0/d0+(1-pf.m0)/d1);
                if (!(distance+map_error+arithmetic_error < pf.rho)) return false;
                out.seeds.push_back( {
                        re, im
                    });
                    return true;
                };
                for(auto& root:angles.boxes) {
                    if(!witness(start, root.lo))return false;
                    start=root.hi;
                }
                if(!witness(start, 1))return false;
            }
            ++out.bands;
            return true;
        };
        for(auto& root:roots.boxes) {
            if(!sample(left, root.lo))return out;
            left=root.hi;
        }
        if(!sample(left, vmax))return out;
        out.completed=!out.seeds.empty();
        return out;
    }
}
