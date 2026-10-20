#pragma once
#include "lcbinint/magnification/algebraic_boundary/d14_real_sturm.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <stdexcept>
#include <vector>
namespace lcbinint::algebraic_boundary::exact_support_detail {
    using Integer=boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 0>>;
    struct Dyadic {
        Integer n=0;
        int e=0;
        Dyadic()=default;
        Dyadic(double x) {
            if(!std::isfinite(x))throw std::runtime_error("nonfinite dyadic");
            if(x!=0) {
                int exponent;
                double mantissa=std::frexp(x, &exponent);
                n=static_cast<long long>(std::ldexp(mantissa, 53));
                e=exponent-53;
                normalize();
            }
        }
        void normalize() {
            if(n==0) {
                e=0;
                return;
            }
            Integer magnitude=n<0?-n:n;
            int k=boost::multiprecision::lsb(magnitude);
            n>>=k;
            e+=k;
        }
    };
    inline Dyadic operator-(Dyadic x) {
        x.n=-x.n;
        return x;
    }
    inline Dyadic operator+(const Dyadic& a, const Dyadic& b) {
        if(a.n==0)return b;
        if(b.n==0)return a;
        Dyadic r;
        r.e=std::min(a.e, b.e);
        r.n=(a.n<<(a.e-r.e))+(b.n<<(b.e-r.e));
        r.normalize();
        return r;
    }
    inline Dyadic operator-(const Dyadic&a, const Dyadic&b) {
        return a+(-b);
    }
    inline Dyadic operator*(const Dyadic&a, const Dyadic&b) {
        if (a.n == 0 || b.n == 0) return {};
        Dyadic r;
        r.n=a.n*b.n;
        r.e=a.e+b.e;
        return r;
    }
    template<int N> struct BernsteinBox {
        double lo=0, hi=0;
        std::array<Integer, N+1> b;
        int count=0;
    };
    template<int N> struct BernsteinResult {
        std::vector<BernsteinBox<N>> boxes;
        bool completed=false;
        int nodes=0;
    };
    template<std::size_t N> inline int variations(const std::array<Integer, N>& b) {
        int last=0, n=0;
        for(auto& x:b) {
            int s=x>0?1:x<0?-1:0;
            if(s) {
                if(last&&last!=s)++n;
                last=s;
            }
        }
        return n;
    }
    template<int N> inline std::pair<BernsteinBox<N>, BernsteinBox<N>> split(const BernsteinBox<N>& box) {
        BernsteinBox<N> left, right;
        double mid=.5*(box.lo+box.hi);
        left.lo=box.lo;
        left.hi=right.lo=mid;
        right.hi=box.hi;
        auto work=box.b;
        left.b[0]=work[0]<<N;
        right.b[N]=work[N]<<N;
        for(int k=1; k<=N; ++k) {
            for(int j=0; j<=N-k; ++j)work[j]+=work[j+1];
            left.b[k]=work[0]<<(N-k);
            right.b[N-k]=work[N-k]<<(N-k);
        }
        left.count=variations(left.b);
        right.count=variations(right.b);
        return {
            std::move(left), std::move(right)
        };
    }
    template<int N> inline BernsteinResult<N> isolate_bernstein(BernsteinBox<N> first, int maximum_nodes=2048) {
        using Box=BernsteinBox<N>;
        BernsteinResult<N> out;
        double vmax=first.hi, vmin=first.lo;
        first.count=variations(first.b);
        auto atom=[&](double v) {
            for(const auto& b:out.boxes)if(b.lo==v&&b.hi==v)return;
            Box b;
            b.lo=b.hi=v;
            out.boxes.push_back(std::move(b));
        };
        if(first.b[0]==0)atom(first.lo);
        if(first.b[N]==0)atom(vmax);
        bool allzero=true;
        for(auto& b:first.b)allzero&=b==0;
        if(allzero)return out;
        std::vector<Box> pending;
        pending.push_back(std::move(first));
        while(!pending.empty()) {
            if(++out.nodes>maximum_nodes)return out;
            Box b=std::move(pending.back());
            pending.pop_back();
            if(!b.count)continue;
            if(b.count==1) {
                out.boxes.push_back(std::move(b));
                continue;
            }
            double mid=.5*(b.lo+b.hi);
            if(!(mid>b.lo&&mid<b.hi))return out;
            auto halves=split(b);
            if(halves.first.b[N]==0)atom(mid);
            pending.push_back(std::move(halves.second));
            pending.push_back(std::move(halves.first));
        }
        std::sort(out.boxes.begin(), out.boxes.end(), [](const Box&a, const Box&b) {
            return a.lo<b.lo;
        });
        // Coarse boxes only need a nonempty root-free gap, not accurate endpoints.
        for(std::size_t i=1; i<out.boxes.size(); ++i)for(int k=0; out.boxes[i-1].hi>=out.boxes[i].lo; ++k) {
            if(k==64)return out;
            for(std::size_t j: {
                i-1, i
            }) {
                auto& b=out.boxes[j];
                if(b.lo==b.hi)continue;
                double mid=.5*(b.lo+b.hi);
                if(!(mid>b.lo&&mid<b.hi))return out;
                auto halves=split(b);
                if(halves.first.b[N]==0) {
                    b.lo=b.hi=mid;
                    b.count=0;
                }else if(halves.first.count==1)b=std::move(halves.first);
                else if(halves.second.count==1)b=std::move(halves.second);
                else return out;
            }
        }
        for(auto& b:out.boxes)for(int k=0; b.lo!=b.hi && (b.lo==vmin || b.hi==vmax); ++k) {
            if(k==64)return out;
            double mid=.5*(b.lo+b.hi);
            if(!(mid>b.lo&&mid<b.hi))return out;
            auto halves=split(b);
            if(halves.first.b[N]==0) {
                b.lo=b.hi=mid;
                b.count=0;
            } else if(halves.first.count==1)b=std::move(halves.first);
            else if(halves.second.count==1)b=std::move(halves.second);
            else return out;
        }
        out.completed=true;
        return out;
    }
    using Box=BernsteinBox<14>;
    using Result=BernsteinResult<14>;
    // For r > a+|X|+|Y|+rho+2, the lens deflection is less than
    // 1/(r-a), so no image of the source disk can lie beyond this radius.
    inline int radial_domain_power(const PrimaryFrame& pf) {
        double upper = pf.a;
        for (double term : {std::abs(pf.X), std::abs(pf.Y), pf.rho, 2.0})
            upper = std::nextafter(upper + term, INFINITY);
        if (!std::isfinite(upper) || !(upper > 0.0)) return -1;
        int exponent;
        std::frexp(upper, &exponent);
        return 2 * exponent;
    }
    inline bool make_radial_bernstein_box(const lcbinint::algebraic_boundary::PrimaryFrame& pf, Box& first) {
        using namespace lcbinint::algebraic_boundary;
        if(!std::isfinite(pf.a+pf.m0+pf.X+pf.Y+pf.rho)||!(pf.a>0&&pf.m0>0&&pf.m0<1&&pf.rho>0))return false;
        const int power = radial_domain_power(pf);
        if (power < 0) return false;
        double vmax=std::ldexp(1., power);
        if(!std::isfinite(vmax))return false;
        auto p=positive_detail::polynomial<Dyadic>(pf);
        int base=0;
        for(int i=0; i<=14; ++i)if(p.c[i].n!=0)base=std::min(base, p.c[i].e+power*i);
        constexpr int binomial[15]= {
            1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1
        };
        constexpr int denominator=24024;
        first=Box{};
        first.hi=vmax;
        for(int i=0; i<=14; ++i) {
            int choose=1;
            for(int k=0; k<=i; ++k) {
                if(k)choose=choose*(i-k+1)/k;
                Integer term=p.c[k].n;
                term<<=p.c[k].e+power*k-base;
                first.b[i]+=term*(denominator/binomial[k])*choose;
            }
        }
        return true;
    }
    inline Result isolate(const lcbinint::algebraic_boundary::PrimaryFrame& pf, int maximum_nodes=2048) {
        Box first;
        if(!make_radial_bernstein_box(pf,first))return {};
        return isolate_bernstein(std::move(first),maximum_nodes);
    }
}
