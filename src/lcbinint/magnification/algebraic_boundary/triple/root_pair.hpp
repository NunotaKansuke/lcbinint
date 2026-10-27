#pragma once
#include "geometry.hpp"
#include "gradient_quadrature.hpp"
#include <optional>
#include "interval.hpp"
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
// Parameter jets are absent from the value instantiation. The polynomial
// builder is shared with the primal boundary, including reciprocal charts.
template<int N> struct Jet {
    double value=0;std::array<double,N> dp{};
    Jet()=default;Jet(double x):value(x){}
};
template<int N> inline __attribute__((always_inline)) Jet<N> operator+(Jet<N> a,const Jet<N>& b){a.value+=b.value;for(int k=0;k<N;++k)a.dp[k]+=b.dp[k];return a;}
template<int N> inline __attribute__((always_inline)) Jet<N> operator-(Jet<N> a){a.value=-a.value;for(auto& x:a.dp)x=-x;return a;}
template<int N> inline __attribute__((always_inline)) Jet<N> operator-(Jet<N> a,const Jet<N>& b){return a+(-b);}
template<int N> inline __attribute__((always_inline)) Jet<N> operator*(const Jet<N>& a,const Jet<N>& b){Jet<N> c(a.value*b.value);for(int k=0;k<N;++k)c.dp[k]=a.dp[k]*b.value+a.value*b.dp[k];return c;}
template<int N> Jet<N> inverse(const Jet<N>& a){Jet<N> c(1/a.value);for(int k=0;k<N;++k)c.dp[k]=-a.dp[k]*c.value*c.value;return c;}
template<int N> Jet<N> root(const Jet<N>& a){Jet<N> c(std::sqrt(a.value));for(int k=0;k<N;++k)c.dp[k]=a.dp[k]/(2*c.value);return c;}
// Fuse complex jet operations rather than copying four intermediate
// eight-component scalar jets per complex product.
template<int N> inline __attribute__((always_inline)) coefficient_detail::C<Jet<N>> operator*(
        const coefficient_detail::C<Jet<N>>& a,const coefficient_detail::C<Jet<N>>& b) {
    coefficient_detail::C<Jet<N>> c;
    c.re.value=a.re.value*b.re.value-a.im.value*b.im.value;
    c.im.value=a.re.value*b.im.value+a.im.value*b.re.value;
    for(int k=0;k<N;++k) {
        c.re.dp[k]=(a.re.dp[k]*b.re.value+a.re.value*b.re.dp[k])-(a.im.dp[k]*b.im.value+a.im.value*b.im.dp[k]);
        c.im.dp[k]=(a.re.dp[k]*b.im.value+a.re.value*b.im.dp[k])+(a.im.dp[k]*b.re.value+a.im.value*b.re.dp[k]);
    }
    return c;
}
inline std::array<double,7> shifted(const std::array<double,7>& p,double m) {
    std::array<double,7> a{};int degree=0;
    for(int j=6;j>=0;--j){for(int k=degree;k>0;--k)a[k]=a[k]*m+a[k-1];a[0]=a[0]*m+p[j];if(degree<6)++degree;}
    return a;
}
struct Equations {double e,o,em,ev,om,ov;};
inline Equations equations(const std::array<double,7>& p,double m,double v) {
    const auto a=shifted(p,m);const double v2=v*v;
    return {a[0]+v*a[2]+v2*a[4]+v2*v*a[6],a[1]+v*a[3]+v2*a[5],
        a[1]+3*v*a[3]+5*v2*a[5],a[2]+2*v*a[4]+3*v2*a[6],
        2*a[2]+4*v*a[4]+6*v2*a[6],a[3]+2*v*a[5]};
}
template<int N> struct Pair {
    Jet<N> mean,square;std::array<Jet<N>,5> quotient{};
    bool valid=false;
    bool initialize(const std::array<Jet<N>,7>& p,double lo,double hi) {
        std::array<double,7> pc{};for(int j=0;j<7;++j)pc[j]=p[j].value;
        double m=(lo+hi)*.5,v=(hi-lo)*(hi-lo)*.25;
        if(!(v>0))return false;
        for(int iteration=0;iteration<8;++iteration) {
            const auto f=equations(pc,m,v);const double det=f.em*f.ov-f.ev*f.om;
            if(!(std::abs(det)>1e-300) || !std::isfinite(det))return false;
            const double dm=(f.ev*f.o-f.ov*f.e)/det,dv=(f.om*f.e-f.em*f.o)/det;
            m+=dm;v+=dv;
            if(!(v>0) || std::abs(m-(lo+hi)*.5)>.01*(hi-lo) || std::abs(v-(hi-lo)*(hi-lo)*.25)>.01*(hi-lo)*(hi-lo))return false;
            if(std::abs(dm)<1e-12*(1+std::abs(m)) &&
               std::abs(dv)<1e-12*(1+m*m))break;
            if(iteration==7)return false;
        }
        return finish(p,m,v);
    }
    bool initialize_complex(const std::array<Jet<N>,7>& p,double m,double v) {
        if(!(v<0))return false;
        const double initial_m=m,initial_v=v;
        std::array<double,7> pc{};for(int j=0;j<7;++j)pc[j]=p[j].value;
        for(int iteration=0;iteration<10;++iteration) {
            const auto f=equations(pc,m,v);const double det=f.em*f.ov-f.ev*f.om;
            if(!(std::abs(det)>1e-300) || !std::isfinite(det))return false;
            const double dm=(f.ev*f.o-f.ov*f.e)/det,dv=(f.om*f.e-f.em*f.o)/det;
            m+=dm;v+=dv;
            if(!(v<0) || std::abs(m-initial_m)>.5*std::sqrt(-initial_v)+1e-12*(1+std::abs(initial_m)) || std::abs(v-initial_v)>.75*std::abs(initial_v))return false;
            if(std::abs(dm)<1e-12*(1+std::abs(m)) && std::abs(dv)<1e-12*(1+m*m))return finish(p,m,v);
        }
        return false;
    }
    bool finish(const std::array<Jet<N>,7>& p,double m,double v) {
        valid=false;std::array<double,7> pc{};for(int j=0;j<7;++j)pc[j]=p[j].value;
        mean=Jet<N>(m);square=Jet<N>(v);
        const auto f=equations(pc,m,v);const double det=f.em*f.ov-f.ev*f.om;
        if(!(std::abs(det)>1e-300) || !std::isfinite(det))return false;
        std::array<double,N> rhs_e{},rhs_o{};
        double even=1,odd=0;
        for(int j=0;j<7;++j) {
            for(int k=0;k<N;++k){rhs_e[k]+=even*p[j].dp[k];rhs_o[k]+=odd*p[j].dp[k];}
            const double next=m*even+v*odd;odd=even+m*odd;even=next;
        }
        const double inv_det=1/det;
        for(int k=0;k<N;++k) {
            mean.dp[k]=(f.ev*rhs_o[k]-f.ov*rhs_e[k])*inv_det;
            square.dp[k]=(f.om*rhs_e[k]-f.em*rhs_o[k])*inv_det;
        }
        for(int k=0;k<N;++k)if(!std::isfinite(mean.dp[k]) || !std::isfinite(square.dp[k]))return false;
        const Jet<N> constant=mean*mean-square,two_m=Jet<N>(2)*mean;
        quotient[4]=p[6];quotient[3]=p[5]+two_m*quotient[4];
        for(int j=2;j>=0;--j)quotient[j]=p[j+2]+two_m*quotient[j+1]-constant*quotient[j+2];
        valid=true;return true;
    }
};
// A frozen rotation of the half-angle chart, applied identically to
// values and every RHS. Centering the chart on the arc avoids an artificial
// rational boundary layer near the global half-angle chart's pole.
inline int binomial(int n,int k){int x=1;for(int j=1;j<=k;++j)x=x*(n-j+1)/j;return x;}
template<int N> std::array<Jet<N>,7> rotate(const std::array<Jet<N>,7>& p,double angle) {
    const double kappa=std::tan(angle*.5),factor=1+kappa*kappa,normal=1/(factor*factor*factor);
    std::array<double,7> power{};power[0]=1;for(int j=1;j<7;++j)power[j]=power[j-1]*kappa;
    std::array<Jet<N>,7> out{};
    for(int i=0;i<7;++i)for(int j=0;j<7;++j) {
        double weight=0;
        for(int a=std::max(0,i-(6-j));a<=std::min(i,j);++a)
            weight+=binomial(j,a)*binomial(6-j,i-a)*power[j-a]*power[i-a]*((i-a)%2?-1:1);
        weight*=normal;out[i].value+=weight*p[j].value;
        for(int k=0;k<N;++k)out[i].dp[k]+=weight*p[j].dp[k];
    }
    return out;
}
// Direct Fourier construction of the same sextic. Avoid multiplying
// polynomials made mostly of constant zeroes while carrying eight RHS.
// The original generic builder remains the independent primal reference.
template<class T> std::array<T,7> spectrum_harmonics(T R,
        const std::array<coefficient_detail::C<T>,3>& lenses,
        const std::array<T,3>& masses,coefficient_detail::C<T> source,T rho) {
    using C=coefficient_detail::C<T>;using coefficient_detail::conj;
    const T R2=R*R,R3=R2*R;const C l1=conj(lenses[1]),l2=conj(lenses[2]);
    const std::array<C,3> d{C{R3},C{}-C{R2}*(l1+l2),C{R}*l1*l2};
    const std::array<C,4> n{C{}-source*d[0],
        C{R}*d[0]-source*d[1]-C{R2*(masses[0]+masses[1]+masses[2])},
        C{R}*d[1]-source*d[2]+C{R}*(C{masses[0]+masses[2]}*l1+C{masses[0]+masses[1]}*l2),
        C{R}*d[2]-C{masses[0]}*l1*l2};
    auto norm=[](C z){return z.re*z.re+z.im*z.im;};
    const T rho2=rho*rho,h0=rho2*(norm(d[0])+norm(d[1])+norm(d[2]))-(norm(n[0])+norm(n[1])+norm(n[2])+norm(n[3]));
    const C h1=C{rho2}*(d[1]*conj(d[0])+d[2]*conj(d[1]))
        -(n[1]*conj(n[0])+n[2]*conj(n[1])+n[3]*conj(n[2]));
    const C h2=C{rho2}*d[2]*conj(d[0])-(n[2]*conj(n[0])+n[3]*conj(n[1]));
    const C h3=C{}-n[3]*conj(n[0]);
    return {h0,h1.re,h1.im,h2.re,h2.im,h3.re,h3.im};
}
// Rotate the three Fourier harmonics before conversion to the sextic. This
// replaces a 7x7 polynomial chart transform for every parameter RHS.
template<class T> T linear_scale(const T& a,double c){return a*T(c);}
template<int N> Jet<N> linear_scale(const Jet<N>& a,double c){Jet<N> out;out.value=a.value*c;for(int j=0;j<N;++j)out.dp[j]=a.dp[j]*c;return out;}
template<class T> std::array<T,7> harmonic_polynomial(std::array<T,7> h,double co=1,double si=0){
    if(co!=1||si!=0){const double c2=co*co-si*si,s2=2*co*si,c3=c2*co-s2*si,s3=s2*co+c2*si;const std::array<double,3> c{co,c2,c3},s{si,s2,s3};
        for(int j=0;j<3;++j){const T re=h[2*j+1],im=h[2*j+2];h[2*j+1]=linear_scale(re,c[j])-linear_scale(im,s[j]);h[2*j+2]=linear_scale(re,s[j])+linear_scale(im,c[j]);}}
    const T& h0=h[0];const T& r1=h[1];const T& i1=h[2];const T& r2=h[3];const T& i2=h[4];const T& r3=h[5];const T& i3=h[6];
    return {h0+linear_scale(r1+r2+r3,2),linear_scale(i1,-4)+linear_scale(i2,-8)+linear_scale(i3,-12),
        linear_scale(h0,3)+linear_scale(r1-linear_scale(r2,5)-linear_scale(r3,15),2),linear_scale(i1,-8)+linear_scale(i3,40),
        linear_scale(h0,3)+linear_scale(-r1-linear_scale(r2,5)+linear_scale(r3,15),2),linear_scale(i1,-4)+linear_scale(i2,8)+linear_scale(i3,-12),
        h0+linear_scale(-r1+r2-r3,2)};
}
template<class T> std::array<T,7> spectrum_coefficients(T R,
        const std::array<coefficient_detail::C<T>,3>& lenses,const std::array<T,3>& masses,
        coefficient_detail::C<T> source,T rho){return harmonic_polynomial(spectrum_harmonics(R,lenses,masses,source,rho));}
// Point-owned radial coefficient tape. Build geometry/parameter work once;
// subsequent radii use Horner and add their material radial velocity.
template<int N> struct RadialPolynomial {
    std::array<Jet<N>,9> c{};int degree=0;
    RadialPolynomial()=default;RadialPolynomial(double x){c[0]=Jet<N>(x);}
    explicit RadialPolynomial(Jet<N> x){c[0]=x;}
};
template<int N> RadialPolynomial<N> operator-(RadialPolynomial<N> a){for(int j=0;j<=a.degree;++j)a.c[j]=-a.c[j];return a;}
template<int N> RadialPolynomial<N> operator+(RadialPolynomial<N> a,const RadialPolynomial<N>& b){a.degree=std::max(a.degree,b.degree);for(int j=0;j<=b.degree;++j)a.c[j]=a.c[j]+b.c[j];return a;}
template<int N> RadialPolynomial<N> operator-(RadialPolynomial<N> a,const RadialPolynomial<N>& b){return a+(-b);}
template<int N> RadialPolynomial<N> operator*(const RadialPolynomial<N>& a,const RadialPolynomial<N>& b){RadialPolynomial<N> c;c.degree=a.degree+b.degree;if(c.degree>8)throw std::logic_error("root-pair radial polynomial degree");for(int i=0;i<=a.degree;++i)for(int j=0;j<=b.degree;++j)c.c[i+j]=c.c[i+j]+a.c[i]*b.c[j];return c;}
template<int N> struct RadialCache {
    using J=Jet<N>;using P=RadialPolynomial<N>;using C=coefficient_detail::C<P>;
    std::array<P,7> coefficients,harmonics;
    explicit RadialCache(const Geometry& p) {
        J X(p.X),Y(p.Y),rho(p.rho),m1(p.m1),m2(p.m2),a(p.a),bx(p.bx),by(p.by);
        if constexpr(N==8){X.dp[0]=1;Y.dp[1]=1;rho.dp[2]=1;m1.dp[3]=1;m2.dp[4]=1;a.dp[5]=1;bx.dp[6]=1;by.dp[7]=1;}
        P R;R.degree=1;R.c[1]=J(1);
        const std::array<C,3> lenses{C{},C{P(a)},C{P(bx),P(by)}};
        harmonics=spectrum_harmonics(R,lenses,std::array<P,3>{P(J(1)-m1-m2),P(m1),P(m2)},C{P(X),P(Y)},P(rho));coefficients=harmonic_polynomial(harmonics);
    }
    std::array<J,7> at(double R,const std::array<double,N>& motion) const {
        std::array<J,7> result;
        for(int k=0;k<7;++k) {
            const auto& p=coefficients[k];J value=p.c[p.degree];double derivative=0;
            for(int j=p.degree-1;j>=0;--j) {
                derivative=derivative*R+value.value;value.value=value.value*R+p.c[j].value;
                for(int i=0;i<N;++i)value.dp[i]=value.dp[i]*R+p.c[j].dp[i];
            }
            for(int i=0;i<N;++i)value.dp[i]+=derivative*motion[i];
            result[k]=value;
        }
        return result;
    }
    std::array<Jet<0>,7> scalar_harmonics(double R)const{
        std::array<Jet<0>,7> result;for(int k=0;k<7;++k){const auto& p=harmonics[k];double value=p.c[p.degree].value;for(int j=p.degree-1;j>=0;--j)value=value*R+p.c[j].value;result[k].value=value;}return result;
    }
    std::array<J,7> at_harmonics(double R,const std::array<double,N>& motion) const {
        std::array<J,7> result;
        for(int k=0;k<7;++k) {
            const auto& p=harmonics[k];J value=p.c[p.degree];double derivative=0;
            for(int j=p.degree-1;j>=0;--j) {
                derivative=derivative*R+value.value;value.value=value.value*R+p.c[j].value;
                for(int i=0;i<N;++i)value.dp[i]=value.dp[i]*R+p.c[j].dp[i];
            }
            for(int i=0;i<N;++i)value.dp[i]+=derivative*motion[i];
            result[k]=value;
        }
        return result;
    }
};
template<int N> struct Coefficients {
    static_assert(N==0 || N==8,"triple root-pair jet dimension");
    using J=Jet<N>;using C=coefficient_detail::C<J>;
    J R,rho,lx1,lx2,ly2;std::array<J,7> harmonics{};
    std::optional<std::array<long double,7>> wide_harmonics;
    std::optional<std::array<__float128,7>> quad_harmonics;
    Coefficients()=default;
    Coefficients(double radius,const Geometry& p,const std::array<double,N>& motion,const RadialCache<N>* radial=nullptr) {
        R=J(radius);rho=J(p.rho);lx1=J(p.a);lx2=J(p.bx);ly2=J(p.by);
        J X(p.X),Y(p.Y),m1(p.m1),m2(p.m2);
        if constexpr(N==8){X.dp[0]=1;Y.dp[1]=1;rho.dp[2]=1;m1.dp[3]=1;m2.dp[4]=1;lx1.dp[5]=1;lx2.dp[6]=1;ly2.dp[7]=1;R.dp=motion;}
        const std::array<C,3> lenses{C{},C{lx1},C{lx2,ly2}};
        harmonics=radial?radial->at_harmonics(radius,motion):spectrum_harmonics(R,lenses,std::array<J,3>{J(1)-m1-m2,m1,m2},C{X,Y},rho);
    }
};
template<int N> struct Arc {
    using J=Jet<N>;using C=coefficient_detail::C<J>;
    Pair<N> pair;std::array<J,7> polynomial{};J R,rho,lx1,ly1,lx2,ly2;double center=0;std::array<std::array<J,3>,2> poles{};double half_width=0,inverse_rho=0;bool reciprocal=false;
    bool configure(double radius,const Geometry& p,double basis,const std::array<double,N>& motion={},
            std::optional<Coefficients<N>>* cache=nullptr,const RadialCache<N>* radial=nullptr) {
        const double pi=std::acos(-1.);center=basis;
        reciprocal=std::abs(std::remainder(center,2*pi))>pi*.5;
        Coefficients<N> local;
        const Coefficients<N>* data=nullptr;
        if(cache){if(!*cache)cache->emplace(radius,p,motion,radial);data=&**cache;}
        else {local=Coefficients<N>(radius,p,motion,radial);data=&local;}
        R=data->R;rho=data->rho;
        const double co=std::cos(center),si=std::sin(center);
        lx1=J(co)*data->lx1;ly1=J(-si)*data->lx1;
        lx2=J(co)*data->lx2+J(si)*data->ly2;ly2=J(co)*data->ly2-J(si)*data->lx2;
        polynomial=harmonic_polynomial(data->harmonics,co,si);
        const auto make_pole=[&](J lx,J ly) {
            const J minus=R-lx,plus=R+lx;
            std::array<J,3> q{minus*minus+ly*ly,J(-4)*R*ly,plus*plus+ly*ly};
            return q;
        };
        poles={make_pole(lx1,ly1),make_pole(lx2,ly2)};
        inverse_rho=1/rho.value;return true;
    }
    bool initialize(double radius,const Geometry& p,double lo,double hi,const std::array<double,N>& motion={},
            std::optional<Coefficients<N>>* cache=nullptr,bool local_center=true,const RadialCache<N>* radial=nullptr,bool solve_inside=true) {
        const double pi=std::acos(-1.),mid=(lo+hi)*.5,width=hi-lo;
        if(!(width>0 && width<1.9*pi))return false;
        const bool reciprocal=std::abs(std::remainder(mid,2*pi))>pi*.5;
        const double base=reciprocal?pi:0,turn=2*pi*std::round((mid-base)/(2*pi));
        const double basis=local_center?mid:base+turn,l=lo-basis,h=hi-basis;
        if(!(l>-pi+.01 && h<pi-.01))return false;
        const double tl=std::tan(l*.5),th=std::tan(h*.5);
        if(!(th>tl) || std::max(std::abs(tl),std::abs(th))>16)return false;
        if(!configure(radius,p,basis,motion,cache,radial))return false;
        if(solve_inside && !pair.initialize(polynomial,tl,th))return false;
        half_width=solve_inside?std::sqrt(pair.square.value):0;return true;
    }
    // Generalize binary's factor-wise Cauchy screen to the quartic
    // remainder. Interval arithmetic bounds every nonconstant Taylor term
    // on |x| <= radius; <= half the constant excludes all factor zeroes.
    // This is a geometry/value screen, never a final-gradient error loop.
    bool disk_resolved(double radius) const {
        using I=interval_detail::Interval<double>;
        const I m(pair.mean.value),h(std::sqrt(pair.square.value)),r=I(radius)*h;
        const auto safe=[&](const std::array<double,5>& coefficients,int degree) {
            std::array<I,5> a{};int n=0;
            for(int j=degree;j>=0;--j){for(int k=n;k>0;--k)a[k]=a[k]*m+a[k-1];a[0]=a[0]*m+I(coefficients[j]);if(n<degree)++n;}
            if(a[0].lo<=0 && a[0].hi>=0)return false;
            const double constant=std::min(std::abs(a[0].lo),std::abs(a[0].hi));
            I tail(0),power(1);
            for(int j=1;j<=degree;++j){power=power*r;tail=tail+power*I(std::max(std::abs(a[j].lo),std::abs(a[j].hi)));}
            return tail.valid() && tail.hi<interval_detail::down(.5*constant);
        };
        std::array<double,5> q{};for(int j=0;j<5;++j)q[j]=pair.quotient[j].value;
        if(!safe(q,4) || !safe({1,0,1,0,0},2))return false;
        for(const auto& pole:poles)if(!safe({pole[0].value,pole[1].value,pole[2].value,0,0},2))return false;
        return true;
    }
    // Rouché screen for a larger critical-chart cell. Roundoff bounds are
    // propagated alongside the Taylor shift; a failed screen retains the
    // incumbent unit cell. No parameter RHS or promoted arithmetic is used.
    bool analytic_disk(double center,double radius)const{
        if(!(radius>0&&std::isfinite(radius)&&std::isfinite(center)))return false;
        const double eps=std::numeric_limits<double>::epsilon(),gamma=512*eps;
        const auto safe=[&](const std::array<double,5>& c,int degree){std::array<double,5> a{},m{};int n=0;const double ac=std::abs(center);
            for(int j=degree;j>=0;--j){for(int k=n;k>0;--k){a[k]=a[k]*center+a[k-1];m[k]=m[k]*ac+m[k-1];}a[0]=a[0]*center+c[j];m[0]=m[0]*ac+std::abs(c[j]);if(n<degree)++n;}
            const double tiny=128*std::numeric_limits<double>::min(),constant=std::abs(a[0])-gamma*m[0]-tiny;double tail=0,power=1;for(int j=1;j<=degree;++j){power=(power*radius+tiny)*(1+gamma);tail=(tail+(std::abs(a[j])+gamma*m[j]+tiny)*power+tiny)*(1+gamma);}
            return constant>0&&std::isfinite(tail)&&(1+gamma)*tail<.4*constant;
        };
        std::array<double,5> q{};for(int j=0;j<5;++j)q[j]=pair.quotient[j].value;if(!safe(q,4)||!safe({1,0,1,0,0},2))return false;
        for(const auto& pole:poles)if(!safe({pole[0].value,pole[1].value,pole[2].value,0,0},2))return false;return true;
    }
    int short_level() const {return disk_resolved(4)?3:(disk_resolved(2)?4:6);}
    double phi_from_factor(double t,double factor) const {
        double q=pair.quotient[4].value;for(int j=3;j>=0;--j)q=q*t+pair.quotient[j].value;
        double denominator=rho.value*rho.value*R.value*R.value*(1+t*t);
        for(const auto& a:poles)denominator*=((a[2].value*t+a[1].value)*t+a[0].value);
        return factor*q/denominator;
    }
    // Material transport through a complex pair's sinh coordinate. The
    // physical endpoint motion is carried in u, so endpoint singular terms
    // cancel before quadrature rather than in separate integral contributions.
    J critical_coordinate(double theta,const std::array<double,N>& motion)const{
        const double tangent=std::tan((theta-center)*.5);J t(tangent);for(int j=0;j<N;++j)t.dp[j]=.5*(1+tangent*tangent)*motion[j];
        const J z=(t-pair.mean)*inverse(root(-pair.square));J u(std::asinh(z.value));const double d=1/std::sqrt(1+z.value*z.value);for(int j=0;j<N;++j)u.dp[j]=d*z.dp[j];return u;
    }
    bool moving_critical_density(const J& u,J& out)const{
        const double v=pair.square.value;if(!(v<0))return false;
        const double h=std::sqrt(-v),sh=std::sinh(u.value),ch=std::cosh(u.value),t=pair.mean.value+h*sh,one=1+t*t;
        const auto& q=pair.quotient;double Q=q[4].value;for(int k=3;k>=0;--k)Q=Q*t+q[k].value;
        const double Qt=((4*q[4].value*t+3*q[3].value)*t+2*q[2].value)*t+q[1].value;
        std::array<double,2> pole{},inverse_pole{};double denominator_slope=6*t/one;
        for(int j=0;j<2;++j){const auto& a=poles[j];pole[j]=(a[2].value*t+a[1].value)*t+a[0].value;inverse_pole[j]=1/pole[j];denominator_slope+=(2*a[2].value*t+a[1].value)*inverse_pole[j];}
        const double ratio=Q/(one*one*one*pole[0]*pole[1]);if(!(ratio>0)||!std::isfinite(ratio))return false;
        out.value=2*inverse_rho*(-v)*ch*ch*std::sqrt(ratio);
        const double inverse_q=1/Q,slope=.5*(Qt*inverse_q-denominator_slope),motion_scale=h*ch,spread_scale=-sh/(2*h),coordinate_slope=2*sh/ch;
        // The scalar field and its logarithmic spatial derivative are shared
        // by all RHS. No eight-lane products/inverses of the denominator.
        for(int j=0;j<N;++j){double qp=q[4].dp[j];for(int k=3;k>=0;--k)qp=qp*t+q[k].dp[j];double pole_partial=0;
            for(int k=0;k<2;++k){const auto& a=poles[k];pole_partial+=((a[2].dp[j]*t+a[1].dp[j])*t+a[0].dp[j])*inverse_pole[k];}
            const double tp=pair.mean.dp[j]+spread_scale*pair.square.dp[j]+motion_scale*u.dp[j];
            out.dp[j]=out.value*(-rho.dp[j]*inverse_rho+pair.square.dp[j]/v+coordinate_slope*u.dp[j]+.5*(qp*inverse_q-pole_partial)+slope*tp);
            if(!std::isfinite(out.dp[j]))return false;
        }return std::isfinite(out.value);
    }
    // Fixed-physical-angle derivative after removing the *critical* pair.
    // With t=m+h*sinh(u) (complex pair) or m +/- h*cosh(u) (real gap),
    // dt/du equals sqrt((t-m)^2-v). The dangerous inverse square root
    // cancels before any parameter arithmetic is performed.
    bool critical_density(double delta,double factor,J& out) const {
        const double t=pair.mean.value+delta,one=1+t*t;
        const auto& q=pair.quotient;double Q=q[4].value;for(int j=3;j>=0;--j)Q=Q*t+q[j].value;
        std::array<double,2> pole{};for(int j=0;j<2;++j){const auto& a=poles[j];pole[j]=(a[2].value*t+a[1].value)*t+a[0].value;}
        const double ratio=Q/(one*one*one*pole[0]*pole[1]);
        if(!(ratio>0) || !std::isfinite(ratio))return false;
        const double K=2*inverse_rho*std::sqrt(ratio),invQ=1/Q;
        const std::array<double,2> invPole{1/pole[0],1/pole[1]};
        J result;result.value=K*factor;
        for(int k=0;k<N;++k) {
            double qp=q[4].dp[k];for(int j=3;j>=0;--j)qp=qp*t+q[j].dp[k];
            double dp_den=0;
            for(int j=0;j<2;++j){const auto& a=poles[j];dp_den+=((a[2].dp[k]*t+a[1].dp[k])*t+a[0].dp[k])*invPole[j];}
            result.dp[k]=K*(-delta*pair.mean.dp[k]-.5*pair.square.dp[k]
                +factor*(.5*(qp*invQ-dp_den)-rho.dp[k]*inverse_rho));
        }
        for(double d:result.dp)if(!std::isfinite(d))return false;
        out=result;return std::isfinite(result.value);
    }
    // Density with the known sqrt(1-x*x) factor removed. Neither value
    // nor parameter numerator is evaluated by subtracting two root values.
    bool density(double x,J& out) const {
        const double v=pair.square.value,h=half_width,t=pair.mean.value+h*x,t2=t*t,one=1+t2;
        const auto& q=pair.quotient;
        double Q=q[4].value;for(int j=3;j>=0;--j)Q=Q*t+q[j].value;
        const double Qt=((4*q[4].value*t+3*q[3].value)*t+2*q[2].value)*t+q[1].value;
        std::array<double,2> pole{},pole_t{};
        for(int j=0;j<2;++j){const auto& a=poles[j];pole[j]=(a[2].value*t+a[1].value)*t+a[0].value;pole_t[j]=2*a[2].value*t+a[1].value;}
        const double ratio=-Q/(one*one*one*pole[0]*pole[1]);
        if(!(ratio>0) || !std::isfinite(ratio))return false;
        const double K=2*inverse_rho*std::sqrt(ratio),invQ=1/Q,invOne=1/one,invRho=inverse_rho;
        const std::array<double,2> invPole{1/pole[0],1/pole[1]};
        J result;result.value=K*v;
        // Carry v*dt, not dt: this avoids dividing a tangent by the
        // vanishing root separation before multiplying by v again.
        for(int k=0;k<N;++k) {
            const double dv=pair.square.dp[k],vdt=v*pair.mean.dp[k]+.5*h*x*dv;
            double qp=q[4].dp[k];for(int j=3;j>=0;--j)qp=qp*t+q[j].dp[k];
            double denominator=6*t*vdt*invOne;
            for(int j=0;j<2;++j){const auto& a=poles[j];const double pp=(a[2].dp[k]*t+a[1].dp[k])*t+a[0].dp[k];denominator+=(v*pp+pole_t[j]*vdt)*invPole[j];}
            result.dp[k]=K*(dv-v*rho.dp[k]*invRho+.5*((v*qp+Qt*vdt)*invQ-denominator));

        }
        for(double d:result.dp)if(!std::isfinite(d))return false;
        out=result;return std::isfinite(result.value);
    }
};
} // namespace lcbinint::algebraic_boundary::triple::root_pair_detail
