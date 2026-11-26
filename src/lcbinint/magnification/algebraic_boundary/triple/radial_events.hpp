#pragma once
#include "angular_topology.hpp"
#include <tuple>
#include "event_surface.hpp"
#include <deque>
#include <map>
namespace lcbinint::algebraic_boundary::triple {
namespace event_detail {
using physical_detail::Jet;
template<class T> std::optional<Jet<T>> physical(T R,T t,const Geometry&p,bool chart) {
    return physical_detail::Ray<T>(R,p).event_jet(t,chart);
}
template<class T> Jet<T> evaluate(T R,T t,const Geometry& p,bool reciprocal_chart) {
    const auto coeff=boundary_from_radius(Jet<T>(R,T(1),T(0)),p).coefficients;
    Jet<T> x(t,T(0),T(1)),value;
    for(int j=6;j>=0;--j) {
        auto c=reciprocal_chart?coeff[6-j]:coeff[j];
        if(reciprocal_chart && j%2)c=-c;
        value=value*x+c;
    }
    return value;
}
template<class T> Jet<T> physical_or_polynomial(T radius,T coordinate,const Geometry& p,bool chart) {
    auto f=physical(radius,coordinate,p,chart);
    return f?*f:evaluate(radius,coordinate,p,chart);
}
using I=angular_detail::I<double>;
struct Box {double rlo,rhi,tlo,thi;bool reciprocal=false;};
inline double magnitude(I x){return std::max(std::abs(x.lo),std::abs(x.hi));}
// Physical disk bounds avoid the badly scaled cleared polynomial near a
// small-mass pole. A stationary boundary event needs both |g|=rho and
// Re(conj(g) g_theta)=0; excluding either excludes the whole box.
struct PhysicalExclusionData {
    std::array<Complex,3> lenses;
    std::array<double,3> masses,source_distance,lens_norm;
    double source_norm;
    std::array<std::array<double,3>,3> lens_distance{};
    explicit PhysicalExclusionData(const Geometry& p):lenses(p.lenses()),masses(p.masses()),source_norm(std::abs(Complex(p.X,p.Y))) {
        for(int j=0;j<3;++j) {
            source_distance[j]=std::abs(lenses[j]-Complex(p.X,p.Y));
            lens_norm[j]=std::abs(lenses[j]);
            for(int k=0;k<3;++k)lens_distance[j][k]=std::abs(lenses[j]-lenses[k]);
        }
    }
};
inline bool physical_exclusion(const Box& box,const Geometry& p,const PhysicalExclusionData* prepared=nullptr,Box* contracted=nullptr) {
    std::optional<PhysicalExclusionData> local;
    if(!prepared){local.emplace(p);prepared=&*local;}
    const double eps=std::numeric_limits<double>::epsilon();
    const double r=box.rlo+(box.rhi-box.rlo)*.5,t=box.tlo+(box.thi-box.tlo)*.5;
    const double sign=box.reciprocal?-1:1,den=1+t*t;
    const Complex direction=sign*Complex((1-t*t)/den,2*t/den);
    const Complex z=r*direction,source(p.X,p.Y);
    const double chord=std::max(2*std::abs(box.tlo-t)/std::hypot(1.,box.tlo),
        2*std::abs(box.thi-t)/std::hypot(1.,box.thi))/std::hypot(1.,t);
    const double delta=std::hypot(.5*(box.rhi-box.rlo),std::sqrt(box.rhi)*std::sqrt(r)*chord)+64*eps*(1+box.rhi);
    const auto& lenses=prepared->lenses;const auto& masses=prepared->masses;
    std::array<double,3> image_distance{};
    for(int j=0;j<3;++j) {
        image_distance[j]=complex_magnitude(z-lenses[j]);
        const double distance=image_distance[j]+delta+64*eps*(1+image_distance[j]);
        if(!(distance>0))continue;
        double remainder=prepared->source_distance[j]+distance+p.rho;bool valid=true;
        for(int k=0;k<3;++k)if(k!=j) {
            const double lower=prepared->lens_distance[j][k]-distance-
                64*eps*(1+prepared->lens_distance[j][k]);
            if(!(lower>0)){valid=false;break;}
            remainder+=(masses[k]+4*eps)/lower;
        }
        if(valid && (masses[j]-4*eps)/distance>remainder*(1+128*eps)+128*eps)return true;
    }
    // In a thin polar cell the circumscribing disk can cross a lens even
    // though the whole sector is separated from it in radius or angle.
    const double dr=std::max(box.rhi-r,r-box.rlo);
    const double du=dr+.5*box.rhi*chord*chord,dv=box.rhi*chord;
    std::array<double,3> lower_distance;
    std::array<Complex,3> inverses;
    Complex off_axis_source=source;double off_axis_error=0;
    Complex g=z-source,U{};double error=delta,U_error=0,U_condition=0,condition=1+complex_magnitude(z)+prepared->source_norm;
    for(int j=0;j<3;++j) {
        const double distance=image_distance[j];
        double lower=distance-delta-64*eps*(1+distance);
        if(lower<.6*distance) {
            const double lens_r=prepared->lens_norm[j];
            const Complex relative=std::conj(direction)*(z-lenses[j]);
            const double gap=std::max({box.rlo-lens_r,lens_r-box.rhi,
                std::abs(relative.real())-du,std::abs(relative.imag())-dv});
            lower=std::max(lower,gap-256*eps*(1+box.rhi+lens_r+distance));
        }
        lower_distance[j]=lower;
        if(!(lower>0))return false;
        const auto inv=inverse_conjugate(z-lenses[j]);
        inverses[j]=inv;
        g-=masses[j]*inv;
        if(j){off_axis_source+=masses[j]*inv;
            off_axis_error+=(masses[j]+4*eps)*delta/(lower*distance)+4*eps/lower;
            U+=masses[j]*std::conj(lenses[j])*inv*inv;
            U_error+=(masses[j]+4*eps)*prepared->lens_norm[j]*(delta/distance/lower)*(1/distance+1/lower)+4*eps*prepared->lens_norm[j]/(lower*lower);
            U_condition+=(masses[j]+4*eps)*complex_magnitude(lenses[j])/(lower*lower);}
        error+=(masses[j]+4*eps)*delta/(lower*distance)+4*eps/lower;
        if(j==0 && box.rlo>0) {
            // In this polar frame the primary-only map is h(R)*exp(i*theta),
            // h(R)=R-m0/R. Near the Einstein ring angular motion changes
            // the mapped source by |h(R)| times the chord, even when the
            // image-plane displacement is large. Bound R and angle separately.
            const double dr=std::max(r-box.rlo,box.rhi-r);
            const double upper_mass=masses[0]+4*eps;
            const double radial=(dr+8*eps*(1+r))*(1+upper_mass/box.rlo/r);
            const double amplitude=std::abs(r-masses[0]/r)
                +64*eps*(1+r+upper_mass/r)+4*eps/r;
            const double primary=(radial+amplitude*(chord+64*eps)+4*eps/box.rlo)
                *(1+128*eps);
            error=std::min(error,primary);
        }
        condition+=masses[j]/distance;
    }
    error+=128*eps*condition;U_error+=128*eps*(1+U_condition);
    const double norm=complex_magnitude(g),margin=128*eps*(1+norm+p.rho+error);
    if(norm-error>p.rho+margin || norm+error<p.rho-margin)return true;
    // g_theta=i(z-conj(z)*kappa)=i(g+source-U).
    // Re(conj(g)*i*g)=0 exactly. The primary lens has l_0=0,
    // so it cancels from U instead of inflating a shear error bound.
    const Complex tangent=Complex(0,1)*(source-U);
    const double tangent_error=U_error+256*eps*(1+prepared->source_norm+complex_magnitude(U));
    const double speed=complex_magnitude(tangent);
    // Remove the primary analytically. At a stationary limb,
    // (R-m0/R)e equals the off-axis source plus one of two diameter tips.
    // Enclose that vector in a disk, then invert both primary-image branches.
    // The extra inverse is profitable when the primary dominates. This
    // mass threshold only selects which enclosure to try.
    if(contracted && masses[0]>.98 && speed>tangent_error && box.tlo>=-1 && box.thi<=1) {
        const Complex offset=(p.rho/speed)*(source-U);
        const double pad=(off_axis_error+2*p.rho*(tangent_error/speed)+
            512*eps*(condition+p.rho+prepared->source_norm))*(1+128*eps);
        Box possible{INFINITY,-INFINITY,INFINITY,-INFINITY,box.reciprocal};
        bool usable=true;
        for(int tip:{-1,1}) {
            const Complex v=off_axis_source+double(tip)*offset;
            const double d=complex_magnitude(v),err=pad+128*eps*(1+d);
            if(!(d>2*err) || !std::isfinite(d)){usable=false;break;}
            for(int parity:{-1,1}) {
                const double h=parity*d;
                const double root=complex_magnitude({d,2*std::sqrt(masses[0])});
                const double R=parity>0?.5*(root+h):2*masses[0]/(root+d);
                // |dR/dh| <= 1, |dR/dm| = 1/sqrt(h^2+4m).
                const double re=err+8*eps/std::max(d-err,std::sqrt(masses[0]))+
                    256*eps*(1+R);
                if(!std::isfinite(R+re)){usable=false;break;}
                const double rl=std::max(box.rlo,R-re),rh=std::min(box.rhi,R+re);
                if(rl>rh)continue;
                const double branch=parity*sign,vx=branch*v.real(),vy=branch*v.imag();
                // Each projective chart here covers only its front half.
                if(vx < -err)continue;
                const double denominator=d+vx;
                double tl=box.tlo,th=box.thi;
                if(denominator>4*err && std::isfinite(denominator)) {
                    const double tc=vy/denominator;
                    const double te=err*(1+2*std::abs(tc))/(denominator-2*err)+256*eps*(1+std::abs(tc));
                    tl=std::max(tl,tc-te);th=std::min(th,tc+te);
                }
                if(tl>th)continue;
                possible.rlo=std::min(possible.rlo,rl);possible.rhi=std::max(possible.rhi,rh);
                possible.tlo=std::min(possible.tlo,tl);possible.thi=std::max(possible.thi,th);
            }
        }
        if(usable) {
            if(possible.rlo>possible.rhi)return true;
            if(possible.rlo<possible.rhi && possible.tlo<possible.thi &&
               (possible.rhi-possible.rlo)*(possible.thi-possible.tlo)<
               .3*(box.rhi-box.rlo)*(box.thi-box.tlo)){*contracted=possible;return false;}
        }
    }
    const double product=std::real(std::conj(g)*tangent);
    const double product_error=norm*tangent_error+speed*error+error*tangent_error+
        256*eps*(1+norm*speed);
    if(std::abs(product)>product_error)return true;

    // At a boundary event |g|=rho and g is perpendicular to H=i(source-U).
    // In the radial frame Im(conj(e)*g) contains no R-m0/R term. Thus
    // |Im(conj(e)*g)| |H| = rho |Re(conj(e)*H)|. This second test remains
    // sharp near a primary Einstein ring, where a wide radial box makes the
    // direct dot-product bound inconclusive.
    const double perp=std::abs(std::imag(std::conj(direction)*off_axis_source));
    const double parallel=std::abs(std::real(std::conj(direction)*tangent));
    const double direction_error=chord+128*eps;
    const double perp_error=(off_axis_error+complex_magnitude(off_axis_source)*direction_error+
        256*eps*condition)*(1+128*eps);
    const double parallel_error=(tangent_error+speed*direction_error+
        256*eps*(1+speed))*(1+128*eps);
    const double left=perp*speed,right=p.rho*parallel;
    const double uncertainty=perp_error*(speed+tangent_error)+perp*tangent_error+
        p.rho*parallel_error;
    if(std::abs(left-right)>uncertainty*(1+256*eps)+256*eps*(1+left+right))return true;
    if(!(speed>tangent_error))return false;

    // At an event g lies at one of the two ends of the source diameter
    // parallel to source-U. Enclose the lens map about the image-box centre
    // and test both ends against its radial and transverse displacements.
    const Complex axis=(source-U)/speed;
    Complex shear{};
    double remainder=0,shear_scale=1;
    for(int j=0;j<3;++j) {
        const double d=image_distance[j],lower=lower_distance[j];
        const double mass=masses[j]+4*eps;
        shear+=masses[j]*inverses[j]*inverses[j];
        // Exact remainder of the first-order expansion of 1/conj(z-l_j).
        remainder+=(mass/lower)*(delta/d)*(delta/d)+4*eps/lower;
        shear_scale+=mass/d/d;
    }
    const Complex applied=shear*std::conj(direction);
    const Complex radial=direction+applied, transverse=Complex(0,1)*(direction-applied);
    const Complex major=complex_magnitude(radial)*du>=complex_magnitude(transverse)*dv?radial:transverse;
    const double length=complex_magnitude(major);
    if(!(length>0 && std::isfinite(length)))return false;
    const Complex along=major/length;
    const auto project=[&](Complex v){return std::conj(along)*v;};
    const auto jr=project(radial),jt=project(transverse);
    const double pad=remainder+2*p.rho*(tangent_error/speed)
        +1024*eps*(condition+p.rho+shear_scale*(1+box.rhi+du+dv));
    const double width_x=(std::abs(jr.real())*du+std::abs(jt.real())*dv+pad)*(1+128*eps);
    const double width_y=(std::abs(jr.imag())*du+std::abs(jt.imag())*dv+pad)*(1+128*eps);
    const auto center=project(g),offset=project(p.rho*axis);
    auto outside=[&](Complex v) {
        return std::isfinite(v.real()) && std::isfinite(v.imag()) &&
            (std::abs(v.real())>width_x || std::abs(v.imag())>width_y);
    };
    const bool plus=outside(center-offset),minus=outside(center+offset);
    if(plus && minus)return true;
    if(!contracted || !std::isfinite(pad))return false;

    const double determinant=std::imag(std::conj(radial)*transverse);
    // A condition estimate only decides whether to attempt contraction.
    // Ill-conditioned cells keep the ordinary subdivision proof.
    const double absolute_det=std::abs(determinant);
    if(!(absolute_det>0) || (pad*complex_magnitude(transverse)>absolute_det*du*.45 &&
                            pad*complex_magnitude(radial)>absolute_det*dv*.45))return false;
    // Invert the linearized image map for each remaining diameter tip.
    // The nonlinear remainder and the motion of the tip are both enclosed
    // by pad. Near a singular Jacobian retain the uncontracted cell.
    const auto cross=[](Complex a,Complex b) {
        return I(a.real())*I(b.imag())-I(a.imag())*I(b.real());
    };
    const auto det=cross(radial,transverse);
    if(std::abs(det.sign())!=1)return false;
    const auto inverse=physical_detail::reciprocal_value(det);
    const double pad_bound=pad*(1+256*eps);
    const auto xerr=I(-pad_bound,pad_bound)*I(complex_magnitude(transverse)),
               yerr=I(-pad_bound,pad_bound)*I(complex_magnitude(radial));
    const double guard=256*eps*(1+box.rhi),curvature=.5*box.rhi*chord*chord+guard;
    const I x_domain(box.rlo-r-curvature-guard,box.rhi-r+guard),y_domain(-dv-guard,dv+guard);
    const auto overlap=[](I a,I b){return I(std::max(a.lo,b.lo),std::min(a.hi,b.hi));};
    Box result{INFINITY,-INFINITY,INFINITY,-INFINITY,box.reciprocal};
    for(int side:{-1,1}) {
        if(side==1?plus:minus)continue;
        const Complex target=(side*p.rho)*axis-g;
        const I raw_x=(cross(target,transverse)+xerr)*inverse;
        const I raw_y=(cross(radial,target)+yerr)*inverse;
        if(!raw_x.valid() || !raw_y.valid())return false;
        const I x=overlap(x_domain,raw_x),y=overlap(y_domain,raw_y);
        if(x.lo>x.hi || y.lo>y.hi)continue;
        if(!x.valid() || !y.valid())return false;
        // r+x = R' cos(delta theta), y = R' sin(delta theta).
        const I parallel=I(r)+x;
        const I rr=overlap(I(box.rlo,box.rhi),parallel+I(-guard,curvature));
        if(rr.lo>rr.hi)continue;
        I tt(box.tlo,box.thi);
        const I den=rr+parallel+I(-guard,guard);
        if(den.lo>0) {
            const I q=(y+I(-guard,guard))*physical_detail::reciprocal_value(den);
            const I divisor=I(1)-I(t)*q;
            if(divisor.lo>0) {
                const I coordinate=(I(t)+q)*physical_detail::reciprocal_value(divisor)+I(-256*eps,256*eps);
                if(coordinate.valid())tt=overlap(tt,coordinate);
            }
        }
        if(tt.lo>tt.hi)continue;
        result.rlo=std::min(result.rlo,rr.lo);result.rhi=std::max(result.rhi,rr.hi);
        result.tlo=std::min(result.tlo,tt.lo);result.thi=std::max(result.thi,tt.hi);
    }
    if(result.rlo>result.rhi)return true;
    if(result.rlo<result.rhi && result.tlo<result.thi &&
       (result.rhi-result.rlo)*(result.thi-result.tlo)<.4*(box.rhi-box.rlo)*(box.thi-box.tlo))
        *contracted=result;
    return false;

}
// The angular polynomial coefficients depend only on R. Keep the same
// factored arithmetic for their interval values and radial derivatives.
template<class S> struct RadialDerivative {
    S value{},derivative{};
    RadialDerivative()=default;
    RadialDerivative(double x):value(x){}
    RadialDerivative(S v,S d):value(v),derivative(d){}
    friend RadialDerivative operator+(RadialDerivative a,RadialDerivative b) {
        return {a.value+b.value,a.derivative+b.derivative};
    }
    friend RadialDerivative operator-(RadialDerivative a) {
        return {-a.value,-a.derivative};
    }
    friend RadialDerivative operator-(RadialDerivative a,RadialDerivative b) {return a+(-b);}
    friend RadialDerivative operator*(RadialDerivative a,RadialDerivative b) {
        return {a.value*b.value,a.derivative*b.value+a.value*b.derivative};
    }
};
template<class T> struct BasicCoefficientCache {
    using Scalar=angular_detail::I<T>;
    const Geometry& geometry;
    std::map<std::pair<double,double>,std::array<Jet<Scalar>,7>> entries;
    std::optional<std::array<surface_detail::RadialPolynomialT<T>,7>> polynomial;
    std::map<std::pair<double,double>,std::array<Jet<Scalar>,7>> direct_entries;
    std::array<Jet<Scalar>,7> factored(double rlo,double rhi) const {
        const RadialDerivative<Scalar> radius(Scalar(T(rlo),T(rhi)),Scalar(1));
        const auto c=boundary_from_radius(radius,geometry).coefficients;
        std::array<Jet<Scalar>,7> out{};
        for(int j=0;j<7;++j){out[j].v=c[j].value;out[j].r=c[j].derivative;}
        return out;
    }
    std::array<Jet<Scalar>,7> coefficients(double rlo,double rhi) {
        if constexpr(std::is_same_v<T,double>)
            return factored(rlo,rhi);
        else {
            // Geometry-only quad coefficients are reconstructed once. Horner
            // evaluation carries their enclosures into each local R interval.
            if(!polynomial) {
                surface_detail::RadialPolynomialT<T> R;R.degree=1;R.c[1]=Scalar(1);
                polynomial=boundary_from_radius(R,geometry).coefficients;
            }
            std::array<Jet<Scalar>,7> out{};const Scalar R{T(rlo),T(rhi)};
            for(int k=0;k<7;++k) {
                const auto& c=(*polynomial)[k];Scalar value=c.c[c.degree],derivative;
                for(int j=c.degree-1;j>=0;--j) {
                    derivative=derivative*R+value;value=value*R+c.c[j];
                }
                out[k].v=value;out[k].r=derivative;
            }
            return out;
        }
    }
    Jet<Scalar> evaluate(double rlo,double rhi,Scalar t,bool reciprocal_chart,bool direct=false) {
        auto key=std::make_pair(rlo,rhi);
        auto& active=direct?direct_entries:entries;
        auto found=active.find(key);
        if(found==active.end())found=active.emplace(key,direct?
            factored(rlo,rhi):
            coefficients(rlo,rhi)).first;
        Jet<Scalar> x(t,Scalar(0),Scalar(1)),value;
        for(int j=6;j>=0;--j) {
            auto c=reciprocal_chart?found->second[6-j]:found->second[j];
            if(reciprocal_chart && j%2)c=-c;
            value=value*x+c;
        }
        return value;
    }
};
using CoefficientCache=BasicCoefficientCache<double>;
struct Proof {int kind=0;I r{},t{};}; // 0 unknown,1 excluded,2 unique
inline Proof proof_from_jets(const Box& b,Jet<I> f,Jet<I> fm) {
    Proof out;
    if(f.v.sign()==1 || f.v.sign()==-1 || f.t.sign()==1 || f.t.sign()==-1) {
        out.kind=1;return out;
    }
    const double rm=b.rlo+(b.rhi-b.rlo)*.5,tm=b.tlo+(b.thi-b.tlo)*.5;
    auto center=[](I x){return x.lo+(x.hi-x.lo)*.5;};
    const double mr=center(fm.r),mt=center(fm.t),mtt=center(fm.tt),mrt=center(fm.rt);
    const double det=mr*mtt-mt*mrt;
    if(det==0 || !std::isfinite(det))return out;
    const double inv[2][2]={{mtt/det,-mt/det},{-mrt/det,mr/det}};
    const I jac[2][2]={{f.r,f.t},{f.rt,f.tt}};

    const I values[2]={fm.v,fm.t};
    I k[2]={I(rm),I(tm)};
    const I delta[2]={I(b.rlo,b.rhi)-I(rm),I(b.tlo,b.thi)-I(tm)};
    for(int row=0;row<2;++row) {
        for(int j=0;j<2;++j)k[row]=k[row]-I(inv[row][j])*values[j];
        for(int col=0;col<2;++col) {
            I residual(row==col?1:0);
            for(int j=0;j<2;++j)residual=residual-I(inv[row][j])*jac[j][col];
            k[row]=k[row]+residual*delta[col];
        }
    }
    if(k[0].hi<b.rlo || k[0].lo>b.rhi || k[1].hi<b.tlo || k[1].lo>b.thi) {
        out.kind=1;return out;
    }
    out.r={std::max(k[0].lo,b.rlo),std::min(k[0].hi,b.rhi)};
    out.t={std::max(k[1].lo,b.tlo),std::min(k[1].hi,b.thi)};
    if(k[0].lo>b.rlo && k[0].hi<b.rhi && k[1].lo>b.tlo && k[1].hi<b.thi)
        out.kind=2;
    return out;
}
inline Proof prove(const Box& b,const Geometry& p,CoefficientCache* cache=nullptr) {
    auto f=cache?cache->evaluate(b.rlo,b.rhi,I(b.tlo,b.thi),b.reciprocal):
        evaluate(I(b.rlo,b.rhi),I(b.tlo,b.thi),p,b.reciprocal);
    if(f.v.sign()==1 || f.v.sign()==-1 || f.t.sign()==1 || f.t.sign()==-1)return {1,{},{}};
    const double rm=b.rlo+(b.rhi-b.rlo)*.5,tm=b.tlo+(b.thi-b.tlo)*.5;
    auto fm=cache?cache->evaluate(rm,rm,I(tm),b.reciprocal):evaluate(I(rm),I(tm),p,b.reciprocal);
    return proof_from_jets(b,f,fm);
}
inline Proof prove_physical(const Box& b,const Geometry& p) {
    auto f=physical_detail::Ray<I>(I(b.rlo,b.rhi),p).event_jet(I(b.tlo,b.thi),b.reciprocal,true);
    if(!f)return {};
    if(f->v.sign()==1 || f->v.sign()==-1 || f->t.sign()==1 || f->t.sign()==-1)return {1,{},{}};
    const double rm=b.rlo+(b.rhi-b.rlo)*.5,tm=b.tlo+(b.thi-b.tlo)*.5;
    const auto center=physical_detail::Ray<I>(I(rm),p).angular(I(tm),b.reciprocal);
    if(!center)return {};
    // A Krawczyk preconditioner may be any represented real matrix. The
    // already enclosed full-box Jacobian supplies one; only the residual at
    // the centre needs a fresh enclosure, not another second-order jet.
    auto midpoint=[](I x){return I(x.lo+(x.hi-x.lo)*.5);};
    Jet<I> fm;fm.v=center->v;fm.t=center->t;
    fm.r=midpoint(f->r);fm.rt=midpoint(f->rt);fm.tt=midpoint(f->tt);
    return proof_from_jets(b,*f,fm);
}
inline Proof prove_surface(const Box& box,const surface_detail::Tensor& polynomial,const Geometry&) {
    namespace surface=surface_detail;
    // A broad box cannot use the local inclusion test. Do not finish
    // computing coefficient extrema after its exclusion sign is lost.
    if(box.rhi-box.rlo>.02*(1+box.rhi) || box.thi-box.tlo>.05) {
        if(surface::excludes_zero(polynomial) || surface::excludes_zero(polynomial,true))return {1,{},{}};
        return {};
    }
    const I value=surface::hull(polynomial);
    if(value.sign()==1 || value.sign()==-1)return {1,{},{}};
    const I difference=surface::difference_hull(polynomial,1);
    // Positive chart widths/degrees do not change this exclusion sign.
    // Scale derivatives only when a local uniqueness test actually needs them.
    if(difference.sign()==1 || difference.sign()==-1)return {1,{},{}};
    const I rw=I(box.rhi)-I(box.rlo),tw=I(box.thi)-I(box.tlo);
    const I scale_t=I(polynomial.nt)*surface::inverse(tw),scale_r=I(polynomial.nr)*surface::inverse(rw);
    const I theta=difference*scale_t;
    const auto gradients=surface::derivative_hulls(polynomial);
    Jet<I> f,fm;f.v=value;f.t=theta;f.r=gradients.r*scale_r;
    f.rt=gradients.rt*scale_r*scale_t;
    f.tt=gradients.tt*scale_t*I(polynomial.nt-1)*surface::inverse(tw);
    const double rm=box.rlo+(box.rhi-box.rlo)*.5,tm=box.tlo+(box.thi-box.tlo)*.5;
    const I x=surface::fraction(rm,box.rlo,box.rhi),y=surface::fraction(tm,box.tlo,box.thi);
    const auto centre=surface::centre_value_theta(polynomial,x,y,tw);
    fm.v=centre.first;fm.t=centre.second;
    // Only the residual and full-box Jacobian need enclosures. The inverse
    // used as a Krawczyk preconditioner may be any represented real matrix;
    // evaluating three more interval tensors at the centre buys no proof.
    // Reuse the already enclosed Jacobian. Its midpoint is a valid
    // Krawczyk preconditioner; rebuilding the entire cleared polynomial at
    // this point is unnecessary and dominates small-box discovery cost.
    auto midpoint=[](I a){return I(a.lo+(a.hi-a.lo)*.5);};
    fm.r=midpoint(f.r);fm.rt=midpoint(f.rt);fm.tt=midpoint(f.tt);
    return proof_from_jets(box,f,fm);
}
inline Proof prove_quad(const Box& b,const Geometry& p,BasicCoefficientCache<__float128>* cache=nullptr,bool natural=false,bool physical_equations=true) {
    using Q=__float128;using IQ=angular_detail::I<Q>;
    Proof out;
    auto direct=physical_equations?physical(IQ(Q(b.rlo),Q(b.rhi)),IQ(Q(b.tlo),Q(b.thi)),p,b.reciprocal):std::nullopt;
    auto f=direct?*direct:cache?cache->evaluate(b.rlo,b.rhi,IQ(Q(b.tlo),Q(b.thi)),b.reciprocal,natural):
        evaluate(IQ(Q(b.rlo),Q(b.rhi)),IQ(Q(b.tlo),Q(b.thi)),p,b.reciprocal);
    if(f.v.sign()==1 || f.v.sign()==-1 || f.t.sign()==1 || f.t.sign()==-1) {
        out.kind=1;return out;
    }
    const Q rm=(Q(b.rlo)+Q(b.rhi))/2,tm=(Q(b.tlo)+Q(b.thi))/2;
    // The centre used by this proof must equal the centre used by delta.
    // Binary64 midpoints are exact inputs in the promoted arithmetic.
    const double rd=b.rlo+(b.rhi-b.rlo)*.5;
    const Q rc=cache?Q(rd):rm;
    Jet<IQ> fm;
    if(direct) {
        const auto center=physical_detail::Ray<IQ>(IQ(rc,rc),p).angular(IQ(tm,tm),b.reciprocal);
        if(!center)return out;
        fm=f;fm.v=center->v;fm.t=center->t;
    } else {
        fm=cache?cache->evaluate(rd,rd,IQ(tm,tm),b.reciprocal,natural):
            evaluate(IQ(rm,rm),IQ(tm,tm),p,b.reciprocal);
    }
    auto center=[](IQ a){return (a.lo+a.hi)/2;};
    const Q mr=center(fm.r),mt=center(fm.t),mtt=center(fm.tt),mrt=center(fm.rt),det=mr*mtt-mt*mrt;
    if(det==0 || !finiteq(det))return out;
    const Q inv[2][2]={{mtt/det,-mt/det},{-mrt/det,mr/det}};
    const IQ jac[2][2]={{f.r,f.t},{f.rt,f.tt}},values[2]={fm.v,fm.t};
    const IQ delta[2]={IQ(Q(b.rlo),Q(b.rhi))-IQ(rc,rc),IQ(Q(b.tlo),Q(b.thi))-IQ(tm,tm)};
    IQ k[2]={IQ(rc,rc),IQ(tm,tm)};
    for(int row=0;row<2;++row) {
        for(int j=0;j<2;++j)k[row]=k[row]-IQ(inv[row][j],inv[row][j])*values[j];
        for(int col=0;col<2;++col) {
            IQ residual(row==col?1:0);
            for(int j=0;j<2;++j)residual=residual-IQ(inv[row][j],inv[row][j])*jac[j][col];
            k[row]=k[row]+residual*delta[col];
        }
    }
    if(k[0].hi<Q(b.rlo) || k[0].lo>Q(b.rhi) || k[1].hi<Q(b.tlo) || k[1].lo>Q(b.thi)) {
        out.kind=1;return out;
    }
    auto lower=[](Q x){double d=double(x);return Q(d)>x?std::nextafter(d,-INFINITY):d;};
    auto upper=[](Q x){double d=double(x);return Q(d)<x?std::nextafter(d,INFINITY):d;};
    out.r={std::max(b.rlo,lower(k[0].lo)),std::min(b.rhi,upper(k[0].hi))};
    out.t={std::max(b.tlo,lower(k[1].lo)),std::min(b.thi,upper(k[1].hi))};
    if(k[0].lo>Q(b.rlo) && k[0].hi<Q(b.rhi) && k[1].lo>Q(b.tlo) && k[1].hi<Q(b.thi))out.kind=2;
    // Polynomial coefficients cheaply exclude boxes through the full
    // Krawczyk test. Keep the local construction for event acceptance.
    if(cache && !natural && !direct)return prove_quad(b,p,cache,true,physical_equations);
    return out;
}

}
struct RadialEvent {
    double radius=0,coordinate=0,radius_lo=0,radius_hi=0;
    bool reciprocal=false;
    double coordinate_lo=0,coordinate_hi=0;
};
inline bool refine_event_center(RadialEvent& event,const Geometry& p,bool double_first=false,bool* used_quad=nullptr,bool radius_precision_only=false) {
    if(used_quad)*used_quad=false;
    // Keep the original certified box. A successful binary64 Newton step
    // is only a proposal; accept it after a fresh physical interval proof
    // contracts to the arithmetic floor. Otherwise retain the proposal for
    // the original promoted solve. No other events or samples are discarded.
    double proposal_r=event.radius,proposal_t=event.coordinate;
    if(double_first) {
        double r=event.radius,t=event.coordinate;
        bool valid=true;
        for(int k=0;k<8;++k) {
            const auto f=event_detail::physical(r,t,p,event.reciprocal);
            if(!f){valid=false;break;}
            const double det=f->r*f->tt-f->t*f->rt;
            if(!std::isfinite(det) || det==0){valid=false;break;}
            const double dr=(-f->v*f->tt+f->t*f->t)/det,dt=(-f->r*f->t+f->rt*f->v)/det;
            if(!std::isfinite(dr) || !std::isfinite(dt)){valid=false;break;}
            r+=dr;t+=dt;
            if(r<event.radius_lo || r>event.radius_hi || t<event.coordinate_lo || t>event.coordinate_hi){valid=false;break;}
            if(std::abs(dr)+std::abs(dt)<8*std::numeric_limits<double>::epsilon()*(1+std::abs(r)+std::abs(t)))break;
        }
        if(valid) {
            proposal_r=r;proposal_t=t;
            const double eps=std::numeric_limits<double>::epsilon();
            const double rw=128*eps*(1+std::abs(r));
            // For a value-controlled material map it is the radial cut that
            // needs arithmetic-floor accuracy. The stationary coordinate is
            // a continuation seed; retain its certified enclosure rather than
            // promoting merely to squeeze that enclosure to binary64 ulps.
            const double tw=(radius_precision_only?1e-10:128*eps)*(1+std::abs(t));
            event_detail::Box box{std::max(event.radius_lo,r-rw),std::min(event.radius_hi,r+rw),
                std::max(event.coordinate_lo,t-tw),std::min(event.coordinate_hi,t+tw),event.reciprocal};
            const auto proof=event_detail::prove_physical(box,p);
            if(proof.kind==2 && proof.r.hi-proof.r.lo<=rw && proof.t.hi-proof.t.lo<=tw) {
                event={std::clamp(r,proof.r.lo,proof.r.hi),std::clamp(t,proof.t.lo,proof.t.hi),
                    proof.r.lo,proof.r.hi,event.reciprocal,proof.t.lo,proof.t.hi};
                return true;
            }
        }
    }
    using namespace event_detail;
    if(used_quad)*used_quad=true;
    __float128 r=proposal_r,t=proposal_t;
    for(int j=0;j<16;++j) {
        // The physical jet has the same stationary roots away from lens
        // poles, without rebuilding the cleared degree-six polynomial.
        const auto direct=physical(r,t,p,event.reciprocal);
        const auto f=direct?*direct:evaluate(r,t,p,event.reciprocal);
        const auto det=f.r*f.tt-f.t*f.rt;
        if(det==0 || !finiteq(det))return false;
        const auto dr=(-f.v*f.tt+f.t*f.t)/det,dt=(-f.r*f.t+f.rt*f.v)/det;
        r+=dr;t+=dt;
        if(fabsq(dr)+fabsq(dt)<(__float128)1e-29)break;
    }
    if(r<(__float128)event.radius_lo || r>(__float128)event.radius_hi ||
       t<(__float128)event.coordinate_lo || t>(__float128)event.coordinate_hi)return false;
    const double rd=double(r),td=double(t);
    const double rw=32*std::numeric_limits<double>::epsilon()*(1+std::abs(rd));
    const double tw=32*std::numeric_limits<double>::epsilon()*(1+std::abs(td));
    Box box{std::max(event.radius_lo,rd-rw),std::min(event.radius_hi,rd+rw),
        std::max(event.coordinate_lo,td-tw),std::min(event.coordinate_hi,td+tw),event.reciprocal};
    auto proof=prove_quad(box,p,nullptr,false,true);
    if(proof.kind!=2)proof=prove_quad(box,p,nullptr,false,false);
    // A fixed absolute arithmetic-floor box can contain both folds of a
    // tiny image. Newton supplies only a proposal: shrink its neighbourhood
    // inside the original certified box and require a fresh uniqueness proof.
    // Keep an outward binary64 enclosure even after the proposed width rounds
    // away. No event or image contribution is removed by this local repair.
    Box previous=box;
    for(int level=1;proof.kind!=2 && level<=12;++level) {
        const double scale=std::ldexp(1.,-4*level);
        Box local{std::max(event.radius_lo,std::min(rd-rw*scale,std::nextafter(rd,-INFINITY))),
            std::min(event.radius_hi,std::max(rd+rw*scale,std::nextafter(rd,INFINITY))),
            std::max(event.coordinate_lo,std::min(td-tw*scale,std::nextafter(td,-INFINITY))),
            std::min(event.coordinate_hi,std::max(td+tw*scale,std::nextafter(td,INFINITY))),event.reciprocal};
        if(local.rlo==previous.rlo && local.rhi==previous.rhi &&
           local.tlo==previous.tlo && local.thi==previous.thi)break;
        previous=local;
        proof=prove_quad(local,p,nullptr,false,true);
    }
    if(proof.kind!=2)return false;
    event={rd,td,proof.r.lo,proof.r.hi,event.reciprocal,proof.t.lo,proof.t.hi};return true;
}
struct RadialEventPlan {
    std::shared_ptr<point_polynomial_detail::LensContext> lens_context;
    std::vector<RadialEvent> events;
    std::vector<ImageExclusion> excluded_images;
    std::vector<ImageExclusion> component_images;
    std::optional<ImageExclusion> included_image;
    int expected_images=0;
    int resultant_nodes=0,resultant_events=0,resultant_solves=0;
    double bounded_component_value=0,bounded_component_error=0;
    bool completed=false;
    // A contraction proof can cover a local component without isolating its folds.
    bool unisolated_folds=false;
    const char* reason="not_started";
    int boxes=0,contractions=0,quad_refinements=0,quad_boxes=0,double_refinements=0;
    std::array<double,4> last_box{};
    std::array<double,8> identity{};
    double rmin=0,rmax=0;
};
// Individually certified full-source components. The global complement
// search, rather than a count of point-source seeds, supplies completeness.
struct PartialComponents {
    std::array<double,8> identity{};
    std::vector<ImageExclusion> images;
    std::vector<RadialEvent> events;
};
// Internal proof packet: prepared contains the certified domain and events;
// every radial interval outside unresolved has been excluded or assigned by
// the support resultant. Identity is enforced just as for supplied plans.
struct CertifiedRadialCover {
    RadialEventPlan prepared;
    std::vector<std::pair<double,double>> unresolved;
};
namespace event_detail {
inline bool radial_domain(const Geometry&p,RadialEventPlan&out) {
    const auto masses=p.masses();const auto lenses=p.lenses();
    const double sum=p.m1+p.m2,bvirtual=sum-p.m1;
    const double sum_error=(p.m1-(sum-bvirtual))+(p.m2-bvirtual);
    const bool positive_primary=sum<1 || (sum==1 && sum_error<0);
    if(!positive_primary || !(masses[0]>0 && masses[1]>0 && masses[2]>0) || p.a==0 ||
       std::abs(lenses[2])==0 || std::abs(lenses[1]-lenses[2])==0) {
        out.reason="degenerate_geometry";return false;
    }
    const double W=std::hypot(p.X,p.Y)+p.rho;
    const double L=std::max(std::abs(lenses[1]),std::abs(lenses[2]));
    // Strict triangle-inequality bounds, padded outward. Halving the inner
    // radius eventually isolates the positive-mass primary pole.
    out.rmax=std::nextafter(.5*(L+W+std::hypot(L-W,2.))+1e-12*(1+L+W),INFINITY);
    out.rmin=.25*std::min(std::abs(lenses[1]),std::abs(lenses[2]));
    bool inner_bound=false;
    for(int j=0;j<1000;++j) {
        const double rhs=out.rmin+W+masses[1]/(std::abs(lenses[1])-out.rmin)+
            masses[2]/(std::abs(lenses[2])-out.rmin);
        if(masses[0]/out.rmin>rhs*(1+1e-12)){inner_bound=true;break;}
        out.rmin*=.5;
    }
    if(!inner_bound || !(out.rmin>0 && out.rmax>out.rmin) || !std::isfinite(out.rmax)){out.reason="domain_bound";return false;}
    // Refine the same triangle-inequality domain bounds before constructing
    // a wide, poorly conditioned Bernstein tensor. Each accepted endpoint
    // proves exclusion of the entire remaining inner/outer radial domain.
    const double guard=1e-12;
    const double Wupper=W+guard*(1+W);
    std::array<double,3> distance{},upper_mass{};
    for(int j=0;j<3;++j){distance[j]=std::abs(lenses[j]);upper_mass[j]=masses[j]+4*std::numeric_limits<double>::epsilon();}
    {
        double lo=std::max(L,W),hi=out.rmax;
        for(int iteration=0;iteration<24;++iteration){
            const double mid=lo+(hi-lo)*.5;double deflection=0;bool valid=true;
            for(int j=0;j<3;++j){double d=mid-distance[j]-guard*(1+distance[j]);if(!(d>0)){valid=false;break;}deflection+=upper_mass[j]/d;}
            if(valid && mid-Wupper>deflection*(1+guard)+guard)hi=mid;else lo=mid;
        }
        out.rmax=hi;
    }
    {
        double lo=out.rmin,hi=std::min(distance[1],distance[2]);
        const double primary_lower=masses[0]-4*std::numeric_limits<double>::epsilon();
        if(primary_lower>0)for(int iteration=0;iteration<24;++iteration){
            const double mid=lo+(hi-lo)*.5;double rhs=mid+Wupper;bool valid=true;
            for(int j=1;j<3;++j){double d=distance[j]-mid-guard*(1+distance[j]);if(!(d>0)){valid=false;break;}rhs+=upper_mass[j]/d;}
            if(valid && primary_lower/mid>rhs*(1+guard)+guard)lo=mid;else hi=mid;
        }
        out.rmin=lo;
    }
    return true;
}
inline bool inside_component(const Box&box,const std::vector<ImageExclusion>&images) {
    if(images.empty())return false;
    const double eps=std::numeric_limits<double>::epsilon();
    const double r=box.rlo+(box.rhi-box.rlo)*.5,t=box.tlo+(box.thi-box.tlo)*.5;
    const double sign=box.reciprocal?-1:1,den=1+t*t;
    const Complex z=sign*r*Complex((1-t*t)/den,2*t/den);
    const double chord=std::max(2*std::abs(box.tlo-t)/std::hypot(1.,box.tlo),2*std::abs(box.thi-t)/std::hypot(1.,box.thi))/std::hypot(1.,t);
    const double delta=std::hypot(.5*(box.rhi-box.rlo),std::sqrt(box.rhi)*std::sqrt(r)*chord)+128*eps*(1+box.rhi);
    for(auto image:images) {
        const double distance=std::abs(z-image.center);
        if(distance+delta+128*eps*(1+distance+std::abs(image.center))<image.radius)return true;
    }
    return false;
}
}
// Boundary images of every nearby source lie in the image of a larger
// source disk. Negative Bernstein coefficients exclude an entire rectangle
// from that disk. Keep the remaining rectangles as starting boxes only.
struct NearbySourceCover {
    Geometry anchor{};
    double rlo=0,rhi=0;
    std::vector<event_detail::Box> boxes;
    int visits=0;
    bool nearby(const Geometry& p) const {
        if(anchor.a!=p.a || anchor.bx!=p.bx || anchor.by!=p.by ||
           anchor.m1!=p.m1 || anchor.m2!=p.m2 || anchor.rho!=p.rho)return false;
        const auto dx=surface_detail::I(p.X)-surface_detail::I(anchor.X);
        const auto dy=surface_detail::I(p.Y)-surface_detail::I(anchor.Y);
        const auto displacement=surface_detail::I(std::max(-dx.lo,dx.hi))
            +surface_detail::I(std::max(-dy.lo,dy.hi));
        return displacement.hi<=p.rho;
    }
    void prepare(const Geometry& p) {
        anchor=p;boxes.clear();visits=0;
        auto disk=p;disk.rho=2*p.rho;
        RadialEventPlan domain;
        if(!std::isfinite(disk.rho) || !event_detail::radial_domain(disk,domain))return;
        rlo=domain.rmin;rhi=domain.rmax;
        const auto base=surface_detail::radial_base(rlo,rhi,disk);
        auto visit=[&](auto& self,event_detail::Box b,const surface_detail::Tensor& tensor,int depth)->void {
            if(visits>=2048){boxes.push_back(b);return;}
            ++visits;
            bool outside=true;
            for(const auto& row:tensor.b)for(const auto coefficient:row)
                outside=outside && coefficient.lo<=coefficient.hi && coefficient.hi<0;
            if(outside)return;
            if(depth==14){boxes.push_back(b);return;}
            const bool angular=depth%2==0;
            const double lo=angular?b.tlo:b.rlo,hi=angular?b.thi:b.rhi;
            const double mid=lo+(hi-lo)*.5;
            if(!(mid>lo && mid<hi)){boxes.push_back(b);return;}
            const auto halves=surface_detail::split(tensor,angular,surface_detail::fraction(mid,lo,hi));
            auto left=b,right=b;
            (angular?left.thi:left.rhi)=mid;
            (angular?right.tlo:right.rlo)=mid;
            self(self,left,halves.first,depth+1);
            self(self,right,halves.second,depth+1);
        };
        for(bool chart:{false,true})visit(visit,{rlo,rhi,-1,1,chart},surface_detail::angular_chart(base,chart),0);
        // Adjacent pixels often describe the same elongated image. Merging
        // exact shared edges avoids rebuilding an event tensor for each pixel.
        for(bool angular:{false,true}) {
            auto key=[&](const auto& b) {
                if(angular)return std::make_tuple(b.reciprocal,b.rlo,b.rhi,b.tlo);
                return std::make_tuple(b.reciprocal,b.tlo,b.thi,b.rlo);
            };
            std::sort(boxes.begin(),boxes.end(),[&](const auto& a,const auto& b){return key(a)<key(b);});
            std::size_t count=0;
            for(auto b:boxes) {
                if(count && boxes[count-1].reciprocal==b.reciprocal) {
                    auto& a=boxes[count-1];
                    if(angular && a.rlo==b.rlo && a.rhi==b.rhi && a.thi==b.tlo){a.thi=b.thi;continue;}
                    if(!angular && a.tlo==b.tlo && a.thi==b.thi && a.rhi==b.rlo){a.rhi=b.rhi;continue;}
                }
                boxes[count++]=b;
            }
            boxes.resize(count);
        }
    }
};

inline RadialEventPlan radial_event_plan(const Geometry& p,int max_boxes=100000,
                                       double flux_tolerance=0,const PartialComponents* components=nullptr,
                                       const CertifiedRadialCover* radial_cover=nullptr) {
    using namespace event_detail;
    RadialEventPlan out;p.validate();
    out.identity={p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};
    if(!angular_detail::enclosure::environment_ok()) {out.reason="floating_environment";return out;}
    const bool narrowed=radial_cover && radial_cover->prepared.identity==out.identity &&
        std::isfinite(radial_cover->prepared.rmax) &&
        radial_cover->prepared.rmin>0 && radial_cover->prepared.rmax>radial_cover->prepared.rmin;
    if(narrowed)out=radial_cover->prepared;
    else if(!radial_domain(p,out))return out;
    const bool partial=components && components->identity==out.identity && !components->images.empty();
    if(partial){out.events.insert(out.events.end(),components->events.begin(),components->events.end());out.component_images=components->images;}

    struct Node {Box box;surface_detail::Tensor polynomial;};
    std::vector<Node> pending;
    std::vector<Box> covered;
    auto contains=[](const Box& outer,const Box& inner) {
        return outer.reciprocal==inner.reciprocal && outer.rlo<=inner.rlo && outer.rhi>=inner.rhi &&
               outer.tlo<=inner.tlo && outer.thi>=inner.thi;
    };
    auto cover=[&](const Box& region) {
        covered.push_back(region);
        if(region.tlo>0 || region.thi<0) {
            // Use inward rounding: every point in this opposite-chart box
            // must map into the already-proved unique-root region.
            const double lo=angular_detail::enclosure::up(-1/region.tlo);
            const double hi=angular_detail::enclosure::down(-1/region.thi);
            if(hi>lo)covered.push_back({region.rlo,region.rhi,lo,hi,!region.reciprocal});
        }
    };
    std::vector<Box> recorded_bounds;
    auto record_bounds=[&](Box b) {
        recorded_bounds.push_back(b);
        if(b.tlo>0 || b.thi<0) {
            const double lo=angular_detail::enclosure::down(-1/b.tlo);
            const double hi=angular_detail::enclosure::up(-1/b.thi);
            recorded_bounds.push_back({b.rlo,b.rhi,lo,hi,!b.reciprocal});
        }
    };
    auto recorded_root=[&](const Box& b) {
        if(std::any_of(covered.begin(),covered.end(),[&](const auto& known){return contains(known,b);}))return true;
        for(const auto& old:recorded_bounds)if(old.reciprocal==b.reciprocal && old.rlo<=b.rhi &&
            old.rhi>=b.rlo && old.tlo<=b.thi && old.thi>=b.tlo) {
            const Box joined{std::min(old.rlo,b.rlo),std::max(old.rhi,b.rhi),
                std::min(old.tlo,b.tlo),std::max(old.thi,b.thi),b.reciprocal};
            const auto proof=prove_physical(joined,p);
            if(proof.kind==2)return true;
            // Overlapping machine-scale inclusions can lose uniqueness in
            // binary64 interval arithmetic. Promote this local union before
            // publishing a second copy of the same physical stationary root.
            if(proof.kind==0 && joined.rhi-joined.rlo<1e-8*(1+joined.rhi) &&
                    joined.thi-joined.tlo<1e-8) {
                ++out.quad_boxes;
                if(prove_quad(joined,p).kind==2)return true;
            }
        }
        return false;
    };
    // A projected event can coincide with an incumbent component event.
    // Use the same physical uniqueness proof as later discoveries; a close
    // radius alone is never sufficient to merge distinct angular events.
    std::vector<RadialEvent> initial_events;
    for(auto e:out.events) {
        const Box enclosure{e.radius_lo,e.radius_hi,e.coordinate_lo,e.coordinate_hi,e.reciprocal};
        if(!recorded_root(enclosure)){initial_events.push_back(e);record_bounds(enclosure);}
    }
    out.events=std::move(initial_events);
    auto publish=[&](const RadialEvent& event,const Box& region) {
        const Box enclosure{event.radius_lo,event.radius_hi,event.coordinate_lo,event.coordinate_hi,event.reciprocal};
        if(!recorded_root(enclosure)) {out.events.push_back(event);record_bounds(enclosure);}
        cover(region);
    };
    if(!narrowed || !radial_cover->unresolved.empty()) {
        const auto radial_base=surface_detail::radial_base(out.rmin,out.rmax,p);
        static thread_local NearbySourceCover nearby_cover;
        const bool eligible=max_boxes>=8192 && narrowed;
        if(eligible && !nearby_cover.nearby(p)) {
            nearby_cover.anchor=p;
            nearby_cover.boxes.clear();
        } else if(eligible && nearby_cover.boxes.empty()) {
            nearby_cover.prepare(p);
            out.boxes+=nearby_cover.visits;
        }
        const bool use_cover=eligible && !nearby_cover.boxes.empty() &&
            nearby_cover.boxes.size()<=96 && out.rmin>=nearby_cover.rlo && out.rmax<=nearby_cover.rhi;
        auto append=[&](double lo,double hi) {
            if(!(hi>lo))return;
            std::optional<surface_detail::Tensor> base;
            for(bool reciprocal:{false,true}) {
                // A nearby-source cover may already rule out this entire
                // radial band in one chart. Only form tensors for survivors.
                if(use_cover && std::none_of(nearby_cover.boxes.begin(),nearby_cover.boxes.end(),
                    [&](const auto& b){return b.reciprocal==reciprocal && b.rlo<hi && b.rhi>lo;}))continue;
                if(!base)base=surface_detail::restrict(radial_base,out.rmin,out.rmax,-1,1,lo,hi,-1,1);
                const auto chart=surface_detail::angular_chart(*base,reciprocal);
                if(!use_cover)pending.push_back({{lo,hi,-1,1,reciprocal},chart});
                else for(auto box:nearby_cover.boxes) {
                    if(box.reciprocal!=reciprocal)continue;
                    box.rlo=std::max(lo,box.rlo);box.rhi=std::min(hi,box.rhi);
                    if(box.rlo<box.rhi)pending.push_back({box,surface_detail::restrict(chart,lo,hi,-1,1,
                        box.rlo,box.rhi,box.tlo,box.thi)});
                }
            }
        };
        if(narrowed)for(auto interval:radial_cover->unresolved)
            append(std::max(out.rmin,interval.first),std::min(out.rmax,interval.second));
        else append(out.rmin,out.rmax);
    }
    const PhysicalExclusionData physical_data(p);
    CoefficientCache cache{p,{},{},{}};
    BasicCoefficientCache<__float128> quad_cache{p,{},{},{}};
    while(!pending.empty()) {
        auto node=std::move(pending.back());pending.pop_back();const auto box=node.box;
        out.last_box={box.rlo,box.rhi,box.tlo,box.thi};
        if(++out.boxes>max_boxes){out.reason="event_box_budget";return out;}
        if(std::any_of(covered.begin(),covered.end(),[&](const auto& known){return contains(known,box);}))continue;
        // Exclude an empty node before subtracting covered rectangles: its
        // retained tensor already certifies every point, so restrictions
        // and child tensors cannot add useful information.
        const bool local_box=box.rhi-box.rlo<=.02*(1+box.rhi) && box.thi-box.tlo<=.05;
        Box affine=box;
        if(local_box && physical_exclusion(box,p,&physical_data,&affine))continue;
        auto did_contract=[&] {return affine.rlo!=box.rlo || affine.rhi!=box.rhi ||
            affine.tlo!=box.tlo || affine.thi!=box.thi;};
        auto proof=did_contract()?prove_physical(affine,p):prove_surface(box,node.polynomial,p);
        if(proof.kind==1)continue;
        if(partial && inside_component(box,components->images))continue;
        // Remove certified regions exactly instead of bisecting down to
        // their scale again. Restrict the retained tensor to each uncovered
        // rectangle; no previously certified space is revisited.
        const Box* removal=nullptr;double removed_area=0;
        for(const auto& known:covered)if(known.reciprocal==box.reciprocal) {
            const double rw=std::min(known.rhi,box.rhi)-std::max(known.rlo,box.rlo);
            const double tw=std::min(known.thi,box.thi)-std::max(known.tlo,box.tlo);
            if(rw>0 && tw>0 && rw*tw>removed_area){removal=&known;removed_area=rw*tw;}
        }
        if(removal) {
            const double rl=std::max(box.rlo,removal->rlo),rh=std::min(box.rhi,removal->rhi);
            const double tl=std::max(box.tlo,removal->tlo),th=std::min(box.thi,removal->thi);
            for(auto child:std::array<Box,4>{{{box.rlo,rl,box.tlo,box.thi,box.reciprocal},
                 {rh,box.rhi,box.tlo,box.thi,box.reciprocal},
                 {rl,rh,box.tlo,tl,box.reciprocal},{rl,rh,th,box.thi,box.reciprocal}}}) {
                if(!(child.rhi>child.rlo && child.thi>child.tlo))continue;
                auto restricted=surface_detail::restrict(node.polynomial,box.rlo,box.rhi,box.tlo,box.thi,
                    child.rlo,child.rhi,child.tlo,child.thi);
                pending.push_back({child,std::move(restricted)});
            }
            continue;
        }
        Box proof_region=box;
        if(!local_box && physical_exclusion(box,p,&physical_data,&affine))continue;
        if(proof.kind==0 && did_contract()) {
            auto tensor=surface_detail::restrict(node.polynomial,box.rlo,box.rhi,box.tlo,box.thi,
                affine.rlo,affine.rhi,affine.tlo,affine.thi);
            pending.push_back({affine,tensor});
            ++out.contractions;continue;
        }
        if(proof.kind==0 && (box.rhi-box.rlo)<.01*(1+box.rhi) && (box.thi-box.tlo)<.01) {
            auto local=prove_physical(box,p);
            if(local.kind!=0)proof=local;
        }
        bool local_quad=proof.kind==0 && (box.rhi-box.rlo)<1e-8*(1+box.rhi) && (box.thi-box.tlo)<1e-8;
        if(local_quad){
            // Reconstruct at this local R interval in binary64 before
            // promoting. A restricted global tensor may be wider purely
            // from cancellation in its original coefficient conversion.
            proof=prove(box,p,&cache);
            if(proof.kind==0){proof=prove_quad(box,p,&quad_cache);++out.quad_boxes;}
        }
        bool radial_edge=false;
        if(narrowed && proof.kind==0)for(auto interval:radial_cover->unresolved)
            if(box.rlo==interval.first || box.rhi==interval.second){radial_edge=true;break;}
        if(proof.kind==0 && (box.rhi-box.rlo)<.01*(1+box.rhi) &&
           (box.thi-box.tlo)<.01 && (box.tlo==-1 || box.thi==1 || radial_edge)) {
            // A chart-edge or projected-band-edge root may not satisfy strict inclusion in
            // either closed half chart. Prove uniqueness in an overlapping
            // neighbourhood containing this entire box. An extra event outside
            // the original half chart is physical and harmless; it is not a
            // reason to drop any unexamined portion of the original box.
            const double dr=.25*(box.rhi-box.rlo)+32*std::numeric_limits<double>::epsilon()*(1+box.rhi);
            const double dt=.25*(box.thi-box.tlo)+32*std::numeric_limits<double>::epsilon();
            Box enlarged{std::max(out.rmin,box.rlo-dr),std::min(out.rmax,box.rhi+dr),
                box.tlo-dt,box.thi+dt,box.reciprocal};
            auto q=prove_physical(enlarged,p);
            if(q.kind==0 && (box.rhi-box.rlo)<1e-8*(1+box.rhi) && (box.thi-box.tlo)<1e-8) {
                q=prove_quad(enlarged,p,&quad_cache);++out.quad_boxes;
            }
            if(q.kind!=0){proof=q;local_quad=true;proof_region=enlarged;}
        }
        if(proof.kind==1)continue;
        if(proof.kind==2) {
            const Box enclosure{proof.r.lo,proof.r.hi,proof.t.lo,proof.t.hi,box.reciprocal};
            const bool recorded=recorded_root(enclosure) || (partial && inside_component(enclosure,components->images));
            // Keep the already proved node. Speculative physical-jet
            // expansion here repeated a costly uniqueness proof, usually
            // failed immediately, and forced arbitrary tensor restrictions.
            if(recorded){cover(proof_region);continue;}
            // The Bernstein proof has isolated one event. Refine only this
            // event centre/enclosure; retain all excluded and sibling boxes.
            const Box initial{proof.r.lo,proof.r.hi,proof.t.lo,proof.t.hi,box.reciprocal};
            double rd=initial.rlo+(initial.rhi-initial.rlo)*.5;
            double td=initial.tlo+(initial.thi-initial.tlo)*.5;
            // Value integration only needs a flux-small event enclosure, not
            // a quad-precision centre. Its full annular uncertainty is still
            // included in the controller's event_error. The default (and all
            // derivative plans) retain the incumbent precision target.
            const double radius_goal=flux_tolerance>0?
                flux_tolerance*p.rho*p.rho/(1024*(1+initial.rhi)):0;
            if(radius_goal>0 && initial.rhi-initial.rlo<=radius_goal) {
                publish({rd,td,initial.rlo,initial.rhi,box.reciprocal,initial.tlo,initial.thi},proof_region);
                continue;
            }
            bool double_solved=false;
            for(int step=0;step<16;++step) {
                const auto f=physical_or_polynomial(rd,td,p,box.reciprocal);
                const double det=f.r*f.tt-f.t*f.rt;
                if(det==0 || !std::isfinite(det))break;
                const double dr=(-f.v*f.tt+f.t*f.t)/det,dt=(-f.r*f.t+f.rt*f.v)/det;
                if(rd+dr<initial.rlo || rd+dr>initial.rhi || td+dt<initial.tlo || td+dt>initial.thi)break;
                rd+=dr;td+=dt;
                if(std::abs(dr)+std::abs(dt)<8*std::numeric_limits<double>::epsilon()*(1+std::abs(rd)+std::abs(td))) {
                    double_solved=true;break;
                }
            }
            if(double_solved) {
                const double dr=std::max(radius_goal,32*std::numeric_limits<double>::epsilon()*(1+std::abs(rd)));
                const double dt=std::max(radius_goal,32*std::numeric_limits<double>::epsilon()*(1+std::abs(td)));
                Box tiny{std::max(initial.rlo,rd-dr),std::min(initial.rhi,rd+dr),
                    std::max(initial.tlo,td-dt),std::min(initial.thi,td+dt),box.reciprocal};
                auto fine=prove_physical(tiny,p);
                if(fine.kind==0)fine=prove(tiny,p,&cache);
                if(fine.kind==2) {
                    publish({rd,td,fine.r.lo,fine.r.hi,box.reciprocal,fine.t.lo,fine.t.hi},proof_region);
                    ++out.double_refinements;continue;
                }
            }
            // Escalate the isolated event only when its binary64 centre or
            // enclosure was insufficient; this never restarts global discovery.
            __float128 r=(__float128(initial.rlo)+__float128(initial.rhi))/2;
            __float128 t=(__float128(initial.tlo)+__float128(initial.thi))/2;
            bool solved=false;
            for(int step=0;step<32;++step) {
                const auto f=physical_or_polynomial(r,t,p,box.reciprocal);
                const auto det=f.r*f.tt-f.t*f.rt;
                if(det==0 || !finiteq(det))break;
                auto dr=(-f.v*f.tt+f.t*f.t)/det;
                auto dt=(-f.r*f.t+f.rt*f.v)/det;
                bool inside=false;
                for(int backtrack=0;backtrack<32;++backtrack) {
                    if(r+dr>=(__float128)initial.rlo && r+dr<=(__float128)initial.rhi &&
                       t+dt>=(__float128)initial.tlo && t+dt<=(__float128)initial.thi){inside=true;break;}
                    dr*=.5;dt*=.5;
                }
                if(!inside)break;
                r+=dr;t+=dt;
                if(fabsq(dr)+fabsq(dt)<std::max((__float128)1e-29,(__float128)radius_goal/16)){solved=true;break;}
            }
            if(!solved){out.reason="event_refinement_limited";return out;}
            const double radius=double(r),coordinate=double(t);
            const double dr=std::max(radius_goal,32*std::numeric_limits<double>::epsilon()*(1+std::abs(radius)));
            const double dt=std::max(radius_goal,32*std::numeric_limits<double>::epsilon()*(1+std::abs(coordinate)));
            Box tiny{std::max(initial.rlo,radius-dr),std::min(initial.rhi,radius+dr),
                std::max(initial.tlo,coordinate-dt),std::min(initial.thi,coordinate+dt),box.reciprocal};
            auto final=prove_quad(tiny,p,&quad_cache);++out.quad_refinements;
            if(final.kind!=2){out.reason="event_enclosure_limited";return out;}
            publish({radius,coordinate,final.r.lo,final.r.hi,box.reciprocal,final.t.lo,final.t.hi},proof_region);
            continue;
        }
        // Contract a box even when strict inclusion has not yet proved existence.
        if(proof.r.valid() && proof.t.valid() && proof.r.hi>proof.r.lo && proof.t.hi>proof.t.lo &&
           (proof.r.hi-proof.r.lo)*(proof.t.hi-proof.t.lo)<
               .5*(box.rhi-box.rlo)*(box.thi-box.tlo)) {
            auto restricted=surface_detail::restrict(node.polynomial,box.rlo,box.rhi,box.tlo,box.thi,
                proof.r.lo,proof.r.hi,proof.t.lo,proof.t.hi);
            pending.push_back({{proof.r.lo,proof.r.hi,proof.t.lo,proof.t.hi,box.reciprocal},restricted});
            ++out.contractions;continue;
        }
        const double rw=(box.rhi-box.rlo)/(1+box.rhi),tw=box.thi-box.tlo;
        // Use retained coefficient variation to balance the two equations,
        // instead of imposing one aspect ratio on every image branch.
        if(surface_detail::prefer_radial_split(node.polynomial,rw,tw)) {
            const double mid=box.rlo+(box.rhi-box.rlo)*.5;
            if(!(mid>box.rlo && mid<box.rhi)){out.reason="degenerate_radial_event";return out;}
            auto halves=surface_detail::split(node.polynomial,0,
                surface_detail::fraction(mid,box.rlo,box.rhi));
            pending.push_back({{mid,box.rhi,box.tlo,box.thi,box.reciprocal},halves.second});
            pending.push_back({{box.rlo,mid,box.tlo,box.thi,box.reciprocal},halves.first});
        } else {
            // Midpoint subdivision has a cheaper exact-dyadic enclosure.
            // Offset only a split at the symmetry axis, keeping t=0 interior
            // to a child. Every split still covers the full parent box.
            double mid=box.tlo+(box.thi-box.tlo)*.5;
            if(mid==0)mid=box.tlo+(box.thi-box.tlo)*.4375;
            if(!(mid>box.tlo && mid<box.thi)){out.reason="degenerate_angular_event";return out;}
            auto halves=surface_detail::split(node.polynomial,1,
                surface_detail::fraction(mid,box.tlo,box.thi));
            pending.push_back({{box.rlo,box.rhi,mid,box.thi,box.reciprocal},halves.second});
            pending.push_back({{box.rlo,box.rhi,box.tlo,mid,box.reciprocal},halves.first});
        }
    }
    std::sort(out.events.begin(),out.events.end(),[](auto a,auto b){return a.radius<b.radius;});
    out.completed=true;out.reason=narrowed?(out.boxes?"radial_resultant_complement":"radial_resultant"):
        partial?"partial_component_complement":"completed";return out;
}
} // namespace lcbinint::algebraic_boundary::triple
