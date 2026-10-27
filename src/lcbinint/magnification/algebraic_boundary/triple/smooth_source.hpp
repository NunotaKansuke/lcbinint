#pragma once
#include "regular_support.hpp"
#include "coefficient_ball.hpp"
#include "../nested_fejer2.hpp"
#include "../quadrature_error_policy.hpp"
#include <map>
namespace lcbinint::algebraic_boundary::triple::smooth_source {
using B=coefficient_ball_detail::Ball<double>;
using C=coefficient_detail::C<B>;
inline B inverse(B a){const double lower=std::abs(a.value)-a.error;if(!(lower>0)||!std::isfinite(a.error))return {0,INFINITY};double v=1/a.value;return {v,(a.error/std::abs(a.value)/lower+8*std::numeric_limits<double>::epsilon()*std::abs(v)+std::numeric_limits<double>::min())*(1+64*std::numeric_limits<double>::epsilon())};}
inline C scale(C a,B b){return {a.re*b,a.im*b};}
inline bool finite(B a){return std::isfinite(a.value)&&std::isfinite(a.error)&&a.error>=0;}
// Complex disks use one 1-norm radius, so the seven RHS reuse the same
// inverse/shear errors instead of replaying scalar forward-mode arithmetic.
inline double norm1(Complex z){return std::abs(z.real())+std::abs(z.imag());}
inline double padded(double e){return (e+std::numeric_limits<double>::min())*(1+128*std::numeric_limits<double>::epsilon());}
struct DiskValue {Complex v{};double e=0;};
inline DiskValue operator+(DiskValue a,DiskValue b){return {a.v+b.v,padded(a.e+b.e+8*std::numeric_limits<double>::epsilon()*(norm1(a.v)+norm1(b.v)))};}
inline DiskValue operator-(DiskValue a){return {-a.v,a.e};}
inline DiskValue operator-(DiskValue a,DiskValue b){return a+(-b);}
inline DiskValue operator*(DiskValue a,DiskValue b){return {a.v*b.v,padded(norm1(a.v)*b.e+norm1(b.v)*a.e+a.e*b.e+8*std::numeric_limits<double>::epsilon()*norm1(a.v)*norm1(b.v))};}
inline DiskValue conjugate(DiskValue a){return {std::conj(a.v),a.e};}
inline DiskValue times(DiskValue a,B b){return {a.v*b.value,padded(norm1(a.v)*b.error+std::abs(b.value)*a.e+a.e*b.error+8*std::numeric_limits<double>::epsilon()*norm1(a.v)*std::abs(b.value))};}
inline B norm_squared(DiskValue a){const double v=std::norm(a.v);return {v,padded(2*norm1(a.v)*a.e+a.e*a.e+8*std::numeric_limits<double>::epsilon()*v)};}
inline DiskValue reciprocal(DiskValue a){return times(conjugate(a),inverse(norm_squared(a)));}
inline B dot(DiskValue a,DiskValue b){return {std::real(std::conj(a.v)*b.v),padded(norm1(a.v)*b.e+norm1(b.v)*a.e+a.e*b.e+8*std::numeric_limits<double>::epsilon()*norm1(a.v)*norm1(b.v))};}
struct Point {B mu;std::array<B,7> g{};bool ok=false;}; // X,Y,m1,m2,a,bx,by
inline Point point(Complex source,double source_roundoff,const Geometry&p,const std::vector<ImageExclusion>&images,bool value_mesh=false){
    Point out;const auto lenses=p.lenses();
    const std::array<B,3> masses{B(1)-B(p.m1)-B(p.m2),B(p.m1),B(p.m2)};
    auto local=p;local.X=source.real();local.Y=source.imag();local.rho=source_roundoff;
    for(const auto&image:images) {
        const Complex z=regular_support_detail::solve_image(image.center,source,p);
        const auto enclosure=regular_support_detail::image_ball(z,local);
        if(!enclosure || std::abs(enclosure->z-image.center)+enclosure->r>=image.radius)return out;
        const DiskValue zb{std::conj(z),2*enclosure->r};
        std::array<DiskValue,3> inv{},square{},cube{};DiskValue K{},H{};
        for(int j=0;j<3;++j) {
            const auto d=zb-DiskValue{std::conj(lenses[j]),0};
            inv[j]=reciprocal(d);square[j]=inv[j]*inv[j];
            K=K+times(square[j],masses[j]);
            if(value_mesh){cube[j].v=square[j].v*inv[j].v;H.v+=(-2*masses[j].value)*cube[j].v;}
            else {cube[j]=square[j]*inv[j];H=H+times(cube[j],B(-2)*masses[j]);}
        }
        const B D=B(1)-norm_squared(K);
        if(!finite(D)||std::abs(D.value)<=D.error)return out;
        const B invD=inverse(D),sign(D.value>0?1.:-1.);
        out.mu=out.mu+sign*invD;
        if(value_mesh) {
            // The primal keeps its point-image and roundoff bounds. The
            // unchecked tangents share that geometry, without constructing
            // eight unused derivative error ledgers.
            const Complex shear=K.v,hessian=H.v;const double determinant=D.value;
            const std::array<Complex,7> rhs{Complex(1,0),Complex(0,1),inv[1].v-inv[0].v,inv[2].v-inv[0].v,
                masses[1].value*square[1].v,masses[2].value*square[2].v,Complex(0,-1)*masses[2].value*square[2].v};
            const std::array<Complex,7> explicit_K{Complex{},Complex{},square[1].v-square[0].v,square[2].v-square[0].v,
                2*masses[1].value*cube[1].v,2*masses[2].value*cube[2].v,Complex(0,-2)*masses[2].value*cube[2].v};
            for(int j=0;j<7;++j){const Complex motion=(rhs[j]-shear*std::conj(rhs[j]))/determinant;
                const Complex dK=hessian*std::conj(motion)+explicit_K[j];
                out.g[j].value+=sign.value*2*std::real(std::conj(shear)*dK)/(determinant*determinant);}
            continue;
        }
        const std::array<DiskValue,7> rhs{DiskValue{{1,0},0},DiskValue{{0,1},0},inv[1]-inv[0],inv[2]-inv[0],
            times(square[1],masses[1]),times(square[2],masses[2]),
            times(square[2]*DiskValue{{0,-1},0},masses[2])};
        const std::array<DiskValue,7> explicit_K{DiskValue{},DiskValue{},square[1]-square[0],square[2]-square[0],
            times(cube[1],B(2)*masses[1]),times(cube[2],B(2)*masses[2]),
            times(cube[2]*DiskValue{{0,-1},0},B(2)*masses[2])};
        for(int j=0;j<7;++j) {
            const auto motion=times(rhs[j]-K*conjugate(rhs[j]),invD);
            const auto dK=H*conjugate(motion)+explicit_K[j];
            out.g[j]=out.g[j]+sign*B(2)*dot(K,dK)*invD*invD;
        }
    }
    out.ok=finite(out.mu);for(auto v:out.g)out.ok=out.ok&&finite(v);return out;
}
using Vector=std::array<B,9>;
struct Result {double mu=0,error=INFINITY;std::array<double,8> gradient{},gradient_error{};bool attempted=false,converged=false;int refinements=0;std::size_t samples=0,reused_samples=0;};
// Interpolatory product weights for the normalized source measure. The
// same nested nodes now integrate low-degree angular means exactly even
// with LD; multiplying an ordinary Fejer rule by the density wastes order.
inline std::vector<double> weights(int level,double limb) {
    const int m=1<<level;std::vector<double>w(m-1);
    const long double u=limb,den=1-u/3,c0=(.5L-.25L*u)/den,c1=.5L/den,c2=.25L*u/den;
    const auto moment=[](int k)->long double {if(k==-2)return -2;if(k<0||k%2)return 0;return 2.L/(k+1);};
    for(int k=1;k<m;++k) {
        const long double theta=acosl(-1.L)*k/m;long double sum=0;
        for(int j=1;j<m;++j) {
            const long double M=c0*moment(j-1)+c1*(moment(j)+moment(j-2))/2+
                c2*(moment(j+1)+2*moment(j-1)+moment(j-3))/4;
            sum+=M*sinl(j*theta);
        }
        w[k-1]=double(2*sinl(theta)*sum/m);
    }
    return w;
}
inline Result integrate(const Geometry&p,double limb,double atol,double rtol,int maximum_refinements,const RadialEventPlan&plan,bool value_mesh=false){
    Result out;
    if(plan.identity!=std::array<double,8>{p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by} || plan.expected_images<4 || plan.component_images.size()!=std::size_t(plan.expected_images) ||
        !plan.excluded_images.empty() || !plan.completed)return out;
    const auto&images=plan.component_images;
    for(std::size_t i=0;i<images.size();++i)for(std::size_t j=0;j<i;++j)
        if(std::abs(images[i].center-images[j].center)<=images[i].radius+images[j].radius+256*std::numeric_limits<double>::epsilon()*(1+std::abs(images[i].center)+std::abs(images[j].center)))return out;
    out.attempted=true;
    // Canonical nested radial and periodic-angular keys preserve every
    // computed point when either dimension is refined.
    std::map<std::pair<int,int>,Point> cache;
    Point unresolved;
    auto sample=[&](int radius_slot,int angle_slot,double z)->const Point& {
        auto key=std::make_pair(radius_slot,angle_slot);auto it=cache.find(key);if(it!=cache.end()){++out.reused_samples;return it->second;}
        if(cache.size()>=4096)return unresolved;
        const int half=128,index=angle_slot<half?angle_slot:angle_slot-half;
        const double sign=angle_slot<half?1.:-1.,angle=2*std::acos(-1.)*index/256;
        const double rr=std::sqrt(std::max(0.,(1-z)*(1+z)));
        const double dx=sign*rr*std::cos(angle),dy=sign*rr*std::sin(angle);
        const Complex source(p.X+p.rho*dx,p.Y+p.rho*dy);
        const double rounding=16*std::numeric_limits<double>::epsilon()*(std::abs(p.X)+std::abs(p.Y)+p.rho*(std::abs(dx)+std::abs(dy)))+std::numeric_limits<double>::min();
        ++out.samples;return cache.emplace(key,point(source,rounding,p,images,value_mesh)).first->second;
    };
    int radial_level=2,angular_level=3;
    for(int iteration=0;iteration<=maximum_refinements;++iteration) {
        const int m=1<<radial_level,N=1<<angular_level;
        const auto&rule=fejer_rule(radial_level);
        const auto fine_weights=weights(radial_level,limb),coarse_weights=weights(radial_level-1,limb);
        std::vector<Vector> values(m-1),angular_errors(m-1);
        double normalizer=0,coarse_normalizer=0;
        for(int k=1;k<m;++k) {
            const double z=(1+rule.x[k-1])*.5,rr=std::sqrt(std::max(0.,(1-z)*(1+z)));
            normalizer+=fine_weights[k-1];
            if(!(k&1))coarse_normalizer+=coarse_weights[k/2-1];
            Vector fine{},low{};
            for(int j=0;j<N/2;++j) {
                const int slot=j*(256/N);const auto&a=sample(k<<(8-radial_level),slot,z);
                const auto&b=sample(k<<(8-radial_level),slot+128,z);
                if(!a.ok||!b.ok)return out;
                const double angle=2*std::acos(-1.)*slot/256;
                Vector pair{};pair[0]=a.mu+b.mu;
                if(value_mesh){pair[1].value=a.g[0].value+b.g[0].value;pair[2].value=a.g[1].value+b.g[1].value;
                    pair[3].value=rr*(std::cos(angle)*(a.g[0].value-b.g[0].value)+std::sin(angle)*(a.g[1].value-b.g[1].value));
                    for(int n=2;n<7;++n)pair[n+2].value=a.g[n].value+b.g[n].value;
                }else{pair[1]=a.g[0]+b.g[0];pair[2]=a.g[1]+b.g[1];
                    pair[3]=B(rr)*(B(std::cos(angle))*(a.g[0]-b.g[0])+B(std::sin(angle))*(a.g[1]-b.g[1]));
                    for(int n=2;n<7;++n)pair[n+2]=a.g[n]+b.g[n];}
                for(int n=0;n<9;++n){
                    if(value_mesh && n>0)fine[n].value+=pair[n].value/N;
                    else {fine[n]=fine[n]+B(1./N)*pair[n];if(!(j&1))low[n]=low[n]+B(2./N)*pair[n];}
                }
            }
            for(int n=0;n<9;++n){values[k-1][n]=fine[n];if(!value_mesh || n==0)angular_errors[k-1][n]=B(std::abs(fine[n].value-low[n].value));}
        }
        Vector fine{},low{};std::array<double,9> inner{},detail{},radial{},errors{};
        for(int k=1;k<m;++k) {
            const double w=fine_weights[k-1]/normalizer;
            const B weight(w,128*std::numeric_limits<double>::epsilon()*std::abs(w));
            for(int n=0;n<9;++n){if(value_mesh && n>0)fine[n].value+=w*values[k-1][n].value;
                else {fine[n]=fine[n]+weight*values[k-1][n];inner[n]+=std::abs(w)*angular_errors[k-1][n].value;}}
            if(!(k&1)) {
                const double wc=coarse_weights[k/2-1]/coarse_normalizer;
                for(int n=0;n<(value_mesh?1:9);++n)low[n]=low[n]+B(wc)*values[k-1][n];
            }
        }
        // Independent interpolation detail prevents a cancelling embedded
        // radial difference from hiding structure. The profile is assembled
        // before the error estimate, just as in the binary channel policy.
        const int nc=m/2-1;
        for(int k=1;k<m;k+=2)for(int n=0;n<(value_mesh?1:9);++n) {
            double predicted=0;
            for(int j=1;j<=nc;++j)predicted+=rule.interp[(k/2)*nc+j-1]*values[2*j-1][n].value;
            detail[n]+=std::abs(fine_weights[k-1]/normalizer)*std::abs(values[k-1][n].value-predicted);
        }
        if(p.Y==0 && p.by==0)for(int n:{2,8}){fine[n]=B(0);low[n]=B(0);inner[n]=detail[n]=0;}
        bool pass=angular_level>=(value_mesh?3:4),roundoff_blocked=false;double radial_score=0,angular_score=0;
        for(int n=0;n<9;++n) {
            if(value_mesh && n>0){errors[n]=INFINITY;pass=pass&&finite(fine[n]);continue;}
            radial[n]=std::max(QuadratureErrorPolicy::difference_safety*std::abs(fine[n].value-low[n].value),QuadratureErrorPolicy::weighted_detail_floor*detail[n]);
            errors[n]=radial[n]+QuadratureErrorPolicy::difference_safety*inner[n]+fine[n].error;
            const double budget=atol+rtol*std::abs(fine[n].value);
            pass=pass&&finite(fine[n]);
            if(!value_mesh || n==0)pass=pass&&errors[n]<=budget;
            if(fine[n].error>budget && radial[n]<=budget && inner[n]<=budget)roundoff_blocked=true;
            const double scale=std::max(budget,std::numeric_limits<double>::min());if(!value_mesh || n==0){radial_score=std::max(radial_score,radial[n]/scale);angular_score=std::max(angular_score,QuadratureErrorPolicy::difference_safety*inner[n]/scale);}
        }
        // Exact reflection identities avoid treating transverse zeros as
        // an impossible relative-only accuracy request.
        if(p.Y==0 && p.by==0){fine[2]=B(0);fine[8]=B(0);errors[2]=errors[8]=0;}
        out.mu=fine[0].value;out.error=errors[0];for(int j=0;j<8;++j){out.gradient[j]=fine[j+1].value;out.gradient_error[j]=errors[j+1];}
        out.refinements=iteration;
        if(pass){out.converged=true;return out;}
        if(roundoff_blocked && angular_level>=4)return out;
        if(angular_level<(value_mesh?3:4) || (angular_score>radial_score && angular_level<8))++angular_level;
        else if(radial_level<6)++radial_level;
        else if(angular_level<8)++angular_level;
        else return out;
    }
    return out;
}
} // namespace
