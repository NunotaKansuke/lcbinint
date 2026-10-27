#pragma once
#include "geometry.hpp"
#include "gradient_quadrature.hpp"
#include "../nested_fejer2.hpp"
#include <vector>
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// Confluent quadrature: values and first derivatives at the primal cosine
// nodes integrate polynomials through degree 2*n-1. Orthogonal moments keep
// construction separate from geometry and from any gradient-error controller.
struct HermiteRule {
    std::vector<double> x,value,derivative;
    struct Witness {double x;std::vector<double> value,derivative;};
    std::vector<Witness> witnesses;
    HermiteRule(int level,bool weighted) {
        x=fejer_rule(level).x;const int n=int(x.size()),m=2*n;
        std::vector<std::vector<long double>> a(m,std::vector<long double>(m+1));
        a[0][m]=weighted?acosl(-1.L)/2:2;
        for(int j=0;j<n;++j) {
            long double previous=1,current=weighted?2*x[j]:x[j],dp=0,dc=weighted?2:1;
            a[0][j]=1;a[1][j]=current;a[1][n+j]=dc;
            for(int k=2;k<m;++k) {
                const long double next=weighted?2*x[j]*current-previous:((2*k-1)*x[j]*current-(k-1)*previous)/k;
                const long double dn=weighted?2*current+2*x[j]*dc-dp:((2*k-1)*(current+x[j]*dc)-(k-1)*dp)/k;
                a[k][j]=next;a[k][n+j]=dn;previous=current;current=next;dp=dc;dc=dn;
            }
        }
        for(int k=0;k<m;++k) {
            int pivot=k;for(int i=k+1;i<m;++i)if(fabsl(a[i][k])>fabsl(a[pivot][k]))pivot=i;
            std::swap(a[k],a[pivot]);if(!(fabsl(a[k][k])>1e-24L))throw std::logic_error("Hermite moment rank");
            const auto d=a[k][k];for(int j=k;j<=m;++j)a[k][j]/=d;
            for(int i=0;i<m;++i)if(i!=k) {const auto q=a[i][k];for(int j=k;j<=m;++j)a[i][j]-=q*a[k][j];}
        }
        value.resize(n);derivative.resize(n);
        for(int j=0;j<n;++j){value[j]=double(a[j][m]);derivative[j]=double(a[n+j][m]);}
        for(double q:gauss_rule(8).x) {
            Witness w;w.x=q;w.value.resize(n);w.derivative.resize(n);
            for(int j=0;j<n;++j) {
                long double l=1,d=0;
                for(int k=0;k<n;++k)if(k!=j){l*=((long double)q-x[k])/(x[j]-x[k]);d+=1.L/(x[j]-x[k]);}
                w.value[j]=double((1-2*(q-x[j])*d)*l*l);w.derivative[j]=double((q-x[j])*l*l);
            }
            witnesses.push_back(std::move(w));
        }
    }
};
inline const HermiteRule& hermite_rule(int level,bool weighted) {
    // Avoid order-dependent anonymous static symbols in diagnostic DSOs.
    if(level==3){static const HermiteRule plain3(3,false),fold3(3,true);return weighted?fold3:plain3;}
    if(level==4){static const HermiteRule plain4(4,false),fold4(4,true);return weighted?fold4:plain4;}
    if(level==5){static const HermiteRule plain5(5,false),fold5(5,true);return weighted?fold5:plain5;}
    throw std::invalid_argument("Hermite replay level");
}
struct PolarMixedJet {
    double phi=0,r=0,t=0,rr=0,rt=0,tt=0,roundoff=0;
    std::array<double,8> p{},rp{},tp{};
};
// Immutable radius data are shared by every angular node. No derivative
// controller or parameter work is added to the value-only workspace.
struct MixedRay {
    const Geometry& p;
    double R,m0,rho2,source_norm,inverse_rho2;
    MixedRay(double radius,const Geometry& geometry):p(geometry),R(radius),
        m0(1-p.m1-p.m2),rho2(p.rho*p.rho),
        source_norm(std::hypot(p.X,p.Y)),inverse_rho2(1/rho2) {}
    template<bool Curvature=true> PolarMixedJet sample(double theta) const {
        const Complex w(std::cos(theta),std::sin(theta)),z=R*w,I(0,1);
        const Complex i0=w/R,i1=inverse_conjugate(z-Complex(p.a)),i2=inverse_conjugate(z-Complex(p.bx,p.by));
        const Complex q0=i0*i0,q1=i1*i1,q2=i2*i2,c0=q0*i0,c1=q1*i1,c2=q2*i2;
        const Complex g=(R-m0/R)*w-Complex(p.X,p.Y)-p.m1*i1-p.m2*i2;
        const Complex shear=m0*q0+p.m1*q1+p.m2*q2,cubic=m0*c0+p.m1*c1+p.m2*c2;
        const Complex gr=w+std::conj(w)*shear,gt=I*(z-std::conj(z)*shear);
        const Complex grr=-2.*std::conj(w)*std::conj(w)*cubic;
        const Complex grt=I*(w-std::conj(w)*shear+2.*std::conj(z)*std::conj(w)*cubic);
        const Complex gtt=-z-std::conj(z)*shear+2.*std::conj(z)*std::conj(z)*cubic;
        const auto dot=[](Complex a,Complex b){return a.real()*b.real()+a.imag()*b.imag();};
        PolarMixedJet out;
        const double eps=std::numeric_limits<double>::epsilon(),norm=std::norm(g);
        const double deflection=m0/R+p.m1*std::abs(i1)+p.m2*std::abs(i2);
        const double bound=m0/(R*R)+p.m1*std::norm(i1)+p.m2*std::norm(i2);
        const double delta=32*eps*(R+source_norm+deflection+R*bound);
        out.roundoff=((2*std::abs(g)*delta+delta*delta)*inverse_rho2+16*eps*(1+norm*inverse_rho2))*(1+1e-12);
        out.phi=1-norm*inverse_rho2;out.r=-2*dot(g,gr)*inverse_rho2;out.t=-2*dot(g,gt)*inverse_rho2;
        out.rr=-2*(std::norm(gr)+dot(g,grr))*inverse_rho2;out.rt=-2*(dot(gr,gt)+dot(g,grt))*inverse_rho2;if constexpr(Curvature)out.tt=-2*(std::norm(gt)+dot(g,gtt))*inverse_rho2;
        const std::array<Complex,8> gp{-1.,-I,0.,i0-i1,i0-i2,-p.m1*q1,-p.m2*q2,I*p.m2*q2};
        const std::array<Complex,8> grp{0.,0.,0.,-std::conj(w)*(q0-q1),-std::conj(w)*(q0-q2),2*p.m1*std::conj(w)*c1,2*p.m2*std::conj(w)*c2,-2.*I*p.m2*std::conj(w)*c2};
        // Angular and radial parameter jets obey g_theta,p = -i R g_R,p.
        for(int j=0;j<8;++j){out.p[j]=-2*dot(g,gp[j])*inverse_rho2;out.rp[j]=-2*(dot(gr,gp[j])+dot(g,grp[j]))*inverse_rho2;out.tp[j]=-2*(dot(gt,gp[j])+R*(g.real()*grp[j].imag()-g.imag()*grp[j].real()))*inverse_rho2;}
        out.p[2]=2*norm*inverse_rho2/p.rho;out.rp[2]=-2*out.r/p.rho;out.tp[2]=-2*out.t/p.rho;
        return out;
    }
};
template<bool Curvature=true> inline PolarMixedJet polar_mixed(double R,double theta,const Geometry& p) {
    return MixedRay(R,p).sample<Curvature>(theta);
}
struct LimbMixedDensity {
    double flux=0,radial_flux=0;
    std::array<double,8> gradient{},radial_gradient{};
};
// Differentiate the fixed-angle shape density, transporting the angular
// measure with B=dtheta/dR. The angular chart itself need not be differentiated.
inline LimbMixedDensity limb_mixed_density(const PolarMixedJet& q,double R,double B,double Btheta,
        const std::array<double,8>& velocity,const std::array<double,8>& slope) {
    LimbMixedDensity out;
    const double root=std::sqrt(q.phi),inverse_root=0.5/root,qr=q.r+q.t*B,root_r=qr*inverse_root;
    const double radial_scale=q.rr+q.rt*B,root_cross=qr/(4*root*root*root);
    out.flux=R*root;out.radial_flux=root+R*root_r+out.flux*Btheta;
    for(int k=0;k<8;++k) {
        const double qp=q.p[k]+q.r*velocity[k];
        const double qrp=q.rp[k]+q.tp[k]*B+radial_scale*velocity[k]+q.r*slope[k];
        const double root_p=qp*inverse_root,root_rp=qrp*inverse_root-root_cross*qp;
        out.gradient[k]=velocity[k]*root+R*root_p;
        out.radial_gradient[k]=slope[k]*root+velocity[k]*root_r+root_p+R*root_rp+out.gradient[k]*Btheta;
    }
    return out;
}

// Transport BOTH parameter and radial angular motions before forming inverse
// square roots. On a physical endpoint q_p and q_R vanish with phi; the
// fixed-angle formula retains a spurious 1/sqrt(phi) angular boundary layer.
inline LimbMixedDensity mapped_limb_mixed_density(const PolarMixedJet& q,double R,
        double half,double half_r,const std::array<double,8>& half_p,const std::array<double,8>& half_rp,
        double B,const std::array<double,8>& A,const std::array<double,8>& C,
        const std::array<double,8>& velocity,const std::array<double,8>& slope,double shape) {
    LimbMixedDensity out;
    const double root=std::sqrt(q.phi/shape),ir=.5/(root*shape),qr=q.r+q.t*B;
    const double root_r=qr*ir,U=R*half,Ur=half+R*half_r;
    out.flux=U*root;out.radial_flux=Ur*root+U*root_r;
    for(int k=0;k<8;++k) {
        const double qp=q.p[k]+q.r*velocity[k]+q.t*A[k];
        const double qrp=q.rp[k]+q.tp[k]*B+(q.rr+q.rt*B)*velocity[k]+q.r*slope[k]
            +(q.rt+q.tt*B)*A[k]+q.t*C[k];
        const double root_p=qp*ir,root_rp=qrp*ir-qr*qp/(4*root*root*root*shape*shape);
        const double Up=velocity[k]*half+R*half_p[k];
        const double Urp=slope[k]*half+velocity[k]*half_r+half_p[k]+R*half_rp[k];
        out.gradient[k]=Up*root+U*root_p;
        out.radial_gradient[k]=Urp*root+Up*root_r+Ur*root_p+U*root_rp;
    }
    return out;
}
} // namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail
