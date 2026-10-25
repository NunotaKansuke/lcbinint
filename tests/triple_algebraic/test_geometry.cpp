#include "lcbinint/magnification/algebraic_boundary/triple/geometry.hpp"
#include "lcbinint/magnification/algebraic_boundary/boundary_polynomial.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
#include <quadmath.h>
using namespace lcbinint::algebraic_boundary::triple;
void check(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
double eval(const std::array<double,7>& p,double t) {
    double out=p[6];for(int j=5;j>=0;--j)out=out*t+p[j];return out;
}
int main() {
    // Check the fast reciprocal and the scaled fallback against promoted
    // arithmetic, including distances whose squared norm over/underflows.
    for(double scale:{1.,1e150,1e-150,1e300,1e-300}) {
        Complex d(scale,.7*scale),v=inverse_conjugate(d);
        const __float128 x=d.real(),y=d.imag(),n=x*x+y*y;
        const __float128 re=x/n,im=y/n;
        const __float128 error=fabsq((__float128)v.real()-re)+fabsq((__float128)v.imag()-im);
        check(error<32*(__float128)std::numeric_limits<double>::epsilon()*(fabsq(re)+fabsq(im))+
              (__float128)std::numeric_limits<double>::denorm_min(),"inverse normal/extreme range");
    }
    std::mt19937 rng(731);
    std::uniform_real_distribution<double> random(-1,1);
    int checks=0;
    for(int n=0;n<200;++n) {
        Geometry p{random(rng),random(rng),.03+.2*std::abs(random(rng)),
            .2,.1,.7+.2*random(rng),-.3+.2*random(rng),.5+.2*random(rng)};
        double R=.1+2*std::abs(random(rng));
        auto b=boundary(R,p);
        for(int k=0;k<13;++k) {
            double t=random(rng), theta=2*std::atan(t);
            Complex z=std::polar(R,theta),g=z-Complex(p.X,p.Y),D=1;
            auto lenses=p.lenses();auto masses=p.masses();
            for(int j=0;j<3;++j) {
                g-=masses[j]/std::conj(z-lenses[j]);
                D*=R-std::conj(lenses[j])*std::polar(1.,theta);
            }
            double expected=std::pow(1+t*t,3)*(p.rho*p.rho-std::norm(g))*std::norm(D);
            double actual=eval(b.coefficients,t);
            check(std::abs(actual-expected)<2e-12*(1+std::abs(expected)),"boundary/direct map");
            auto reversed=reciprocal(b.coefficients);
            if(std::abs(t)>.1)
                check(std::abs(eval(reversed,t)-std::pow(t,6)*eval(b.coefficients,-1/t))<2e-12*(1+std::abs(actual)),"reciprocal chart");
            ++checks;
        }
        double theta=.4;
        auto jet=evaluate<true>(R,theta,p);
        check(jet.phi==evaluate(R,theta,p).phi,"value route consistency");
        double Geometry::* members[]={&Geometry::X,&Geometry::Y,&Geometry::rho,
            &Geometry::m1,&Geometry::m2,&Geometry::a,&Geometry::bx,&Geometry::by};
        // Independently perturb every local coordinate with two step sizes.
        for(int j=0;j<8;++j) for(double step:{1e-5,3e-6}) {
            auto plus=p,minus=p;
            plus.*members[j]+=step;minus.*members[j]-=step;
            double fd=(evaluate(R,theta,plus).phi-evaluate(R,theta,minus).phi)/(2*step);
            check(std::abs(fd-jet.dp[j])<3e-6*(1+std::abs(fd)),"analytic parameter derivative");
        }
        for(double step:{1e-5,3e-6}) {
            double fd=(evaluate(R+step,theta,p).phi-evaluate(R-step,theta,p).phi)/(2*step);
            check(std::abs(fd-jet.dR)<3e-6*(1+std::abs(fd)),"radial derivative");
            fd=(evaluate(R,theta+step,p).phi-evaluate(R,theta-step,p).phi)/(2*step);
            check(std::abs(fd-jet.dtheta)<3e-6*(1+std::abs(fd)),"angular derivative");
        }
        // Binary limit: the sextic equals the incumbent quartic times
        // the tertiary pole factor after clearing denominators.
        p.m2=0;
        b=boundary(R,p);
        lcbinint::algebraic_boundary::PrimaryFrame binary{p.a,1-p.m1,p.X,p.Y,p.rho};
        auto q=lcbinint::algebraic_boundary::boundary_quartic(R,binary);
        double t=.31, theta2=2*std::atan(t);
        double quartic=0;for(int j=4;j>=0;--j)quartic=quartic*t+q.p[j];
        double factor=std::norm(R-std::conj(Complex(p.bx,p.by))*std::polar(1.,theta2))*(1+t*t);
        check(std::abs(eval(b.coefficients,t)-factor*quartic)<2e-12*(1+std::abs(factor*quartic)),"incumbent binary limit");
    }
    // Symmetry: reflection across the primary axis leaves phi unchanged.
    Geometry reflected{.13,.27,.08,.2,.07,1.1,-.3,.6};
    auto mirror=reflected; mirror.Y=-mirror.Y; mirror.by=-mirror.by;
    for(double theta:{-.8,.1,1.9,3.0})
        check(std::abs(evaluate(.9,theta,reflected).phi-
            evaluate(.9,-theta,mirror).phi)<1e-10,"reflection identity");
    // Coincident lenses reduce to a single lens; angular degree drops without
    // requiring division by a leading coefficient.
    Geometry coincident{.1,-.2,.04,.2,.1,0,0,0};
    auto single=coincident; single.m1=0; single.m2=0;
    for(double theta:{-.7,.2,2.4}) {
        double R=1.3,t=std::tan(theta/2);
        check(std::abs(evaluate(R,theta,coincident).phi-
            evaluate(R,theta,single).phi)<1e-10,"coincident lens limit");
        check(std::abs(eval(boundary(R,coincident).coefficients,t)-
            eval(boundary(R,single).coefficients,t))<1e-10,"degree reduction");
    }
    bool rejected=false;
    try { (void)evaluate(0.,0.,Geometry{0,0,.1,.2,.1,1,0,1}); }
    catch(const std::domain_error&) {rejected=true;}
    check(rejected,"pole rejection");
    std::cout << "PASS: " << checks << " direct-map comparisons; 200 geometries, derivatives at two steps, reciprocal charts and incumbent binary limit\n";
}
