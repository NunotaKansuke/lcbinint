#pragma once
// Research-only cold provider: one native quintic solve plus local critical
// witnesses. No caustic polyline, critical-curve scan or D14 solve.
#include <complex>
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_boundary_events.hpp"
#include "lcbinint/magnification/holonomic/seeded_epoch.hpp"

namespace lcbinint::holonomic {
struct PointSeedProposals {
    std::vector<Cplx<double>> positions;
    int physical_images=0, quintic_solves=0;
    int critical_attempts=0, critical_witnesses=0, critical_rejections=0;
};
namespace point_seed_detail {
using Z=std::complex<double>;
struct CriticalState {double h=0,g=0,hx=0,hy=0,gx=0,gy=0; Z mapped{}; bool valid=false;};
inline CriticalState critical_state(Z z,const PrimaryFrame&pf) {
    Z u=1./z,v=1./(z-pf.a);
    Z k=pf.m0*u*u+(1-pf.m0)*v*v;
    Z kp=-2.*(pf.m0*u*u*u+(1-pf.m0)*v*v*v);
    Z mapped=z-pf.m0*std::conj(u)-(1-pf.m0)*std::conj(v);
    Z residual=mapped-Z(pf.X,pf.Y);
    Z half=std::polar(1.,.5*std::arg(k)),phase_derivative=kp/k;
    Z hg=kp*std::conj(k);
    CriticalState s;
    s.h=std::norm(k)-1;s.g=(residual*half).real();
    s.hx=2*hg.real();s.hy=-2*hg.imag();
    s.gx=((1.+std::conj(k))*half+residual*half*Z(0,.5*phase_derivative.imag())).real();
    s.gy=(Z(0,1)*(1.-std::conj(k))*half+residual*half*Z(0,.5*phase_derivative.real())).real();
    s.mapped=mapped;
    s.valid=std::isfinite(s.h+s.g+s.hx+s.hy+s.gx+s.gy+mapped.real()+mapped.imag());
    return s;
}
inline bool phase_correct(Z&z,double phi,const PrimaryFrame&pf){
    Z target=std::polar(1.,phi);
    for(int it=0;it<12;++it){
        Z u=1./z,v=1./(z-pf.a),k=pf.m0*u*u+(1-pf.m0)*v*v;
        Z kp=-2.*(pf.m0*u*u*u+(1-pf.m0)*v*v*v);
        if(!std::isfinite(std::abs(kp))||std::abs(kp)==0)return false;
        Z step=(k-target)/kp;
        double limit=.3*std::min(std::abs(z),std::abs(z-pf.a));
        if(std::abs(step)>limit)step*=limit/std::abs(step);
        z-=step;
        if(std::abs(step)<8e-14*(1+std::abs(z))){
            Z u=1./z,v=1./(z-pf.a);
            return std::abs(pf.m0*u*u+(1-pf.m0)*v*v-target)<1e-10;
        }
    }return false;
}
inline bool critical_phase_witness(Z&z,const PrimaryFrame&pf){
    Z u=1./z,v=1./(z-pf.a);double phi=std::arg(pf.m0*u*u+(1-pf.m0)*v*v);
    if(!phase_correct(z,phi,pf))return false;
    for(int it=0;it<16;++it){
        auto s=critical_state(z,pf);if(!s.valid)return false;
        // A point already inside the disk is sufficient; distance stationarity
        // is only a way to locate a witness, never an extra acceptance gate.
        if(std::norm(s.mapped-Z(pf.X,pf.Y))<pf.rho*pf.rho)return true;
        Z u=1./z,v=1./(z-pf.a),k=std::polar(1.,phi);
        Z kp=-2.*(pf.m0*u*u*u+(1-pf.m0)*v*v*v),dz=Z(0,1)*k/kp;
        Z half=std::polar(1.,.5*phi),residual=s.mapped-Z(pf.X,pf.Y);
        double g=(residual*half).real();
        double dg=((dz+std::conj(k)*std::conj(dz))*half+residual*half*Z(0,.5)).real();
        if(!std::isfinite(dg)||dg==0)return false;
        double step=std::clamp(g/dg,-.5,.5);bool accepted=false;
        for(int back=0;back<8;++back){
            Z trial=z+dz*(-step);double next_phi=phi-step;
            if(phase_correct(trial,next_phi,pf)){
                auto ns=critical_state(trial,pf);
                double ng=((ns.mapped-Z(pf.X,pf.Y))*std::polar(1.,.5*next_phi)).real();
                if(ns.valid&&std::fabs(ng)<std::fabs(g)){z=trial;phi=next_phi;accepted=true;break;}
            }
            step*=.5;
        }
        if(!accepted)return false;
    }return false;
}

inline bool critical_witness(Z&z,const PrimaryFrame&pf) {
    for(int it=0;it<32;++it){
        auto s=critical_state(z,pf);if(!s.valid)return false;
        double det=s.hx*s.gy-s.hy*s.gx;
        if(!std::isfinite(det)||det==0)return false;
        Z step{(s.h*s.gy-s.hy*s.g)/det,(s.hx*s.g-s.gx*s.h)/det};
        double limit=.3*std::min(std::abs(z),std::abs(z-pf.a));
        if(std::abs(step)>limit)step*=limit/std::abs(step);
        double merit=s.h*s.h+s.g*s.g;
        bool accepted=false;
        for(int back=0;back<12;++back){
            auto trial=critical_state(z-step,pf);
            if(trial.valid&&trial.h*trial.h+trial.g*trial.g<=merit){z-=step;accepted=true;break;}
            step*=.5;
        }
        if(!accepted)return false;
        if(std::abs(step)<8e-14*(1+std::abs(z))){
            s=critical_state(z,pf);
            return s.valid&&std::fabs(s.h)<1e-9&&std::fabs(s.g)<1e-10*(1+pf.rho)&&
                std::norm(s.mapped-Z(pf.X,pf.Y))<pf.rho*pf.rho;
        }
    }
    return false;
}
}
inline PointSeedProposals point_seed_proposals(
    const LensParams&p,lcbinint::magnification::PointSourceMagnifier& point) {
    PointSeedProposals out;
    const auto pf=PrimaryFrame::from(p);
    const lcbinint::SourcePosition source{p.barycentric?p.xs:p.xs-p.m1()*p.a,p.ys};
    auto candidates=point.binary_image_candidates(p.a,p.q,source);++out.quintic_solves;
    const double shift=p.q<1?p.a:0;
    out.positions.reserve(7);
    std::vector<point_seed_detail::Z> attempted;
    for(const auto& candidate:candidates){
        Cplx<double> proposal{candidate.position.x+shift,candidate.position.y};
        if(!std::isfinite(proposal.re+proposal.im))continue;
        out.positions.push_back(proposal);
        if(candidate.physical)++out.physical_images;
        point_seed_detail::Z original{proposal.re,proposal.im};
        auto partner=point_seed_detail::Z(pf.X,pf.Y)+pf.m0/std::conj(original)+(1-pf.m0)/std::conj(original-pf.a);
        auto midpoint=.5*(original+partner);
        // The ghost conjugate-image midpoint approaches the merging pair.
        for(auto initial:{original,midpoint}){
            bool repeated=false;for(auto previous:attempted)if(std::abs(initial-previous)<1e-10*(1+std::abs(initial)))repeated=true;
            if(repeated)continue;attempted.push_back(initial);
            ++out.critical_attempts;auto z=initial;
            if(point_seed_detail::critical_phase_witness(z,pf)){
                bool duplicate=false;
                for(auto old:out.positions)if(std::hypot(old.re-z.real(),old.im-z.imag())<1e-10*(1+std::abs(z)))duplicate=true;
                if(!duplicate){out.positions.push_back({z.real(),z.imag()});++out.critical_witnesses;}
            }else ++out.critical_rejections;
        }
    }
    return out;
}
struct SeededPointEpoch {
    PointSeedProposals proposals;
    SeededBoundaryResult discovery;
    TopologyResult topology;
    AdaptiveResult integral;
};
inline SeededPointEpoch epoch_adaptive_point_seeded(
    const LensParams&p,double u,const AdaptiveConfig&cfg,AdaptiveWorkspace&workspace,
    const SeededBoundaryConfig&discovery_cfg={}) {
    lcbinint::magnification::PointSourceMagnifier point;
    SeededPointEpoch out;
    out.proposals=point_seed_proposals(p,point);
    auto epoch=epoch_adaptive_seeded(p,out.proposals.positions,u,cfg,workspace,discovery_cfg);
    out.discovery=std::move(epoch.discovery);
    out.topology=std::move(epoch.topology);
    out.integral=std::move(epoch.integral);
    return out;
}
}
