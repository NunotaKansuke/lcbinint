#pragma once
// Internal implementation included after Ray and endpoint in gradient_replay.hpp.
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// Validate the primal boundary interpolation against the original equation.
// This uses spatial jets only; no parameter derivatives decide acceptance.
inline bool hermite_boundary_resolved(const RadialPanel<false>& panel,const Geometry& p,bool with_limb,double value_gate=1e-6) {
    const bool weighted=panel.weighted_radial();const auto& map=panel.radial_map();
    const auto& h=hermite_rule(panel.level(),weighted);const std::size_t n=h.x.size();
    using Curves=std::array<std::array<double,4>,3>;std::vector<Curves> values(n);
    const auto& first=panel.retained_radii()[1<<(8-panel.level())];if(!first)return false;
    std::size_t count=0;bool full=first->topology.roots.empty();
    using Intensity=std::array<std::array<double,2>,27>;
    std::vector<Intensity> intensities(with_limb?n:0);
    auto angles=fejer_rule(3).x;
    if(full){angles.push_back(-1);angles.push_back(1);}
    for(std::size_t k=0;k<n;++k) {
        const auto& seed=panel.retained_radii()[int(k+1)<<(8-panel.level())];
        if(!seed)return false;
        const auto& topology=seed->topology;
        if(k==0){count=topology.arcs.size();full=topology.roots.empty();}
        if(count>3 || topology.arcs.size()!=count || full!=topology.roots.empty())return false;
        if(full && !with_limb)continue;
        const double x=h.x[k],shape=(1-x)*(1+x),scale=weighted?std::sqrt(shape):1;
        const auto mapped=weighted?FoldRadialMap{map.a,map.b,false,false}(x):map(x);
        Ray ray(p,seed->radius());
        std::size_t rotation=0;
        if(k && count>1) {
            double best=INFINITY;
            // Arc ordering is cyclic, not a fixed label. A seam crossing
            // permutes the arcs as well as adding a 2pi turn to one arc.
            for(std::size_t shift=0;shift<count;++shift) {
                double cost=0;
                for(std::size_t j=0;j<count;++j) {
                    const auto& arc=topology.arcs[(j+shift)%count];
                    const double d=std::remainder((arc.lo+arc.hi)*.5-values[k-1][j][0],2*std::acos(-1.));cost+=d*d;
                }
                if(cost<best){best=cost;rotation=shift;}
            }
        }
        for(std::size_t j=0;j<count;++j) {
            auto arc=topology.arcs[(j+rotation)%count];
            // Canonical arc ordering can rotate across the angular seam.
            // Use the first radius as a continuous anchor and match modulo 2pi.
            if(k) {
                const double target=values[k-1][j][0],mid=(arc.lo+arc.hi)*.5;
                const double shift=2*std::acos(-1.)*std::round((target-mid)/(2*std::acos(-1.)));
                arc.lo+=shift;arc.hi+=shift;
            }
            const double width=arc.hi-arc.lo;double lr=0,hr=0;
            if(!full) {
                double error=0;const auto lo=ray.sample<false>(arc.lo,error),hi=ray.sample<false>(arc.hi,error);
                if(!(lo.dtheta>0 && hi.dtheta<0))return false;
                lr=-lo.dR/lo.dtheta;hr=-hi.dR/hi.dtheta;
            }
            const double mean=(arc.lo+arc.hi)*.5,half=width*.5;
            values[k][j]={mean,mapped[1]*(lr+hr)*.5,half/scale,
                mapped[1]*(hr-lr)*.5/scale+(weighted?half*x/(shape*scale):0)};
            if(with_limb)for(std::size_t a=0;a<angles.size();++a) {
                const double t=angles[a],theta=mean+half*t;
                double error=0;const auto field=ray.sample<false>(theta,error);
                if(!(field.phi>0 && error<tangent_phi_roundoff*field.phi))return false;
                const double theta_x=mapped[1]*((lr+hr)*.5+t*(hr-lr)*.5);
                const double radial_shape=weighted?shape:1;
                const double root=std::sqrt(field.phi/radial_shape);
                intensities[k][j*angles.size()+a]={root,
                    ((field.dR*mapped[1]+field.dtheta*theta_x)/radial_shape+(weighted?2*x*field.phi/(shape*shape):0))/(2*root)};
            }
        }
    }
    if(full && !with_limb)return true;
    for(const auto& witness:h.witnesses) {
        const auto mapped=weighted?FoldRadialMap{map.a,map.b,false,false}(witness.x):map(witness.x);
        const double scale=weighted?std::sqrt((1-witness.x)*(1+witness.x)):1;Ray ray(p,mapped[0]);
        for(std::size_t j=0;j<count;++j) {
            double mean=0,half=0;
            for(std::size_t k=0;k<n;++k){const auto& c=values[k][j];mean+=witness.value[k]*c[0]+witness.derivative[k]*c[1];half+=witness.value[k]*c[2]+witness.derivative[k]*c[3];}
            half*=scale;if(!(half>0))return false;
            double unused=0;if(!(ray.sample<false>(mean,unused).phi>0))return false;
            if(with_limb)for(std::size_t a=0;a<angles.size();++a) {
                double predicted=0;
                for(std::size_t k=0;k<n;++k) {
                    const auto& c=intensities[k][j*angles.size()+a];
                    predicted+=witness.value[k]*c[0]+witness.derivative[k]*c[1];
                }
                const double radial_shape=weighted?(1-witness.x)*(1+witness.x):1;
                double error=0;const auto field=ray.sample<false>(mean+half*angles[a],error);
                if(!(field.phi>0 && std::abs(predicted*std::sqrt(radial_shape)-std::sqrt(field.phi))+error/std::sqrt(field.phi)<value_gate*std::sqrt(field.phi)))return false;
            }
            if(full)continue;
            for(int sign:{-1,1}) {
                double error=0;const auto jet=ray.sample<false>(mean+sign*half,error);
                // Match the endpoint implementation's geometric meaning:
                // residual / angular slope is an angular displacement.
                if(!(std::abs(jet.phi)+error<value_gate*std::abs(jet.dtheta)*2*half))return false;
            }
        }
    }
    return true;
}
struct HermiteRadius {
    double flux=0,radial_flux=0,limb_flux=0,radial_limb_flux=0;
    std::array<double,8> gradient{},radial_gradient{},limb_gradient{},radial_limb_gradient{};
};
inline bool hermite_radius(double R,const Geometry& p,const AngularTopology& topology,
        const std::array<double,8>& velocity,const std::array<double,8>& slope,
        HermiteRadius& out,ShapeGradientResult& work,bool with_limb=false,int angular_level=4,double value_gate=1e-6) {
    const bool full=topology.roots.empty();
    const MixedRay mixed_ray(R,p);
    for(const auto& arc:topology.arcs) {
        double width=arc.hi-arc.lo,width_r=0;
        std::array<double,8> width_p{},width_rp{};
        std::array<std::array<double,8>,2> angle_p{},angle_rp{};
        Endpoint ends[2];ends[0].theta=arc.lo;ends[1].theta=arc.hi;
        if(!full) {
            std::array<PolarMixedJet,2> jets;
            if(!endpoint(R,p,arc.lo,width,ends[0],work,&jets[0]) || !endpoint(R,p,arc.hi,width,ends[1],work,&jets[1]))return false;
            if(!(ends[0].slope>0 && ends[1].slope<0))return false;
            width=ends[1].theta-ends[0].theta;
            for(int side=0;side<2;++side) {
                const auto& j=jets[side];
                const double tr=-j.r/j.t,sign=side?1:-1;
                width_r+=sign*tr;
                for(int k=0;k<8;++k) {
                    const double tp=-(j.p[k]+j.r*velocity[k])/j.t;
                    const double tpr=-(j.rp[k]+j.tp[k]*tr+(j.rr+j.rt*tr)*velocity[k]+j.r*slope[k]+(j.rt+j.tt*tr)*tp)/j.t;
                    width_p[k]+=sign*tp;width_rp[k]+=sign*tpr;
                    angle_p[side][k]=tp;angle_rp[side][k]=tpr;
                }
            }
        }
        const double flux=R*width,fr=width+R*width_r;
        out.flux+=flux;out.radial_flux+=fr;
        for(int k=0;k<8;++k) {
            out.gradient[k]+=velocity[k]*width+R*width_p[k]+slope[k]*flux;
            out.radial_gradient[k]+=slope[k]*width+velocity[k]*width_r+width_p[k]+R*width_rp[k]+slope[k]*fr;
        }
        if(with_limb) {
            const double lo=ends[0].theta,hi=ends[1].theta,half=(hi-lo)*.5;
            const double br=width_r/width;Ray ray(p,R);
            const auto accumulate=[&](double theta,double weight) {
                const auto q=mixed_ray.sample<false>(theta);++work.evaluations;
                if(!(q.phi>0 && q.roundoff<tangent_phi_roundoff*q.phi))return false;
                const double x=(theta-lo)/width;
                const double B=ends[0].radial_motion*(1-x)+ends[1].radial_motion*x;
                const auto density=limb_mixed_density(q,R,B,br,velocity,slope);
                out.limb_flux+=weight*density.flux;out.radial_limb_flux+=weight*density.radial_flux;
                for(int k=0;k<8;++k) {
                    out.limb_gradient[k]+=weight*(density.gradient[k]+slope[k]*density.flux);
                    out.radial_limb_gradient[k]+=weight*(density.radial_gradient[k]+slope[k]*density.radial_flux);
                }
                return true;
            };
            if(const auto chart=complex_arc_geometry(ray,R,p,lo,hi,!full,work)) {
                if(!chart->valid)return false;
                for(const auto& node:chart->nodes)if(!accumulate(node.theta,node.jac*node.weight))return false;
            } else if(!full) {
                const int maximum=std::max(angular_level,6);
                std::vector<LimbMixedDensity> samples(1<<maximum);std::vector<bool> ready(1<<maximum);
                double previous=0;bool accepted=false;
                std::array<double,8> half_p,half_rp;
                for(int k=0;k<8;++k){half_p[k]=width_p[k]*.5;half_rp[k]=width_rp[k]*.5;}
                for(int level=2;level<=maximum;++level) {
                    const int n=1<<level;const auto& rule=angular_rule_extended(level);LimbMixedDensity total;
                    for(int k=1;k<n;++k) {
                        const int slot=k<<(maximum-level);const double x=rule.x[k-1],shape=(1-x)*(1+x),t=(1+x)*.5;
                        if(!ready[slot]) {
                            const auto q=mixed_ray.sample(lo+half*(1+x));++work.evaluations;
                            if(!(q.phi>0 && q.roundoff<tangent_phi_roundoff*q.phi))return false;
                            std::array<double,8> A,C;
                            for(int j=0;j<8;++j){A[j]=angle_p[0][j]*(1-t)+angle_p[1][j]*t;C[j]=angle_rp[0][j]*(1-t)+angle_rp[1][j]*t;}
                            const double B=ends[0].radial_motion*(1-t)+ends[1].radial_motion*t;
                            samples[slot]=mapped_limb_mixed_density(q,R,half,width_r*.5,half_p,half_rp,B,A,C,velocity,slope,shape);ready[slot]=true;
                        }
                        const double w=rule.norm_w[k-1];const auto& value=samples[slot];
                        total.flux+=w*value.flux;total.radial_flux+=w*value.radial_flux;
                        for(int j=0;j<8;++j){total.gradient[j]+=w*value.gradient[j];total.radial_gradient[j]+=w*value.radial_gradient[j];}
                    }
                    // Only the value controls angular order, as in binary.
                    if(level>=3 && std::abs(total.flux-previous)<=value_gate*std::abs(total.flux)) {
                        out.limb_flux+=total.flux;out.radial_limb_flux+=total.radial_flux;
                        for(int j=0;j<8;++j){out.limb_gradient[j]+=total.gradient[j]+slope[j]*total.flux;out.radial_limb_gradient[j]+=total.radial_gradient[j]+slope[j]*total.radial_flux;}
                        accepted=true;break;
                    }
                    previous=total.flux;
                }
                if(!accepted)return false;
            } else {
                const int level=angular_level,m=1<<level;const auto& angular=angular_rule_extended(level);
                for(int k=1;k<m;++k)if(!accumulate(lo+half*(1+angular.x[k-1]),half*angular.w[k-1]))return false;

            }
        }
    }
    const auto finite=[](const auto& values){return std::all_of(values.begin(),values.end(),[](double x){return std::isfinite(x);});};
    return std::isfinite(out.flux) && std::isfinite(out.radial_flux) && std::isfinite(out.limb_flux) && std::isfinite(out.radial_limb_flux) &&
        finite(out.gradient) && finite(out.radial_gradient) && finite(out.limb_gradient) && finite(out.radial_limb_gradient);
}
} // namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail
