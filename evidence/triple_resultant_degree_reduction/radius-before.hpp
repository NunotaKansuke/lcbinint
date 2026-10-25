#pragma once
#include "angular_topology.hpp"
#include "local_precision.hpp"
#include "../nested_fejer2.hpp"
#include <string>
#include <map>

namespace lcbinint::algebraic_boundary::triple {
inline const Fejer2Rule& angular_rule_extended(int level) {
    if(level<=8)return fejer_rule(level);
    if(level==9){static const Fejer2Rule rule(9);return rule;}
    if(level==10){static const Fejer2Rule rule(10);return rule;}
    throw std::invalid_argument("triple angular level must be 1..10");
}
// The workspace belongs to a single immutable radius/geometry/topology.
// Value and gradient workspaces are different instantiations: value calls
// contain no derivative samples or endpoint differentiation.
template<bool Gradient> struct RadiusSample;
template<> struct RadiusSample<false> {double value=0,roundoff=0;bool ready=false;};
template<> struct RadiusSample<true> {
    double value=0,roundoff=0;bool ready=false;std::array<double,8> dp{};
};
template<bool Gradient> struct RadiusIntegral;
template<> struct RadiusIntegral<false> {
    double f0=0,fh=0,angular_error=0,endpoint_error=0;
};
template<> struct RadiusIntegral<true> {
    double f0=0,fh=0,angular_error=0,endpoint_error=0;
    std::array<double,8> df0{},dfh{},gradient_error{};
};
template<bool Gradient=false> class RadiusWorkspace {
    Geometry geometry_{};
    double radius_=0;
    bool initialized_=false;
    bool uniform_source_=false;
    double sample_relative_tolerance_=0;
    double fixed_radial_=0,primary_deflection_=0,primary_shear_=0,source_norm_=0;
    Evaluation<false> fixed_value(double theta)const {
        if(!std::isfinite(fixed_radial_) || !std::isfinite(primary_shear_))
            return evaluate<false,true>(radius_,theta,geometry_);
        const Complex unit(std::cos(theta),std::sin(theta)),z=radius_*unit;
        Complex g=fixed_radial_*unit-Complex(geometry_.X,geometry_.Y);
        double deflection=primary_deflection_,shear=primary_shear_;
        const std::array<Complex,2> lenses{Complex(geometry_.a,0),Complex(geometry_.bx,geometry_.by)};
        const std::array<double,2> masses{geometry_.m1,geometry_.m2};
        for(int j=0;j<2;++j) {
            if(masses[j]==0)continue; // Removable zero-mass poles.
            const auto inverse=inverse_conjugate(z-lenses[j]);
            g-=masses[j]*inverse;deflection+=masses[j]*std::abs(inverse);shear+=masses[j]*std::norm(inverse);
        }
        const double rho2=geometry_.rho*geometry_.rho,norm=std::norm(g),size=std::abs(g);
        const double eps=std::numeric_limits<double>::epsilon();
        const double delta=32*eps*(radius_+source_norm_+deflection+radius_*shear);
        return {1-norm/rho2,(2*size*delta+delta*delta)/rho2+16*eps*(1+norm/rho2),
            2*size*radius_*(1+shear)/rho2};
    }

    std::vector<std::vector<RadiusSample<Gradient>>> samples_;
    RadiusIntegral<Gradient> accepted_{};
    std::vector<std::array<RadiusSample<Gradient>,2>> endpoints_;
    std::vector<local_precision::Arc<Gradient>> precise_arcs_;
    std::shared_ptr<const local_precision::ValueRay> precise_ray_;
    const local_precision::ValueRay& precise_ray() {
        if(!precise_ray_)precise_ray_=std::make_shared<const local_precision::ValueRay>(radius_,geometry_);
        return *precise_ray_;
    }
    struct RepairCell {std::size_t arc;double lo,hi;int level=3;RadiusIntegral<Gradient> result;};
    std::vector<RepairCell> repair_cells_;
    std::map<std::pair<std::size_t,double>,RadiusSample<Gradient>> repair_samples_;
    bool repaired_=false;
    int level_=0;
    bool endpoints_ready_=false,precision_restart_=false;
public:
    AngularTopology topology;
    bool continued_topology_used=false;
    std::string reason="not_started";
    std::size_t value_evaluations=0,gradient_evaluations=0,reused_samples=0,quad_samples=0;
    bool initialize(double R,const Geometry& p,AngularOptions options={},bool uniform_source=false,
                    double sample_relative_tolerance=0,AngularCoefficientCache* coefficients=nullptr,
                    const AngularTopology* inherited=nullptr,int certified_count=-1) {
        // Explicit reinitialization discards geometry-dependent samples.
        *this=RadiusWorkspace{};
        geometry_=p;radius_=R;uniform_source_=uniform_source;
        if(inherited)topology=continued_topology(R,p,*inherited,options,certified_count,Gradient?nullptr:coefficients);
        continued_topology_used=inherited && topology.completed;
        if(!continued_topology_used)topology=angular_topology(R,p,options,coefficients);
        if(!topology.completed && options.endpoint_tolerance>1e-12 &&
           (std::string(topology.reason)=="overlapping_root_brackets" ||
            std::string(topology.reason)=="chart_endpoint_root" ||
            std::string(topology.reason)=="ambiguous_arc_witness" ||
            std::string(topology.reason)=="exclusion_root_ambiguous")) {
            // Repair only this unresolved radius. All other radii and their
            // accumulated samples remain in the shared workspace.
            const auto previous=topology;
            auto tight=options;tight.endpoint_tolerance=1e-12;
            topology=angular_topology(R,p,tight,coefficients);
            topology.double_boxes+=previous.double_boxes;
            topology.quad_boxes+=previous.quad_boxes;
            topology.quad_promotions+=previous.quad_promotions;
        }
        sample_relative_tolerance_=sample_relative_tolerance;
        if constexpr(!Gradient)if(sample_relative_tolerance>0) {
            const double primary=1-p.m1-p.m2;
            primary_deflection_=primary/R;primary_shear_=primary_deflection_/R;
            fixed_radial_=R-primary_deflection_;source_norm_=std::hypot(p.X,p.Y);
        }

        reason=topology.reason;
        if(!topology.completed)return false;
        samples_.resize(topology.arcs.size());precise_arcs_.resize(topology.arcs.size());
        if constexpr(Gradient)endpoints_.resize(topology.arcs.size());
        initialized_=true;
        return true;
    }
    bool prepare_precise_arc(std::size_t a) {
        auto& q=precise_arcs_[a];
        if(q.ready) {
            if constexpr(Gradient)for(int j=0;j<8;++j) {
                endpoints_[a][0].dp[j]=double(q.dlo[j]);endpoints_[a][1].dp[j]=double(q.dhi[j]);
            }
            return true;
        }
        const auto arc=topology.arcs[a];
        const auto endpoint=[&](double theta,local_precision::Q& result) {
            if constexpr(Gradient)return local_precision::endpoint(radius_,theta,geometry_,result);
            else return local_precision::endpoint(precise_ray(),theta,geometry_,result);
        };
        if(!endpoint(arc.lo,q.lo) || !endpoint(arc.hi,q.hi) || !(q.hi>q.lo)) {
            reason="precise_endpoint_unresolved";return false;
        }
        if constexpr(Gradient) {
            const auto lo=local_precision::evaluate<true>(radius_,q.lo,geometry_);
            const auto hi=local_precision::evaluate<true>(radius_,q.hi,geometry_);
            gradient_evaluations+=2;
            for(int j=0;j<8;++j) {
                q.dlo[j]=-lo.dp[j]/lo.dtheta;q.dhi[j]=-hi.dp[j]/hi.dtheta;
                endpoints_[a][0].dp[j]=double(q.dlo[j]);endpoints_[a][1].dp[j]=double(q.dhi[j]);
            }
        }
        q.ready=true;
        if constexpr(Gradient) {
            // A late endpoint upgrade changes the moving angular map. Keep
            // every sampled node, upgrade this arc only, and rebuild nested
            // sums from those retained samples before trusting a difference.
            auto& cached=samples_[a];
            if(level_>0)precision_restart_=true;
            if(!cached.empty()) {
                int level=0;for(std::size_t n=cached.size();n>1;n>>=1)++level;
                for(std::size_t k=1;k<cached.size();++k)if(cached[k].ready) {
                    if(!precise_sample(a,fejer_node(level,int(k)),cached[k]))return false;
                    ++gradient_evaluations;
                }
            }
        }
        return true;
    }
    bool precise_sample(std::size_t a,double x,RadiusSample<Gradient>& sample) {
        auto& arcq=precise_arcs_[a];
                            using Q=local_precision::Q;
                            const Q halfq=(arcq.hi-arcq.lo)/2,xq=Q(x);
                            const Q thetaq=arcq.lo+halfq*(1+xq),shapeq=(1-xq)*(1+xq);
                            const auto q=[&] {
                                if constexpr(Gradient)return local_precision::evaluate<true>(Q(radius_),thetaq,geometry_);
                                else return precise_ray().evaluate(thetaq);
                            }();
                            ++quad_samples;
                            if(!(q.phi>0) || !finiteq(q.phi)) {
                                reason="precise_angular_sample_outside";return false;
                            }
                            const Q root=sqrtq(q.phi/shapeq);
                            sample.value=double(Q(radius_)*root*halfq);
                            if constexpr(Gradient)for(int j=0;j<8;++j) {
                                const Q theta_p=(arcq.dlo[j]*(1-xq)+arcq.dhi[j]*(1+xq))/2;
                                sample.dp[j]=double(Q(radius_)*(halfq*(q.dp[j]+q.dtheta*theta_p)/(2*root*shapeq)
                                    +root*(arcq.dhi[j]-arcq.dlo[j])/2));
                            }
        return true;
    }
    bool repair(double tolerance,int budget=1024) {
        if(uniform_source_)return true;
        if(!initialized_ || !(tolerance>0)){reason="invalid_repair";return false;}
        const bool full=topology.roots.empty();
        if constexpr(Gradient)if(!full)for(std::size_t a=0;a<topology.arcs.size();++a)
            if(!prepare_precise_arc(a))return false;
        if(repair_cells_.empty())for(std::size_t a=0;a<topology.arcs.size();++a)
            repair_cells_.push_back({a,-1,1,3,{}});
        auto cell_integral=[&](RepairCell& cell,int level,RadiusIntegral<Gradient>& out) {
            const auto& rule=angular_rule_extended(level);
            const auto arc=topology.arcs[cell.arc];
            const double half=(arc.hi-arc.lo)*.5,mid=arc.lo+half;
            const double width=(cell.hi-cell.lo)*.5,center=cell.lo+width,pi=std::acos(-1.);
            for(int k=1;k<(1<<level);++k) {
                const double x=center+width*rule.x[k-1];
                auto& sample=repair_samples_[{cell.arc,x}];
                if(!sample.ready) {
                    const double sine=std::sin(pi*.5*x),cosine=std::cos(pi*.5*x);
                    const double theta=mid+half*sine,jac=half*pi*.5*cosine;
                    const auto jet=evaluate<Gradient>(radius_,theta,geometry_);
                    ++value_evaluations;if constexpr(Gradient)++gradient_evaluations;
                    const bool precise=jet.phi<1e-7 || half<.01;
                    if(precise) {
                        if(!full && !prepare_precise_arc(cell.arc))return false;
                        using Q=local_precision::Q;
                        const Q lo=full?Q(arc.lo):precise_arcs_[cell.arc].lo;
                        const Q hi=full?Q(arc.hi):precise_arcs_[cell.arc].hi;
                        const Q h=(hi-lo)/2;static const Q pq=acosq(Q(-1));
                        Q sn,cs;sincosq(pq*Q(.5)*Q(x),&sn,&cs);
                        const Q th=lo+h*(1+sn),jacq=h*pq*Q(.5)*cs;
                        const auto q=[&] {
                            if constexpr(Gradient)return local_precision::evaluate<true>(Q(radius_),th,geometry_);
                            else return precise_ray().evaluate(th);
                        }();++quad_samples;
                        if(!(q.phi>0) || !finiteq(q.phi)){reason="precise_repair_outside";return false;}
                        const Q root=sqrtq(q.phi);sample.value=double(Q(radius_)*root*jacq);
                        if constexpr(Gradient)for(int j=0;j<8;++j) {
                            const Q dl=full?Q(0):precise_arcs_[cell.arc].dlo[j];
                            const Q dh=full?Q(0):precise_arcs_[cell.arc].dhi[j];
                            const Q theta_p=(dl*(1-sn)+dh*(1+sn))/2;
                            const Q jac_p=(dh-dl)*pq*Q(.25)*cs;
                            sample.dp[j]=double(Q(radius_)*((q.dp[j]+q.dtheta*theta_p)*jacq/(2*root)+root*jac_p));
                        }
                    } else {
                        if(!(jet.phi>0) || !std::isfinite(jet.phi)){reason="repair_sample_outside";return false;}
                        const double root=std::sqrt(jet.phi);sample.value=radius_*root*jac;
                        if constexpr(Gradient)for(int j=0;j<8;++j) {
                            const double dl=full?0:double(precise_arcs_[cell.arc].dlo[j]);
                            const double dh=full?0:double(precise_arcs_[cell.arc].dhi[j]);
                            const double theta_p=(dl*(1-sine)+dh*(1+sine))*.5;
                            const double jac_p=(dh-dl)*pi*.25*cosine;
                            sample.dp[j]=radius_*((jet.dp[j]+jet.dtheta*theta_p)*jac/(2*root)+root*jac_p);
                        }
                    }
                    sample.ready=true;
                } else ++reused_samples;
                out.fh+=width*rule.w[k-1]*sample.value;
                if constexpr(Gradient)for(int j=0;j<8;++j)out.dfh[j]+=width*rule.w[k-1]*sample.dp[j];
            }
            return true;
        };
        auto update=[&](RepairCell& cell) {
            RadiusIntegral<Gradient> coarse{},fine{};
            if(!cell_integral(cell,cell.level-1,coarse) || !cell_integral(cell,cell.level,fine))return false;
            fine.angular_error=std::abs(fine.fh-coarse.fh);
            if constexpr(Gradient)for(int j=0;j<8;++j)fine.gradient_error[j]=std::abs(fine.dfh[j]-coarse.dfh[j]);
            cell.result=fine;return true;
        };
        for(auto& cell:repair_cells_)if(!update(cell))return false;
        for(int iteration=0;iteration<budget;++iteration) {
            RadiusIntegral<Gradient> total{};total.f0=accepted_.f0;
            total.endpoint_error=accepted_.endpoint_error;
            if constexpr(Gradient)for(std::size_t a=0;a<topology.arcs.size();++a)for(int j=0;j<8;++j)
                if(!full)total.df0[j]+=radius_*double(precise_arcs_[a].dhi[j]-precise_arcs_[a].dlo[j]);
            std::size_t worst=0;double largest=-1;
            for(std::size_t a=0;a<repair_cells_.size();++a) {
                const auto& value=repair_cells_[a].result;
                total.fh+=value.fh;total.angular_error+=value.angular_error;
                double score=value.angular_error;
                if constexpr(Gradient)for(int j=0;j<8;++j) {
                    total.dfh[j]+=value.dfh[j];total.gradient_error[j]+=value.gradient_error[j];
                    score=std::max(score,value.gradient_error[j]);
                }
                if(score>largest){largest=score;worst=a;}
            }
            if constexpr(Gradient)if(geometry_.Y==0 && geometry_.by==0)
                for(int j:{1,7}){total.df0[j]=0;total.dfh[j]=0;total.gradient_error[j]=0;}
            double error=total.angular_error;
            if constexpr(Gradient)for(double e:total.gradient_error)error=std::max(error,e);
            if(error<tolerance){accepted_=total;repaired_=true;reason="angular_repaired";return true;}
            auto& cell=repair_cells_[worst];
            if(cell.level<5) {++cell.level;if(!update(cell))return false;}
            else {
                const double mid=cell.lo+(cell.hi-cell.lo)*.5;
                RepairCell right{cell.arc,mid,cell.hi,3,{}};
                cell.hi=mid;cell.level=3;
                if(!update(cell) || !update(right))return false;
                repair_cells_.push_back(std::move(right));
            }
        }
        reason="angular_repair_budget";return false;
    }
    int level() const {return level_;}
    const RadiusIntegral<Gradient>& result() const {return accepted_;}
    // Reuse every sample. A local relative budget prevents easy radii from
    // following a difficult sibling to the panel's full angular ceiling.
    // Higher ceilings eventually tighten it to avoid a fixed error floor.
    bool refine_local(int ceiling,double relative,double absolute) {
        if(ceiling<2 || ceiling>10){reason="invalid_level";return false;}
        if constexpr(Gradient)return refine(ceiling);
        if(repaired_)return true;
        if(uniform_source_ || !(relative>0))return refine(ceiling);
        if(!refine(2))return false;
        const double factor=std::ldexp(1.,std::min(0,6-ceiling));
        while(level_<ceiling) {
            const double budget=factor*(absolute+relative*std::abs(accepted_.fh));
            if(std::isfinite(accepted_.angular_error) && accepted_.angular_error<=budget)break;
            if(!refine(level_+1))return false;
        }
        return true;
    }
    bool refine(int requested_level) {
        if(!initialized_){reason="topology_incomplete";return false;}
        if(requested_level<2 || requested_level>10) {reason="invalid_level";return false;}
        if(repaired_ || requested_level<=level_)return true;
        // Process every intervening level so coarse/fine error estimates use
        // adjacent rules; existing canonical samples are never re-evaluated.
        for(int lev=std::max((Gradient || uniform_source_)?2:1,level_+1);lev<=requested_level;++lev) {
            RadiusIntegral<Gradient> next{};
            next.endpoint_error=radius_*topology.angular_measure_error;
            const bool full_circle=topology.roots.empty();
            if constexpr(Gradient) {
                bool close_roots=false;
                for(std::size_t k=0;k<topology.roots.size();++k) {
                    const auto& a=topology.roots[k];
                    const auto& b=topology.roots[(k+1)%topology.roots.size()];
                    const double gap=(b.lo+b.hi-a.lo-a.hi)*.5+
                        (k+1==topology.roots.size()?2*std::acos(-1.):0);
                    close_roots=close_roots || gap<.02;
                }
                if(!endpoints_ready_ && !full_circle) {
                    for(std::size_t a=0;a<topology.arcs.size();++a) {
                        const auto arc=topology.arcs[a];
                        for(int side=0;side<2;++side) {
                            const double theta=side?arc.hi:arc.lo;
                            const auto jet=evaluate<true>(radius_,theta,geometry_);
                            ++gradient_evaluations;
                            if(!std::isfinite(jet.dtheta) || std::abs(jet.dtheta)<1e-14) {
                                reason="endpoint_derivative_unresolved";return false;
                            }
                            for(int j=0;j<8;++j)endpoints_[a][side].dp[j]=-jet.dp[j]/jet.dtheta;
                        }
                        if((precise_arcs_[a].ready || close_roots || arc.hi-arc.lo<.02) && !prepare_precise_arc(a))return false;
                    }
                }
                for(std::size_t a=0;a<topology.arcs.size();++a)for(int j=0;j<8;++j)
                    next.df0[j]+=radius_*(precise_arcs_[a].ready?
                        double(precise_arcs_[a].dhi[j]-precise_arcs_[a].dlo[j]):
                        endpoints_[a][1].dp[j]-endpoints_[a][0].dp[j]);
            }
            const auto& rule=angular_rule_extended(lev);
            for(std::size_t a=0;a<topology.arcs.size();++a) {
                auto arc=topology.arcs[a];
                next.f0+=radius_*(arc.hi-arc.lo);
                if(uniform_source_)continue;
                auto& cache=samples_[a];
                if(cache.size()<std::size_t(1<<lev)) {
                    const auto old=cache.size();cache.resize(1<<lev);
                    if(old)for(std::size_t j=old;j-->0;) {
                        cache[2*j]=std::move(cache[j]);
                        if(2*j!=j)cache[j]=RadiusSample<Gradient>{};
                    }
                }
                const double half=(arc.hi-arc.lo)*.5,mid=arc.lo+half;
                for(int k=1;k<(1<<lev);++k) {
                    const int slot=k*int(cache.size()/(1<<lev));
                    auto& sample=samples_[a][slot];
                    if(!sample.ready) {
                        // Physical arcs have a sqrt(1-x*x) factor. The
                        // incumbent nested Chebyshev-II rule integrates that
                        // weight directly, avoiding nodes crowded at O(n^-4)
                        // distances from the physical endpoints.
                        const double x=rule.x[k-1],theta=mid+half*x;
                        const double shape=full_circle?1:(1-x)*(1+x);
                        const auto jet=[&] {
                            if constexpr(!Gradient)if(sample_relative_tolerance_>0)return fixed_value(theta);
                            return evaluate<Gradient>(radius_,theta,geometry_);
                        }();
                        ++value_evaluations;
                        if constexpr(Gradient)++gradient_evaluations;
                        bool needs_precision=!full_circle &&
                            (jet.phi<1e-7 || half<.01 || (Gradient && precise_arcs_[a].ready));
                        double phi_error=0;
                        if constexpr(!Gradient)if(sample_relative_tolerance_>0) {
                            phi_error=jet.roundoff+jet.angular_sensitivity*topology.angular_measure_error;
                            needs_precision=!full_circle &&
                                (!(jet.phi>0) || phi_error>sample_relative_tolerance_*jet.phi);
                        }
                        if(needs_precision) {
                            if(!prepare_precise_arc(a))return false;
                            if(!precise_sample(a,x,sample))return false;
                        } else {
                            if(!(jet.phi>0) || !std::isfinite(jet.phi)) {
                                reason="angular_sample_outside";return false;
                            }
                            const double root=std::sqrt(jet.phi/shape);
                            sample.value=radius_*root*half;
                            if constexpr(!Gradient)if(phi_error>0)
                                sample.roundoff=std::abs(sample.value)*phi_error/jet.phi;
                            if constexpr(Gradient)for(int j=0;j<8;++j) {
                                const double dlo=endpoints_[a][0].dp[j],dhi=endpoints_[a][1].dp[j];
                                const double theta_p=(dlo*(1-x)+dhi*(1+x))*.5;
                                sample.dp[j]=radius_*(half*(jet.dp[j]+jet.dtheta*theta_p)/(2*root*shape)
                                    +root*(dhi-dlo)*.5);
                            }
                        }
                        sample.ready=true;
                    } else ++reused_samples;
                    next.fh+=(full_circle?rule.w[k-1]:rule.norm_w[k-1])*sample.value;
                    next.endpoint_error+=(full_circle?rule.w[k-1]:rule.norm_w[k-1])*sample.roundoff;
                    if constexpr(Gradient)for(int j=0;j<8;++j)
                        next.dfh[j]+=(full_circle?rule.w[k-1]:rule.norm_w[k-1])*sample.dp[j];
                }
            }
            if constexpr(Gradient) {
                next.df0.fill(0);
                for(std::size_t a=0;a<topology.arcs.size();++a)for(int j=0;j<8;++j)
                    next.df0[j]+=radius_*(precise_arcs_[a].ready?
                        double(precise_arcs_[a].dhi[j]-precise_arcs_[a].dlo[j]):
                        endpoints_[a][1].dp[j]-endpoints_[a][0].dp[j]);
            }
            if constexpr(Gradient)if(geometry_.Y==0 && geometry_.by==0) {
                // Reflection of an exactly axial source and collinear lenses
                // makes both transverse derivatives identically zero at every
                // radius; do not chase cancellation noise between mirror arcs.
                for(int j:{1,7}){next.df0[j]=0;next.dfh[j]=0;}
            }
            if(precision_restart_) {
                precision_restart_=false;accepted_={};level_=0;lev=1;continue;
            }
            next.angular_error=uniform_source_?0:level_?std::abs(next.fh-accepted_.fh):INFINITY;
            if constexpr(Gradient)for(int j=0;j<8;++j)
                next.gradient_error[j]=uniform_source_?0:level_?std::abs(next.dfh[j]-accepted_.dfh[j]):INFINITY;
            if constexpr(Gradient)if(geometry_.Y==0 && geometry_.by==0)
                for(int j:{1,7})next.gradient_error[j]=0;
            // Publish only a complete level. Samples collected before a failed
            // level remain available for a retry without changing accepted_.
            accepted_=next;level_=lev;endpoints_ready_=true;
        }
        reason="completed";return true;
    }
};
} // namespace lcbinint::algebraic_boundary::triple
