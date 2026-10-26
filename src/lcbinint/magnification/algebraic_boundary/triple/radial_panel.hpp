#pragma once
#include "radius_terms.hpp"
#include "../quadrature_error_policy.hpp"
#include <memory>
#include <map>

namespace lcbinint::algebraic_boundary::triple {
namespace panel_detail {
// Reuse all existing samples to embed a positive (n-2)-node rule in
// the unweighted n-node Fejer rule. The omitted nodes remain independent
// interpolation witnesses. Weighted Jacobi panels keep their incumbent rule.
struct EmbeddedRadialRule {
    std::vector<int> slots;
    std::vector<double> weights;
    std::array<std::vector<double>,2> interpolation;
    std::array<int,2> missing;
    explicit EmbeddedRadialRule(int level) {
        const auto& rule=fejer_rule(level);const int n=int(rule.x.size());
        missing={n/2-1,n/2+1};
        for(int j=0;j<n;++j)if(j!=missing[0] && j!=missing[1])slots.push_back(j);
        const int count=int(slots.size());
        std::vector<std::vector<long double>> matrix(count,std::vector<long double>(count+1));
        matrix[0][count]=2;
        for(int j=0;j<count;++j) {
            const long double x=rule.x[slots[j]];long double previous=1,current=x;
            matrix[0][j]=1;matrix[1][j]=x;
            for(int k=2;k<count;++k) {
                const long double next=((2*k-1)*x*current-(k-1)*previous)/k;
                matrix[k][j]=next;previous=current;current=next;
            }
        }
        for(int k=0;k<count;++k) {
            int pivot=k;
            for(int i=k+1;i<count;++i)if(std::abs(matrix[i][k])>std::abs(matrix[pivot][k]))pivot=i;
            std::swap(matrix[k],matrix[pivot]);
            if(!(std::abs(matrix[k][k])>1e-20L))throw std::logic_error("embedded radial moment rank");
            const auto scale=matrix[k][k];for(int j=k;j<=count;++j)matrix[k][j]/=scale;
            for(int i=0;i<count;++i)if(i!=k) {
                const auto factor=matrix[i][k];
                for(int j=k;j<=count;++j)matrix[i][j]-=factor*matrix[k][j];
            }
        }
        weights.resize(count);
        for(int j=0;j<count;++j) {
            weights[j]=double(matrix[j][count]);
            if(!(weights[j]>0))throw std::logic_error("embedded radial positive weights");
        }
        for(int k=0;k<2;++k) {
            interpolation[k].resize(count);
            for(int j=0;j<count;++j) {
                long double value=1;
                for(int i=0;i<count;++i)if(i!=j)
                    value*=((long double)rule.x[missing[k]]-rule.x[slots[i]])/
                           ((long double)rule.x[slots[j]]-rule.x[slots[i]]);
                interpolation[k][j]=double(value);
            }
        }
    }
};
inline const EmbeddedRadialRule& embedded_radial_rule(int level) {
    if(level==3){static const EmbeddedRadialRule rule(3);return rule;}
    if(level==4){static const EmbeddedRadialRule rule(4);return rule;}
    if(level==5){static const EmbeddedRadialRule rule(5);return rule;}
    throw std::invalid_argument("embedded radial level");
}
template<class Samples,class Accessor> std::pair<double,double> embedded_estimate(
        int level,const Samples& samples,Accessor value) {
    const auto& embedded=embedded_radial_rule(level);
    const auto& rule=fejer_rule(level);double coarse=0,detail=0;
    for(std::size_t j=0;j<embedded.slots.size();++j)coarse+=embedded.weights[j]*value(samples[embedded.slots[j]]);
    for(int k=0;k<2;++k) {
        double predicted=0;
        for(std::size_t j=0;j<embedded.slots.size();++j)predicted+=embedded.interpolation[k][j]*value(samples[embedded.slots[j]]);
        detail+=rule.w[embedded.missing[k]]*std::abs(value(samples[embedded.missing[k]])-predicted);
    }
    return {coarse,detail};
}
template<class Samples,class Accessor> double interpolation_detail(
    const Fejer2Rule& rule,const Samples& samples,Accessor value) {
    const int m=int(rule.x.size())+1,nc=m/2-1;
    double detail=0;
    for(int k=1;k<m;k+=2) {
        double predicted=0;
        for(int j=1;j<=nc;++j)
            predicted+=rule.interp[(k/2)*nc+j-1]*value(samples[2*j-1]);
        detail+=rule.w[k-1]*std::abs(value(samples[k-1])-predicted);
    }
    return detail;
}
}

// Integrate a FIXED radial panel. This is a building block for the frontier,
// not a full-support solver. Parameter-dependent endpoints require the frontier
// to supply event motion; do not use this fixed-panel gradient across them.
template<bool Gradient> struct SharedRadiusCache {
    std::map<double,std::shared_ptr<RadiusWorkspace<Gradient>>> radii;
    std::optional<std::array<double,8>> identity;
    AngularOptions angular;
    std::optional<bool> uniform_source;
    double sample_relative_tolerance=0;
    std::shared_ptr<AngularCoefficientCache> coefficients;

};
template<bool Gradient=false> class RadialPanel {
    Geometry geometry_;
    FoldRadialMap map_;
    AngularOptions angular_options_;
    std::array<std::shared_ptr<RadiusWorkspace<Gradient>>,256> radii_{};
    std::shared_ptr<SharedRadiusCache<Gradient>> shared_;
    RadiusIntegral<Gradient> accepted_{};
    int level_=0,angular_level_=0;
    double radial_error_=INFINITY,inner_error_=INFINITY;
    double repair_absolute_=0,repair_relative_=0;
    bool uniform_source_=false,weighted_=false;
    double error_uniform_weight_=1,error_limb_weight_=1;
    bool profile_error_=false;
    std::optional<AngularTopology> band_topology_;
    double safe_lo_=0,safe_hi_=0;
public:
    std::string reason="not_started";
    std::size_t radial_evaluations=0,reused_radii=0;
    RadialPanel(const Geometry& p,double lo,double hi,bool left_fold=false,
                bool right_fold=false,AngularOptions options={},
                std::shared_ptr<SharedRadiusCache<Gradient>> shared=nullptr,
                double repair_absolute=0,double repair_relative=0,bool uniform_source=false,bool weighted=false,double error_profile=-1)
        :geometry_(p),map_{lo,hi,left_fold,right_fold},angular_options_(options),shared_(std::move(shared)),
         repair_absolute_(repair_absolute),repair_relative_(repair_relative),uniform_source_(uniform_source),weighted_(weighted) {
        if(Gradient && weighted_)throw std::invalid_argument("weighted value panel cannot carry gradients");
        p.validate();
        // Component errors combine with the same nonnegative coefficients
        // as flux. The global difference safety and detail floor stay intact.
        // A negative profile selects the incumbent standalone/gradient policy.
        if constexpr(!Gradient)if(error_profile>=0) {
            if(!std::isfinite(error_profile) || error_profile>1)throw std::invalid_argument("invalid error profile");
            error_uniform_weight_=1-error_profile;error_limb_weight_=error_profile;profile_error_=true;
        }
        if(shared_) {
            const std::array<double,8> identity{p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};
            if(shared_->identity && (*shared_->identity!=identity ||
               shared_->angular.endpoint_tolerance!=options.endpoint_tolerance ||
               shared_->angular.maximum_boxes!=options.maximum_boxes ||
               shared_->angular.allow_quad!=options.allow_quad ||
               shared_->angular.fast_roots!=options.fast_roots ||
               shared_->angular.exclusions!=options.exclusions ||
               shared_->angular.inclusion!=options.inclusion ||
               (shared_->uniform_source && *shared_->uniform_source!=uniform_source) ||
               shared_->sample_relative_tolerance!=repair_relative*.2))
                throw std::invalid_argument("triple radius cache identity mismatch");
            if(!shared_->coefficients)shared_->coefficients=std::make_shared<AngularCoefficientCache>(p);
            shared_->identity=identity;shared_->angular=options;shared_->uniform_source=uniform_source;
            shared_->sample_relative_tolerance=repair_relative*.2;
        }
        if(!(lo>0 && hi>lo) || !std::isfinite(hi))
            throw std::invalid_argument("invalid triple radial panel");
    }
    void set_certified_band(const AngularTopology& topology,double lo,double hi) {
        if(!topology.completed || !(hi>lo))return;
        band_topology_=topology;safe_lo_=lo;safe_hi_=hi;
    }
    const std::optional<AngularTopology>& band_topology() const{return band_topology_;}
    double safe_lo() const{return safe_lo_;}
    double safe_hi() const{return safe_hi_;}
    int level() const {return level_;}
    double radial_error() const{return radial_error_;}
    double inner_error() const{return inner_error_;}
    const RadiusIntegral<Gradient>& result() const {return accepted_;}
    std::size_t value_evaluations() const {
        std::size_t count=0;for(const auto& r:radii_)if(r)count+=r->value_evaluations;return count;
    }
    // Endpoint maps and canonical nodes are shared with the binary solver.
    // Increasing angular accuracy updates the existing radial workspaces.
    bool refine(int radial_level,int angular_level) {
        if(radial_level<2 || radial_level>8 || angular_level<2 || angular_level>10) {
            reason="invalid_level";return false;
        }
        if(radial_level<=level_ && angular_level<=angular_level_)return true;
        const int target=std::max(level_,radial_level);
        const int angular_target=std::max(angular_level_,angular_level);
        const bool angular_changed=angular_target>angular_level_;
        // Rebuild the adjacent coarse estimate after an angular refinement;
        // geometry and sample caches remain intact.
        const int minimum=(weighted_ || !Gradient)?1:2;
        const int first=angular_changed?std::max(minimum,level_-1):std::max(minimum,level_+1);
        RadiusIntegral<Gradient> previous=accepted_;
        bool have_previous=!angular_changed && level_>0;
        for(int lev=first;lev<=target;++lev) {
            RadiusIntegral<Gradient> next{};
            const auto& rule=radial_rule(lev,weighted_);
            const auto& limb_rule=weighted_?linear_radial_rule(lev):rule;
            std::vector<RadiusIntegral<Gradient>> integrands(rule.x.size());
            double inner_error=0;
            RadiusIntegral<Gradient> inner_errors{};
            for(int k=1;k<(1<<lev);++k) {
                const int slot=k<<(8-lev);
                auto mapped=weighted_?FoldRadialMap{map_.a,map_.b,false,false}(rule.x[k-1]):map_(rule.x[k-1]);
                if(!Gradient && rule.x[k-1]==0)mapped[0]=map_.a+(map_.b-map_.a)*.5;
                const double shape=(1-rule.x[k-1])*(1+rule.x[k-1]);
                const double scale0=weighted_?1/std::sqrt(shape):1;
                const double scaleh=weighted_?1/shape:1;
                const double R=mapped[0],weight=rule.w[k-1]*mapped[1]*scale0;
                const double weighth=limb_rule.w[k-1]*mapped[1]*scaleh;
                if(!radii_[slot]) {
                    if(shared_) {
                        auto found=shared_->radii.find(R);
                        if(found!=shared_->radii.end())radii_[slot]=found->second;
                    }
                    if(!radii_[slot]) {
                        radii_[slot]=std::make_shared<RadiusWorkspace<Gradient>>();
                        const AngularTopology* inherited=nullptr;
                        if(band_topology_ && R>safe_lo_ && R<safe_hi_) {
                            inherited=&*band_topology_;
                            double distance=INFINITY;
                            if(shared_) {
                                auto next=shared_->radii.lower_bound(R);
                                auto consider=[&](const auto& entry) {
                                    if(entry.first>safe_lo_ && entry.first<safe_hi_ && entry.second->topology.completed &&
                                       entry.second->topology.roots.size()==band_topology_->roots.size() &&
                                       std::abs(entry.first-R)<distance) {
                                        inherited=&entry.second->topology;distance=std::abs(entry.first-R);
                                    }
                                };
                                if(next!=shared_->radii.end())consider(*next);
                                if(next!=shared_->radii.begin())consider(*std::prev(next));
                            }
                        }
                        if(!radii_[slot]->initialize(R,geometry_,angular_options_,uniform_source_,repair_relative_*.2,shared_?shared_->coefficients.get():nullptr,inherited)) {
                            reason=radii_[slot]->reason;return false;
                        }
                        if(shared_)shared_->radii.emplace(R,radii_[slot]);
                        ++radial_evaluations;
                    } else ++reused_radii;
                } else ++reused_radii;
                auto& radius=*radii_[slot];
                const bool angular_ok=(!Gradient && profile_error_ && !uniform_source_)?
                    radius.refine_local(angular_target,repair_relative_,repair_absolute_):radius.refine(angular_target);
                if(!radius.topology.completed || !angular_ok) {
                    reason=radius.reason;return false;
                }
                if(angular_target>=6 && (repair_absolute_>0 || repair_relative_>0)) {
                    const auto& current=radius.result();
                    double error=current.angular_error,scale=std::max(std::abs(current.f0),std::abs(current.fh));
                    if constexpr(Gradient)for(int j=0;j<8;++j) {
                        error=std::max(error,current.gradient_error[j]);
                        scale=std::max(scale,std::max(std::abs(current.df0[j]),std::abs(current.dfh[j])));
                    }
                    const double target=repair_absolute_+repair_relative_*scale;
                    if(error>target && !radius.repair(target)) {reason=radius.reason;return false;}
                }
                const auto& terms=radius.result();
                auto& sample=integrands[k-1];
                sample.f0=mapped[1]*terms.f0*scale0;sample.fh=mapped[1]*terms.fh*scaleh;
                next.f0+=weight*terms.f0;next.fh+=weighth*terms.fh;
                const double error_weight=std::max(error_uniform_weight_*std::abs(weight),uniform_source_?0:error_limb_weight_*std::abs(weighth));
                next.endpoint_error+=error_weight*terms.endpoint_error;
                inner_error+=(profile_error_?error_limb_weight_*std::abs(weighth):error_weight)*terms.angular_error;
                if constexpr(Gradient)for(int j=0;j<8;++j) {
                    sample.df0[j]=mapped[1]*terms.df0[j];sample.dfh[j]=mapped[1]*terms.dfh[j];
                    next.df0[j]+=weight*terms.df0[j];next.dfh[j]+=weight*terms.dfh[j];
                    inner_errors.gradient_error[j]+=std::abs(weight)*terms.gradient_error[j];
                }
            }
            // Both uniform and limb components need radial error estimates.
            const bool joint_profile=!Gradient && profile_error_ && !weighted_;
            const double detail0=joint_profile?0:panel_detail::interpolation_detail(rule,integrands,[](const auto& a){return a.f0;});
            const double detailh=(uniform_source_ || joint_profile)?0:panel_detail::interpolation_detail(limb_rule,integrands,[](const auto& a){return a.fh;});
            const double detail_scale=QuadratureErrorPolicy::weighted_detail_floor/QuadratureErrorPolicy::difference_safety;
            radial_error_=have_previous?std::max(
                error_uniform_weight_*std::abs(next.f0-previous.f0)+error_limb_weight_*std::abs(next.fh-previous.fh),
                detail_scale*(error_uniform_weight_*detail0+error_limb_weight_*detailh)):INFINITY;
            if(joint_profile) {
                const auto observable=[&](const auto& value){return error_uniform_weight_*value.f0+error_limb_weight_*value.fh;};
                const double detail=panel_detail::interpolation_detail(rule,integrands,observable);
                const double fine=observable(next),old=observable(previous);
                radial_error_=have_previous?std::max(std::abs(fine-old),detail_scale*detail):INFINITY;
                // Require contraction across old, embedded and fine estimates;
                // keep the shared difference safety and interpolation floor.
                // Derivative controllers retain their incumbent estimator.
                if(lev>=3 && lev<=5 && have_previous) {
                    const auto embedded=panel_detail::embedded_estimate(lev,integrands,observable);
                    const double low=std::abs(embedded.first-old),high=std::abs(fine-embedded.first);
                    if(low>64*std::numeric_limits<double>::epsilon()*std::abs(fine) && high<=low*.25)
                        radial_error_=std::max(high,detail_scale*embedded.second);
                }
            }
            inner_error_=inner_error;
            next.angular_error=radial_error_+inner_error_;
            if constexpr(Gradient)for(int j=0;j<8;++j) {
                const double detail=panel_detail::interpolation_detail(rule,integrands,[&](const auto& a){return a.df0[j];})+
                    panel_detail::interpolation_detail(rule,integrands,[&](const auto& a){return a.dfh[j];});
                next.gradient_error[j]=have_previous?std::max(
                    std::abs(next.df0[j]-previous.df0[j])+std::abs(next.dfh[j]-previous.dfh[j]),detail_scale*detail)+
                        inner_errors.gradient_error[j]:INFINITY;
            }
            previous=next;have_previous=true;
        }
        accepted_=previous;level_=target;angular_level_=angular_target;
        reason="level_completed";return true;
    }
};

// Normalize a full-support sum (or a clearly labelled partial-panel sum).
// The coefficient u is held fixed; all eight geometry derivatives include
// the source-area normalization. Errors are estimates, not enclosure bounds.
template<bool Gradient> RadiusIntegral<Gradient> normalize(
    const RadiusIntegral<Gradient>& integral,const Geometry& p,double u) {
    if(!std::isfinite(u) || u<0 || u>1)throw std::invalid_argument("invalid linear limb coefficient");
    const double denominator=std::acos(-1.)*p.rho*p.rho*(1-u/3);
    RadiusIntegral<Gradient> out{};
    out.f0=((1-u)*integral.f0+u*integral.fh)/denominator;
    // Match the incumbent nested-difference safety factor (AdaptiveConfig:
    // nested_difference_safety=2 and gradient_difference_safety=2).
    out.angular_error=QuadratureErrorPolicy::difference_safety*integral.angular_error/denominator;
    out.endpoint_error=integral.endpoint_error/denominator;
    if constexpr(Gradient)for(int j=0;j<8;++j) {
        out.df0[j]=((1-u)*integral.df0[j]+u*integral.dfh[j])/denominator;
        if(j==2)out.df0[j]-=2*out.f0/p.rho;
        out.gradient_error[j]=QuadratureErrorPolicy::difference_safety*integral.gradient_error[j]/denominator;
        if(j==2)out.gradient_error[j]+=2*out.angular_error/p.rho;
    }
    return out;
}
} // namespace lcbinint::algebraic_boundary::triple
