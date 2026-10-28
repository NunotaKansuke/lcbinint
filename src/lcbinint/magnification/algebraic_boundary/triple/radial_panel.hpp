#pragma once
#include <boost/container/flat_map.hpp>
#include "fused_radial_rule.hpp"
#include "fused_boundary_guard.hpp"
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
    explicit EmbeddedRadialRule(int level):EmbeddedRadialRule(fejer_rule(level),
        int(fejer_rule(level).x.size())/2-1,int(fejer_rule(level).x.size())/2+1) {}
    template<class Rule> EmbeddedRadialRule(const Rule& rule,int omit_low,int omit_high) {
        const int n=int(rule.x.size());missing={omit_low,omit_high};
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
template<class Rule,class Samples,class Accessor> std::pair<double,double> embedded_estimate_from(
        const Rule& rule,const EmbeddedRadialRule& embedded,const Samples& samples,Accessor value) {
    double coarse=0,detail=0;
    for(std::size_t j=0;j<embedded.slots.size();++j)coarse+=embedded.weights[j]*value(samples[embedded.slots[j]]);
    for(int k=0;k<2;++k) {
        double predicted=0;
        for(std::size_t j=0;j<embedded.slots.size();++j)predicted+=embedded.interpolation[k][j]*value(samples[embedded.slots[j]]);
        detail+=rule.w[embedded.missing[k]]*std::abs(value(samples[embedded.missing[k]])-predicted);
    }
    return {coarse,detail};
}
template<class Samples,class Accessor> std::pair<double,double> embedded_estimate(
        int level,const Samples& samples,Accessor value) {
    return embedded_estimate_from(fejer_rule(level),embedded_radial_rule(level),samples,value);
}
template<class Rule,class Samples,class Accessor> double interpolation_detail(
    const Rule& rule,const Samples& samples,Accessor value) {
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
class BoundaryControlAtlas;
#ifdef LCBININT_TRIPLE_WINDING_ARCS
namespace winding_label_detail {class Plan;}
#endif
template<bool> struct SharedCoefficientFrames {};
template<> struct SharedCoefficientFrames<true>{
#ifdef LCBININT_TRIPLE_WINDING_ARCS
std::shared_ptr<winding_label_detail::Plan> winding_labels;
#endif
#ifdef LCBININT_TRIPLE_BOUNDARY_CONTROL
std::shared_ptr<const BoundaryControlAtlas> boundary_control;
#endif
std::shared_ptr<root_pair_detail::PreparedFrames> coefficient_frames;std::shared_ptr<std::unique_ptr<root_pair_detail::RadialCache<8>>> first_order_coefficients;};
template<bool Gradient> struct SharedRadiusCache:SharedCoefficientFrames<Gradient> {
    double requested_relative_tolerance=0;
    boost::container::flat_map<double,std::shared_ptr<RadiusWorkspace<Gradient>>> radii;
    std::optional<std::array<double,8>> identity;
    AngularOptions angular;
    std::optional<bool> uniform_source;
    double sample_relative_tolerance=0;
    std::shared_ptr<AngularCoefficientCache> coefficients;

};
template<bool Gradient=false> class RadialPanel {
    Geometry geometry_;
    double motion_origin_=0;
    bool moving_=false;
    std::array<double,Gradient?8:0> motion_base_{},motion_slope_{};
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
    std::size_t geometry_promotions_=0;
public:
    static constexpr int split_level=6;
    std::string reason="not_started";
    std::size_t radial_evaluations=0,reused_radii=0;
    RadialPanel(const Geometry& p,double lo,double hi,bool left_fold=false,
                bool right_fold=false,AngularOptions options={},
                std::shared_ptr<SharedRadiusCache<Gradient>> shared=nullptr,
                double repair_absolute=0,double repair_relative=0,bool uniform_source=false,bool weighted=false,double error_profile=-1)
        :geometry_(p),map_{lo,hi,left_fold,right_fold},angular_options_(options),shared_(std::move(shared)),
         repair_absolute_(repair_absolute),repair_relative_(repair_relative),uniform_source_(uniform_source),weighted_(weighted) {

        p.validate();
        // Component errors combine with the same nonnegative coefficients
        // as flux. The global difference safety and detail floor stay intact.
        // A negative profile selects the incumbent standalone component policy.
        // Gradient frontiers supply the actual profile too: project the
        // derivative before estimating its interpolation error.
        if(error_profile>=0) {
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
    void set_radial_motion(double origin,const std::array<double,8>& base,const std::array<double,8>& slope) {
        if constexpr(Gradient){moving_=true;motion_origin_=origin;motion_base_=base;motion_slope_=slope;}
    }
    bool has_radial_motion()const{return moving_;}
    double motion_origin()const{return motion_origin_;}
    const auto& motion_base()const{return motion_base_;}
    const auto& motion_slope()const{return motion_slope_;}
    void set_certified_band(const AngularTopology& topology,double lo,double hi) {
        if(!topology.completed || !(hi>lo))return;
        band_topology_=topology;safe_lo_=lo;safe_hi_=hi;
    }
    const std::optional<AngularTopology>& band_topology() const{return band_topology_;}
    bool full_annulus() const {
        return band_topology_ && band_topology_->roots.empty() && band_topology_->arcs.size()==1 &&
            band_topology_->arcs.front().hi-band_topology_->arcs.front().lo==2*std::acos(-1.);
    }
    double safe_lo() const{return safe_lo_;}
    double safe_hi() const{return safe_hi_;}
    int level() const {return level_;}
    int angular_level() const {return angular_level_;}
    std::size_t geometry_promotions()const{return geometry_promotions_;}
    const FoldRadialMap& radial_map() const {return map_;}
    bool weighted_radial() const {return weighted_;}
    const auto& retained_radii() const {return radii_;}
    const AngularOptions& angular_options() const {return angular_options_;}
    AngularCoefficientCache* coefficient_cache() const {return shared_?shared_->coefficients.get():nullptr;}
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
        if constexpr(!Gradient)if(uniform_source_ && full_annulus()) {
            // A proved full angular circle has exactly 2*pi*R area density.
            // Integrate that polynomial directly, retaining a rounding bound.
            accepted_={};accepted_.f0=std::acos(-1.)*(map_.b-map_.a)*(map_.a+map_.b);
            radial_error_=16*std::numeric_limits<double>::epsilon()*std::abs(accepted_.f0);
            inner_error_=0;accepted_.angular_error=radial_error_;
            level_=target;angular_level_=angular_target;reason="full_annulus_exact";return true;
        }
        if constexpr(Gradient)if(uniform_source_ && full_annulus()) {
            const double pi=std::acos(-1.);accepted_={};accepted_.f0=pi*(map_.b-map_.a)*(map_.a+map_.b);
            for(int j=0;j<8;++j){const double va=motion_base_[j]+motion_slope_[j]*(map_.a-motion_origin_),vb=motion_base_[j]+motion_slope_[j]*(map_.b-motion_origin_);accepted_.df0[j]=2*pi*(map_.b*vb-map_.a*va);}
            radial_error_=16*std::numeric_limits<double>::epsilon()*std::abs(accepted_.f0);inner_error_=0;accepted_.angular_error=radial_error_;
            level_=target;angular_level_=angular_target;reason="fused_full_annulus_exact";return true;
        }
        // Rebuild the adjacent coarse estimate after an angular refinement;
        // geometry and sample caches remain intact.
        const int minimum=(weighted_ || !Gradient || moving_)?1:2;
        const int first=angular_changed?std::max(minimum,level_-1):std::max(minimum,level_+1);
        RadiusIntegral<Gradient> previous=accepted_;
        bool have_previous=!angular_changed && level_>0;
        for(int lev=first;lev<=target;++lev) {
            RadiusIntegral<Gradient> next{};
            const auto& rule=radial_rule(lev,weighted_);
            const auto& limb_rule=weighted_?linear_radial_rule(lev):rule;
            std::vector<RadiusIntegral<Gradient>> integrands(rule.x.size()),spatial;
            const fused_detail::RadialRule *h0=nullptr,*hh=nullptr;
            if constexpr(Gradient) {spatial.resize(rule.x.size());h0=&fused_detail::radial_rule(lev,weighted_?1:0);hh=&fused_detail::radial_rule(lev,weighted_?2:0);}
            double inner_error=0;
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
                if constexpr(Gradient){if(shared_){if(!shared_->coefficient_frames)shared_->coefficient_frames=std::make_shared<root_pair_detail::PreparedFrames>();radius.set_coefficient_frames(shared_->coefficient_frames);}radius.set_value_spatial_scale(map_.b-map_.a);}
                if constexpr(Gradient)if(moving_) {
                    std::array<double,8> motion;
                    for(int j=0;j<8;++j)motion[j]=motion_base_[j]+motion_slope_[j]*(R-motion_origin_);
                    radius.set_radius_motion(motion,motion_slope_);
                }
                const bool angular_ok=(profile_error_ && !uniform_source_)?
                    radius.refine_local(angular_target,repair_relative_,repair_absolute_):radius.refine(angular_target);
                if(!radius.topology.completed || !angular_ok) {
                    reason=radius.reason;return false;
                }
                if(angular_target>=6 && (repair_absolute_>0 || repair_relative_>0)) {
                    const auto& current=radius.result();
                    double error=current.angular_error,scale=std::max(std::abs(current.f0),std::abs(current.fh));

                    const double target=repair_absolute_+repair_relative_*scale;
                    if(error>target && !radius.repair(target)) {reason=radius.reason;return false;}
                }
                const auto& terms=radius.result();
                auto& sample=integrands[k-1];
                sample.f0=mapped[1]*terms.f0*scale0;sample.fh=mapped[1]*terms.fh*scaleh;
                if(!Gradient){next.f0+=weight*terms.f0;next.fh+=weighth*terms.fh;}
                const double error_weight=std::max(error_uniform_weight_*std::abs(weight),uniform_source_?0:error_limb_weight_*std::abs(weighth));
                next.endpoint_error+=error_weight*terms.endpoint_error;
                if(!Gradient)inner_error+=(profile_error_?error_limb_weight_*std::abs(weighth):error_weight)*terms.angular_error;
                if constexpr(Gradient)for(int j=0;j<8;++j) {
                    sample.df0[j]=mapped[1]*scale0*(terms.df0[j]+motion_slope_[j]*terms.f0);sample.dfh[j]=mapped[1]*scaleh*(terms.dfh[j]+motion_slope_[j]*terms.fh);

                }
                if constexpr(Gradient) {
                    double jac_prime=0;const double x=rule.x[k-1],pi=std::acos(-1.);
                    if(!weighted_){if(map_.left&&map_.right)jac_prime=pi*pi*(map_.b-map_.a)/8*std::cos(pi*(1+x)*.5);else if(map_.left)jac_prime=(map_.b-map_.a)*.5;else if(map_.right)jac_prime=-(map_.b-map_.a)*.5;}
                    const double a0=jac_prime*scale0+(weighted_?mapped[1]*scale0*x/shape:0),ah=jac_prime*scaleh+(weighted_?2*x*mapped[1]*scaleh/shape:0);
                    const double b0=mapped[1]*mapped[1]*scale0,bh=mapped[1]*mapped[1]*scaleh;
                    auto& derivative=spatial[k-1];derivative.f0=a0*terms.f0+b0*terms.radial_f0;derivative.fh=ah*terms.fh+bh*terms.radial_fh;
                    next.f0+=h0->value[k-1]*sample.f0+h0->derivative[k-1]*derivative.f0;next.fh+=hh->value[k-1]*sample.fh+hh->derivative[k-1]*derivative.fh;
                    for(int j=0;j<8;++j) {
                        derivative.df0[j]=a0*(terms.df0[j]+motion_slope_[j]*terms.f0)+b0*(terms.radial_df0[j]+motion_slope_[j]*terms.radial_f0);
                        derivative.dfh[j]=ah*(terms.dfh[j]+motion_slope_[j]*terms.fh)+bh*(terms.radial_dfh[j]+motion_slope_[j]*terms.radial_fh);
                        next.df0[j]+=h0->value[k-1]*sample.df0[j]+h0->derivative[k-1]*derivative.df0[j];next.dfh[j]+=hh->value[k-1]*sample.dfh[j]+hh->derivative[k-1]*derivative.dfh[j];
                    }
                    if(!uniform_source_) {
                        // Replace the ordinary inner estimate by the same
                        // confluent weights, including the spatial-slope error.
                        const double A=hh->value[k-1]*mapped[1]*scaleh+hh->derivative[k-1]*ah,B=hh->derivative[k-1]*bh;
                        inner_error+=error_limb_weight_*((A!=0?std::abs(A)*terms.angular_error:0)+(B!=0?std::abs(B)*terms.radial_angular_error:0));
                    }
                }
            }
            const auto confluent_detail=[&](const auto& weights,auto value) {
                if(lev<2)return double(INFINITY);
                double detail=0;
                for(std::size_t k=0;k<integrands.size();k+=2) {
                    double predicted=0;for(std::size_t j=0;j<weights.interpolation[k].size();++j)predicted+=weights.interpolation[k][j]*value(integrands[2*j+1])+weights.interpolation_derivative[k][j]*value(spatial[2*j+1]);
                    detail+=rule.w[k]*std::abs(value(integrands[k])-predicted);
                }
                return detail;
            };
            // Both uniform and limb components need radial error estimates.
            const bool joint_profile=profile_error_ && !weighted_;
            double detail0=joint_profile?0:panel_detail::interpolation_detail(rule,integrands,[](const auto& a){return a.f0;});
            if constexpr(Gradient)if(!joint_profile)detail0=confluent_detail(*h0,[](const auto& a){return a.f0;});
            double detailh=(uniform_source_ || joint_profile)?0:panel_detail::interpolation_detail(limb_rule,integrands,[](const auto& a){return a.fh;});
            if constexpr(Gradient)if(!uniform_source_ && !joint_profile)detailh=confluent_detail(*hh,[](const auto& a){return a.fh;});
            const double detail_scale=QuadratureErrorPolicy::weighted_detail_floor/QuadratureErrorPolicy::difference_safety;
            radial_error_=have_previous?std::max(
                error_uniform_weight_*std::abs(next.f0-previous.f0)+error_limb_weight_*std::abs(next.fh-previous.fh),
                detail_scale*(error_uniform_weight_*detail0+error_limb_weight_*detailh)):INFINITY;
            if(joint_profile) {
                const auto observable=[&](const auto& value){return error_uniform_weight_*value.f0+error_limb_weight_*value.fh;};
                double detail=panel_detail::interpolation_detail(rule,integrands,observable);
                if constexpr(Gradient)detail=confluent_detail(*h0,observable);
                const double fine=observable(next),old=observable(previous);
                radial_error_=have_previous?std::max(std::abs(fine-old),detail_scale*detail):INFINITY;
                // A shared-node embedded estimate can underestimate a narrow
                // fold missed by both rules. Keep the independent nested
                // difference as well as the interpolation-detail floor.

            }
            if constexpr(Gradient) {
                // Check the actual boundary equation at independent spatial
                // witnesses before accepting a low-order confluent integral.
                const int saved_level=level_;level_=lev;
                const bool resolved=fused_detail::boundary_resolved(*this,geometry_,
                    !uniform_source_,
                    repair_relative_*(uniform_source_?.005:.05),&geometry_promotions_);
                level_=saved_level;
                if(!resolved) {
                    // Phi lies in [0,1] on proved image support. The entire
                    // annulus bounds this panel's missing scalar flux, even
                    // when a relative intensity witness reaches roundoff.
                    // Keep its finite bound so a tiny unresolved panel cannot
                    // starve the rest of the adaptive frontier indefinitely.
                    const double upper=std::acos(-1.)*(map_.b-map_.a)*(map_.a+map_.b);
                    const double estimate=error_uniform_weight_*next.f0+error_limb_weight_*next.fh;
                    radial_error_=std::max(radial_error_,std::max(std::abs(estimate),std::abs(upper-estimate)));
                }
            }
            inner_error_=inner_error;
            next.angular_error=radial_error_+inner_error_;

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
    // nested_difference_safety=2).
    out.angular_error=QuadratureErrorPolicy::difference_safety*integral.angular_error/denominator;
    out.endpoint_error=integral.endpoint_error/denominator;
    if constexpr(Gradient)for(int j=0;j<8;++j) {
        out.df0[j]=((1-u)*integral.df0[j]+u*integral.dfh[j])/denominator;
        if(j==2)out.df0[j]-=2*out.f0/p.rho;

    }
    return out;
}
} // namespace lcbinint::algebraic_boundary::triple
