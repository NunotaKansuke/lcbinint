// Private adaptive kernel; included in the finite-source implementation namespace.
struct AdaptiveStageTimer {
    double* elapsed;
    std::chrono::steady_clock::time_point start;
    explicit AdaptiveStageTimer(double* field):elapsed(field) {
        if(elapsed)start=std::chrono::steady_clock::now();
    }
    ~AdaptiveStageTimer() {
        if(elapsed)*elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
};

// Solve F=0 and F_x=0 for the actual image tip. This is a boundary
// geometry solve, not an approximation to the image flux.
template<class Mapper>
std::optional<double> adaptive_cartesian_tip_phase(const Mapper& mapper,SourcePosition source,
    double rho2,double x0,double y0,double hx,double hy,double width,int direction) {
 if constexpr(!std::is_same_v<Mapper,BinaryLensMapper>)return std::nullopt;
 else {
  const double extent=std::max(4*hx,2*width);
  double x=x0,y=y0+direction*.5*hy;
  const auto evaluate=[&](double xx,double yy) {
   const Complex z(xx,yy),a=std::conj(z)-mapper.separation,b=std::conj(z);
   const Complex ia=1./a,ib=1./b;
   const Complex w=z-mapper.m1*ia-mapper.m2*ib-Complex(mapper.source_origin_shift+source.x,source.y);
   const Complex q=mapper.m1*ia*ia+mapper.m2*ib*ib,t=mapper.m1*ia*ia*ia+mapper.m2*ib*ib*ib;
   const Complex wx=1.+q,wy=Complex(0,1)*(1.-q),wxx=-2.*t,wxy=Complex(0,2)*t;
   return std::array<double,5>{std::norm(w)-rho2,2*std::real(std::conj(w)*wx),
       2*std::real(std::conj(w)*wy),2*(std::norm(wx)+std::real(std::conj(w)*wxx)),
       2*std::real(std::conj(wy)*wx+std::conj(w)*wxy)};
  };
  auto v=evaluate(x,y);
  for(int k=0;k<8;++k) {
   const double norm=std::max(std::abs(v[0]),std::abs(v[1])*hx)/rho2;
   if(!std::isfinite(norm)||!(v[3]>0))return std::nullopt;
   if(norm<1e-7) {
    const double eta=direction*(y-y0)/hy;
    if(eta>0 && eta<=1.5 && direction*v[2]>0 &&
       evaluate(.5*(x+x0),.5*(y+y0))[0]<=0)return eta;
    return std::nullopt;
   }
   const double det=v[1]*v[4]-v[2]*v[3];
   if(!std::isfinite(det)||det==0)return std::nullopt;
   const double dx=(-v[0]*v[4]+v[2]*v[1])/det,dy=(-v[1]*v[1]+v[0]*v[3])/det;
   double step=1;bool moved=false;
   for(int back=0;back<6;++back,step*=.5) {
    const double nx=x+step*dx,ny=y+step*dy,eta=direction*(ny-y0)/hy;
    if(std::abs(nx-x0)>extent || eta<-.1 || eta>1.6)continue;
    const auto trial=evaluate(nx,ny);
    if(std::max(std::abs(trial[0]),std::abs(trial[1])*hx)<norm*rho2){x=nx;y=ny;v=trial;moved=true;break;}
   }
   if(!moved)return std::nullopt;
  }
  return std::nullopt;
 }
}

struct AdaptiveBoundaryBracketDiagnostic {
    double lower=0,upper=1;
    bool rounded_zero=false;
};

// Locate a source-limb crossing within a known inside/outside image-plane
// cell.  A converged Newton step is accepted only after two residual-only
// probes explicitly bracket it to the requested width; otherwise the bounded
// bisection tail supplies the same guarantee where the derivative is locally
// ill-conditioned.
template <typename ImageMap>
double refine_adaptive_boundary_crossing_fraction(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius2,
    double inside_x,
    double outside_x,
    double y,
    double initial,
    double grid_spacing,
    AdaptiveBoundaryBracketDiagnostic* bracket_diagnostic=nullptr,
    double* limb_inverse_gradient=nullptr)
{
    const auto finish=[&](double value,double lo,double hi,bool rounded_zero=false,double derivative_hint=0.) {
        // A bracket midpoint has a systematic area bias when boundary errors
        // cancel the outer quadrature residual. Polish within the existing
        // bracket once; no tighter bisection tolerance or image solve is needed.
        if (!rounded_zero && std::isfinite(value)) {
            const auto evaluation=boundary_root_evaluation(mapper,
                inside_x+value*(outside_x-inside_x),y,source,source_radius2);
            derivative_hint=evaluation.derivative_x;
            const double derivative=evaluation.derivative_x*(outside_x-inside_x);
            if (std::isfinite(evaluation.residual)&&std::isfinite(derivative)&&std::abs(derivative)>1e-30) {
                const double candidate=value-evaluation.residual/derivative;
                if (candidate>=lo&&candidate<=hi&&std::isfinite(candidate))value=candidate;
            }
        }
        if(limb_inverse_gradient!=nullptr)*limb_inverse_gradient=std::isfinite(derivative_hint)&&derivative_hint!=0?1/std::abs(derivative_hint):0;
        if (bracket_diagnostic!=nullptr) *bracket_diagnostic={lo,hi,rounded_zero};
        return value;
    };
    double inside_fraction = 0.0;
    double outside_fraction = 1.0;
    double fraction = std::clamp(initial, 0.0, 1.0);
    // Bennett locates the boundary to 0.1 h^2 in physical coordinates.  The
    // fraction along a one-cell segment therefore needs a bracket of 0.1 h.
    // Keeping this tied to h is important: a fixed fraction leaves an O(h)
    // boundary-location error when the source grid is refined.
    const double maximum_boundary_bracket = std::clamp(
        0.1 * std::abs(grid_spacing),
        1.0e-14,
        0.25);
    constexpr int kBoundaryNewtonIterations = 3;
    bool convergence_probed = false;
    for (int iteration = 0; iteration < kBoundaryNewtonIterations; ++iteration) {
        const double x = inside_x + fraction * (outside_x - inside_x);
        const auto evaluation = boundary_root_evaluation(
            mapper, x, y, source, source_radius2);
        if (evaluation.residual == 0.0) {
            return finish(fraction,inside_fraction,outside_fraction,true,evaluation.derivative_x);
        }
        if (evaluation.residual <= 0.0) {
            inside_fraction = fraction;
        } else {
            outside_fraction = fraction;
        }
        if (outside_fraction - inside_fraction <= maximum_boundary_bracket) {
            return finish(0.5 * (inside_fraction + outside_fraction),inside_fraction,outside_fraction);
        }
        const double fraction_derivative =
            evaluation.derivative_x * (outside_x - inside_x);
        double next = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(fraction_derivative) &&
            std::abs(fraction_derivative) > 1.0e-30) {
            next = fraction - evaluation.residual / fraction_derivative;
        }
        if (!(next > inside_fraction && next < outside_fraction)) {
            next = 0.5 * (inside_fraction + outside_fraction);
        }
        if (!convergence_probed &&
            std::abs(next - fraction) <= maximum_boundary_bracket) {
            convergence_probed = true;
            // The current Newton evaluation already supplies one bracket sign.
            // Probe only the other side before paying for a two-sided check.
            {
                const bool inside = evaluation.residual <= 0.0;
                const double other = inside
                    ? std::min(outside_fraction, fraction + maximum_boundary_bracket)
                    : std::max(inside_fraction, fraction - maximum_boundary_bracket);
                const double lo = inside ? fraction : other;
                const double hi = inside ? other : fraction;
                if (next >= lo && next <= hi && hi > lo &&
                    hi - lo <= maximum_boundary_bracket) {
                    const double residual = boundary_root_residual(mapper,
                        inside_x + other * (outside_x - inside_x), y, source, source_radius2);
                    if (std::isfinite(residual)) {
                        if ((inside && residual > 0.0) || (!inside && residual <= 0.0))
                            return finish(0.5 * (lo + hi), lo, hi);
                        if (inside) inside_fraction = other;
                        else outside_fraction = other;
                        if (!(next > inside_fraction && next < outside_fraction))
                            next = 0.5 * (inside_fraction + outside_fraction);
                    }
                }
            }
            const double half_width = 0.5 * maximum_boundary_bracket;
            const double local_inside = std::max(inside_fraction, next - half_width);
            const double local_outside = std::min(outside_fraction, next + half_width);
            if (local_outside - local_inside <= maximum_boundary_bracket) {
                const double inside_x_probe =
                    inside_x + local_inside * (outside_x - inside_x);
                const double outside_x_probe =
                    inside_x + local_outside * (outside_x - inside_x);
                const double inside_residual = boundary_root_residual(
                    mapper, inside_x_probe, y, source, source_radius2);
                const double outside_residual = boundary_root_residual(
                    mapper, outside_x_probe, y, source, source_radius2);
                if (inside_residual <= 0.0 && outside_residual > 0.0) {
                    return finish(0.5 * (local_inside + local_outside),local_inside,local_outside);
                }
                // Failed local certification still supplies evaluated signs.
                // Retain them for the safeguarded continuation rather than
                // bisecting the original, much wider interval again.
                for (const auto& probe : {std::pair<double,double>{local_inside,inside_residual},
                                          std::pair<double,double>{local_outside,outside_residual}}) {
                    if(probe.first>inside_fraction && probe.first<outside_fraction && std::isfinite(probe.second)) {
                        if(probe.second<=0)inside_fraction=probe.first;
                        else outside_fraction=probe.first;
                    }
                }
                if(!(next>inside_fraction && next<outside_fraction))
                    next=0.5*(inside_fraction+outside_fraction);
            }
        }
        fraction = next;
    }
    for (int iteration = 0;
         iteration < 64 &&
             outside_fraction - inside_fraction > maximum_boundary_bracket;
         ++iteration) {
        const double midpoint = 0.5 * (inside_fraction + outside_fraction);
        const double x = inside_x + midpoint * (outside_x - inside_x);
        const double residual = boundary_root_residual(
            mapper, x, y, source, source_radius2);
        if (residual <= 0.0) {
            inside_fraction = midpoint;
        } else {
            outside_fraction = midpoint;
        }
    }
    return finish(0.5 * (inside_fraction + outside_fraction),inside_fraction,outside_fraction);
}

struct AdaptiveEvaluatedCartesianSeed {
    SourcePosition position;
    double jacobian = 0.0;
};

struct AdaptiveRunCellState {
    double mapped_distance2 = std::numeric_limits<double>::quiet_NaN();
};

template <typename ImageMap>
adaptive_detail::CartesianRunFillResult fill_adaptive_cartesian_run_union(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<adaptive_detail::CartesianLatticeSeed>& seeds,
    double incr,
    std::int64_t maximum_evaluations,
    std::size_t expected_rows,
    adaptive_detail::CartesianRunFillTrace* trace = nullptr,
    const std::vector<adaptive_detail::CartesianLatticeSeed>* coverage_probes = nullptr,
    bool certify_coverage=false,
    const std::vector<SourcePosition>* physical_support_seeds=nullptr,
    double row_spacing=0,
    const adaptive_detail::CartesianRunFillResult* retained_rows=nullptr,int retained_row_factor=1,int retained_column_factor=1,
    const std::array<std::int64_t,2>* row_clip=nullptr,unsigned physical_tip_mask=3,int cap_weight_mode=0,
    const adaptive_detail::CartesianRunFillResult* retained_fine_rows=nullptr)
{
    if(row_spacing==0)row_spacing=incr;
    if(certify_coverage && row_spacing!=incr)throw std::invalid_argument("rectangular adaptive grid does not support the square-grid coverage audit");
    auto* retained_work=adaptive_flux_request();
    const bool light_error=lightweight_flux_error_mode();
    const bool boundary_error=boundary_diagnostics_enabled();
    const double source_radius2 = source_radius * source_radius;
    const double inverse_source_radius2 = 1.0 / source_radius2;
    if (trace != nullptr) {
        trace->cell_spacing = incr;
    }
    const bool use_limb_darkening =
        settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0;
    const double edge_brightness = use_limb_darkening
        ? (finite_magnifier != nullptr && settings.limb_darkening_d != 0.0
              ? finite_magnifier->limb_darkening_table_brightness(1.0)
              : source_surface_brightness(1.0, settings))
        : 1.0;
    const bool bennett_root_enabled =
        std::getenv("LCBININT_DIAGNOSTIC_DISABLE_BENNETT_LIMB") == nullptr;
    const auto brightness_at_distance2 = [&](double distance2) {
        if (!use_limb_darkening) {
            return 1.0;
        }
        const double normalized2 = std::clamp(
            distance2 * inverse_source_radius2, 0.0, 1.0);
        if (settings.limb_darkening_d == 0.0) {
            const double mu=std::sqrt(std::max(0.0,1.0-normalized2));
            return 1.0-settings.limb_darkening_c*(1.0-mu);
        }
        return finite_magnifier != nullptr
            ? finite_magnifier->limb_darkening_table_brightness(normalized2)
            : source_surface_brightness(normalized2, settings);
    };
    // A run contains at least one evaluated inside cell, so the evaluation
    // budget is also a strict upper bound on useful run storage.  Retain the
    // independent ceiling as the memory safety envelope for very large maps.
    constexpr std::size_t kRunMemorySafetyCeiling = 1U << 21;
    const std::size_t evaluation_run_ceiling = maximum_evaluations > 0
        ? static_cast<std::size_t>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(maximum_evaluations),
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())))
        : 0;

    std::optional<adaptive_detail::CartesianCoverageDomain> coverage_domain;
    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {

    }
    const auto run_fill = [&](auto& trace_sink) {
        const auto evaluate_cell = [&](std::int64_t ix, std::int64_t iy) {
            if(row_clip && (iy<(*row_clip)[0] || iy>(*row_clip)[1]))return AdaptiveRunCellState{std::numeric_limits<double>::infinity()};
            const double x = static_cast<double>(ix) * incr;
            const double y = static_cast<double>(iy) * row_spacing;
            if(auto* work=retained_work) {
                if (work->depth > 0) work->mapped.materialize();
                const std::array<double,2> key{x,y};const auto [saved,inserted]=work->mapped.access(key,work->new_maps<work->budget);
                if(!saved)throw AdaptiveFluxBudgetExhausted{};
                if(!inserted){++work->reused_maps;return AdaptiveRunCellState{*saved};}
                *saved=mapped_lens_distance2(mapper,x,y,source);++work->new_maps;return AdaptiveRunCellState{*saved};
            }
            return AdaptiveRunCellState {
                mapped_lens_distance2(mapper, x, y, source),
            };
        };
        const auto inside_cell = [&](const AdaptiveRunCellState& state) {
            return state.mapped_distance2 <= source_radius2;
        };
        const auto weight_cell = [&](std::int64_t, std::int64_t, const AdaptiveRunCellState& state) {
            if (!use_limb_darkening) {
                return 1.0;
            }
            const double normalized2 =
                state.mapped_distance2 * inverse_source_radius2;
            if (settings.limb_darkening_d == 0.0) {
                // Only accepted disk cells reach this hot path: normalized2
                // is nonnegative. Avoid a generic profile call and its
                // redundant upper clamp / two-coefficient dispatch.
                const double depth = std::max(0.0, 1.0 - normalized2);
                return 1.0 - settings.limb_darkening_c * (1.0 - std::sqrt(depth));
            }
            return finite_magnifier != nullptr
                ? finite_magnifier->limb_darkening_table_brightness(normalized2)
                : source_surface_brightness(normalized2, settings);
        };
        const auto diagnostic_boundary = [&](const adaptive_detail::CartesianRun& run,
            const AdaptiveRunCellState& left_inside,
            const AdaptiveRunCellState& left_outside,
            const AdaptiveRunCellState& right_inside,
            const AdaptiveRunCellState& right_outside,
            const std::optional<AdaptiveRunCellState>& left_second,
            const std::optional<AdaptiveRunCellState>& right_second) {
#ifdef LCBININT_TRACE_ROW_BOUNDARY
            AdaptiveStageTimer boundary_timer(retained_work && retained_work->cost_trace?&retained_work->row_boundary_seconds:nullptr);
#endif

            adaptive_detail::CartesianBoundaryContribution contribution;
            contribution.error_recorded=light_error;
            const bool estimate_edge=boundary_error&&(!light_error||use_limb_darkening);
            if (boundary_error && !light_error) contribution.diagnostic.emplace();
            const auto edge = [&](std::int64_t inside_ix,
                                  const AdaptiveRunCellState& inside,
                                  std::int64_t outside_ix,
                                  const AdaptiveRunCellState& outside,
                                  const std::optional<AdaptiveRunCellState>& second) {
                const double inside_x = static_cast<double>(inside_ix) * incr;
                const double outside_x = static_cast<double>(outside_ix) * incr;
                const double y = static_cast<double>(run.iy) * row_spacing;
                // The raw mapper gives NaN at an exact pole (0*infinity).
                // Only that known geometric pole is an infinite outside witness.
                const bool outside_is_pole = [&]() {
                    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
                        return y == 0.0 && (outside_x == 0.0 || outside_x == mapper.separation.real());
                    } else {
                        for (std::size_t j=0;j<mapper.mass.size();++j)
                            if (mapper.mass[j]>0.0 && outside_x==mapper.lens_x[j] && y==mapper.lens_y[j]) return true;
                        return false;
                    }
                }();
                const double outside_distance2 = outside_is_pole
                    ? std::numeric_limits<double>::infinity() : outside.mapped_distance2;
                // A lens pole has +infinite mapped distance and is a valid
                // outside witness. Keep refining the finite interior segment.
                if (!std::isfinite(inside.mapped_distance2) ||
                    !(outside_distance2 > source_radius2) ||
                    !(inside.mapped_distance2 <= source_radius2) ||
                    !(outside_distance2 > source_radius2)) {
                    return 0.0;
                }
                const double inside_radius =
                    std::sqrt(std::max(inside.mapped_distance2, 0.0));
                const double outside_radius =
                    std::sqrt(std::max(outside_distance2, 0.0));
                const double mapped_step = outside_radius - inside_radius;
                const double initial = std::isfinite(mapped_step) && mapped_step > 0.0
                    ? std::clamp(
                        (source_radius - inside_radius) / mapped_step,
                        0.0, 1.0)
                    : 0.5;
                AdaptiveBoundaryBracketDiagnostic bracket;
                double limb_inverse_gradient=0;
                const double fraction = bennett_root_enabled
                    ? refine_adaptive_boundary_crossing_fraction(
                        mapper, source, source_radius2,
                        inside_x, outside_x, y, initial, incr,
                        contribution.diagnostic ? &bracket : nullptr,
                        contribution.error_recorded&&use_limb_darkening&&settings.limb_darkening_d!=0?&limb_inverse_gradient:nullptr)
                    : initial;
                contribution.error_limb_density+=incr*limb_inverse_gradient/kPi;
                if (contribution.diagnostic) {
                    if (outside_ix<inside_ix) {
                        contribution.diagnostic->left_fraction=fraction;
                        contribution.diagnostic->left_bracket_lower=bracket.lower;
                        contribution.diagnostic->left_bracket_upper=bracket.upper;
                    } else {
                        contribution.diagnostic->right_fraction=fraction;
                        contribution.diagnostic->right_bracket_lower=bracket.lower;
                        contribution.diagnostic->right_bracket_upper=bracket.upper;
                    }
                    contribution.diagnostic->rounded_zero_edges+=bracket.rounded_zero;
                }
                if(contribution.error_recorded)contribution.error_fractions[outside_ix<inside_ix?0:1]=fraction;
                ++contribution.edges;
                const double node_value = brightness_at_distance2(inside.mapped_distance2);
                const auto boundary_weights=adaptive_detail::bennett_boundary_weights(fraction,settings.bennett_delta_c);
                if (estimate_edge) {
                    auto& counters = probe_counters();
                    if (second && bennett_root_enabled &&
                        settings.limb_darkening_d == 0.0) {
                        const double second_dz2 = second->mapped_distance2;
                        ++counters.boundary_model_reused_evaluations;
                        auto model = second_dz2 <= source_radius2
                            ? (light_error
                                ? adaptive_detail::bennett_linear_ld_endpoint_flux_residual(fraction,
                                    1-std::clamp(inside.mapped_distance2*inverse_source_radius2,0.,1.),
                                    1-std::clamp(second_dz2*inverse_source_radius2,0.,1.),
                                    settings.limb_darkening_c,boundary_weights)
                                : adaptive_detail::bennett_boundary_model_residual(fraction,node_value,
                                    brightness_at_distance2(second_dz2),edge_brightness,settings.bennett_delta_c))
                            : adaptive_detail::BennettBoundaryModelResidual {};
                        if(model.resolved && retained_work && retained_work->defect_correction) {
                            // Held-out third interior value validates the two-node
                            // Puiseux fit on this same lattice. Its interpolating
                            // t^(5/2) defect supplies model uncertainty, not a
                            // convergence comparison between different grids.
                            if(run.hi-run.lo<2)model.model_remainder+=std::abs(model.signed_residual);
                            else {
                                const auto third=evaluate_cell(inside_ix+(outside_ix<inside_ix?2:-2),run.iy);
                                if(!(std::isfinite(third.mapped_distance2)&&third.mapped_distance2<=source_radius2))model.model_remainder+=std::abs(model.signed_residual);
                                else {
                                    const double t=fraction+2;
                                    const double actual=settings.limb_darkening_c*std::sqrt(std::max(0.,1-third.mapped_distance2*inverse_source_radius2)/t);
                                    const double quadratic=(actual-model.sqrt_coefficient-model.three_half_coefficient*t)/2;
                                    const auto moments=adaptive_detail::power_endpoint_quadrature_defects(fraction,boundary_weights.node);
                                    const double held_defect=quadratic*(moments[2]-(2*fraction+1)*moments[1]+fraction*(fraction+1)*moments[0]);
                                    model.signed_residual+=held_defect;
                                    model.model_remainder+=std::abs(held_defect);
                                }
                            }
                        }
                        if (model.resolved) {
                            ++counters.boundary_model_edges;
                            const double flux_residual = model.signed_residual*incr*row_spacing;
                            counters.boundary_model_signed_flux_residual += flux_residual;
                            counters.boundary_model_absolute_flux_residual += std::abs(flux_residual);
                            contribution.error_signed+=flux_residual;
                            contribution.error_absolute+=std::abs(flux_residual);
                            contribution.error_remainder+=model.model_remainder*incr*row_spacing;
                            if(contribution.diagnostic) {
                                contribution.diagnostic->signed_model_flux_residual+=flux_residual;
                                contribution.diagnostic->absolute_model_flux_residual+=std::abs(flux_residual);
                            }
                        } else {
                            ++counters.boundary_model_unresolved;
                            ++contribution.error_unresolved;
                            if(contribution.diagnostic) ++contribution.diagnostic->unresolved_edges;
                        }
                    } else {
                        ++counters.boundary_model_unresolved;
                        ++contribution.error_unresolved;
                        if(contribution.diagnostic) ++contribution.diagnostic->unresolved_edges;
                    }
                }
                return boundary_weights.limb*edge_brightness+(boundary_weights.node-1)*node_value;
            };
            if (run.lo == std::numeric_limits<std::int64_t>::min() ||
                run.hi == std::numeric_limits<std::int64_t>::max()) {
                contribution.valid = false;
                return contribution;
            }
            contribution.area += edge(
                run.lo, left_inside, run.lo - 1, left_outside,left_second);
            contribution.area += edge(
                run.hi, right_inside, run.hi + 1, right_outside,right_second);
            if(retained_work && retained_work->defect_correction && use_limb_darkening && settings.limb_darkening_d==0 && run.lo==run.hi && contribution.error_recorded) {
                // No second sample exists at either endpoint of a one-node run.
                // Fit finite parabolic depth between the two actual roots.
                const double dl=contribution.error_fractions[0],dr=contribution.error_fractions[1],width=dl+dr;
                const double raw=weight_cell(run.lo,run.iy,left_inside)+contribution.area;
                double correction=0,remainder=std::max(std::abs(raw),std::abs(width-raw));
                if constexpr(std::is_same_v<ImageMap,BinaryLensMapper>) {
                    if(dl*dr>64*std::numeric_limits<double>::epsilon()) {
                        const double x=run.lo*incr,y=run.iy*row_spacing;
                        const double depth=std::max(0.,1-left_inside.mapped_distance2*inverse_source_radius2);
                        double f1=1,f2=0,f3=0;
                        const auto pole=[&](double pos,double mass) {
                            const double dx=std::max({x-dl*incr-pos,pos-x-dr*incr,0.});
                            const double radius2=dx*dx+y*y;
                            f1+=mass/radius2;f2+=2*mass/(radius2*std::sqrt(radius2));f3+=6*mass/(radius2*radius2);
                        };
                        pole(mapper.separation.real(),mapper.m1);pole(0,mapper.m2);
                        const double extent=std::max(dl,dr);
                        const double f0=std::sqrt(std::max(0.,left_inside.mapped_distance2))+f1*incr*extent;
                        const double depth3=2*(f0*f3+3*f1*f2)*inverse_source_radius2*incr*incr*incr;
                        // D-P <= sup|D'''| |(t+dl)t(t-dr)|/6. The leading
                        // parabola exceeding this envelope also proves D>0.
                        const auto model=adaptive_detail::finite_one_node_ld_model(dl,dr,depth,settings.limb_darkening_c,depth3);
                        if(model.resolved) {
                            correction=model.integral-raw;
                            remainder=model.uncertainty+
                                128*std::numeric_limits<double>::epsilon()*(1+std::abs(model.integral)+std::abs(raw));
                        }
                    }
                }
                auto& counters=probe_counters();counters.boundary_model_unresolved-=contribution.error_unresolved;
                counters.boundary_model_signed_flux_residual-=contribution.error_signed;
                counters.boundary_model_absolute_flux_residual-=contribution.error_absolute;
                contribution.error_unresolved=0;contribution.error_signed=correction*incr*row_spacing;
                contribution.error_absolute=std::abs(contribution.error_signed);contribution.error_remainder=remainder*incr*row_spacing;
                counters.boundary_model_signed_flux_residual+=contribution.error_signed;
                counters.boundary_model_absolute_flux_residual+=contribution.error_absolute;
            }
            return contribution;
        };
        // Diagnostic exact interior grouping. A failed interval enclosure simply
        // retains the original per-cell scan. No sampled endpoint can certify a block.
        std::function<std::optional<std::array<double,2>>(std::int64_t,std::int64_t,std::int64_t)> interior_block;
        if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {

        }
        std::unordered_map<std::int64_t,std::vector<std::size_t>> retained_index;
        std::function<std::optional<adaptive_detail::CartesianRetainedRun>(std::int64_t,std::int64_t)> retained_row;
        if(retained_rows && retained_work && retained_rows->run_flux_errors.size()==retained_rows->runs.size()) {
            for(std::size_t j=0;j<retained_rows->runs.size();++j)retained_index[retained_rows->runs[j].run.iy].push_back(j);
            std::int64_t retained_min_y=std::numeric_limits<std::int64_t>::max(),retained_max_y=std::numeric_limits<std::int64_t>::min();
            for(const auto& entry:retained_rows->runs){retained_min_y=std::min(retained_min_y,entry.run.iy);retained_max_y=std::max(retained_max_y,entry.run.iy);}
            const auto boundary_strip_row=[&](std::size_t j,std::int64_t ix,std::int64_t iy)->std::optional<adaptive_detail::CartesianRetainedRun> {
                if(settings.limb_darkening_d!=0 || retained_rows->run_flux_errors[j][2]!=0)return std::nullopt;
                const auto& old=retained_rows->runs[j].run;
                const auto& fractions=retained_rows->run_error_fractions[j];
                const double coarse_h=incr*retained_column_factor;
                const double left=(old.lo-fractions[0])*retained_column_factor,right=(old.hi+fractions[1])*retained_column_factor;
                const auto lo=static_cast<std::int64_t>(std::ceil(left)),hi=static_cast<std::int64_t>(std::floor(right));
                if(lo>ix || hi<ix || lo>hi)return std::nullopt;
                const auto ls=evaluate_cell(lo,iy),rs=evaluate_cell(hi,iy),outside_l=evaluate_cell(lo-1,iy),outside_r=evaluate_cell(hi+1,iy);
                if(!inside_cell(ls)||!inside_cell(rs)||inside_cell(outside_l)||inside_cell(outside_r))return std::nullopt;
                adaptive_detail::RetainedRowCore core;
                const auto value=[&](std::int64_t x)->std::optional<double> {
                    const auto state=evaluate_cell(x,iy);if(!inside_cell(state))return std::nullopt;
                    const double v=weight_cell(x,iy,state);if(!std::isfinite(v))return std::nullopt;return v;
                };
                if(retained_rows->run_cores.size()==retained_rows->runs.size() && retained_rows->run_cores[j]) {
                    core=*retained_rows->run_cores[j];
                    if(core.spacing<=0)return std::nullopt; // Interpolated row has no exact coarse core.
                } else {
                    constexpr int strip=4;
                    if(old.hi-old.lo<2*strip+8)return std::nullopt;
                    // Setup reads the coarse core once, two seven-point
                    // derivative stencils and both coarse/fine edge strips.
                    // If this requests at least as many samples as a complete
                    // fine row, retain the ordinary scan instead.
                    const auto span=old.hi-old.lo;
                    const auto setup_samples=(span-2*strip+1)+14+2*(strip+1)+2*(strip*retained_column_factor+1);
                    if(std::getenv("LCBININT_DIAGNOSTIC_STRIP_SETUP_GATE") && setup_samples>=span*retained_column_factor+1)return std::nullopt;
                    core.left=(old.lo+strip)*coarse_h;core.right=(old.hi-strip)*coarse_h;core.spacing=coarse_h;
                    // Both endpoints of every coarse core interval map inside
                    // the convex source disk. The linear-interpolation defect
                    // is at most h^2/8 sup|lens_map''|. Bound the derivative
                    // over the complete core once; subsequent refinements
                    // retain this same checked core and incur no new scan.
                    ++retained_work->strip_support_checks;
                    double map_defect=std::numeric_limits<double>::infinity();
                    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
                        const double y=iy*row_spacing;
                        const auto pole_bound=[&](double lens,double mass) {
                            const double dx=lens<core.left?core.left-lens:(lens>core.right?lens-core.right:0);
                            const double d2=dx*dx+y*y;return mass/(d2*std::sqrt(d2));
                        };
                        map_defect=coarse_h*coarse_h/4*(pole_bound(mapper.separation.real(),mapper.m1)+pole_bound(0,mapper.m2));
                    }
                    map_defect*=1+128*std::numeric_limits<double>::epsilon();
                    const double margin=2*source_radius*map_defect+map_defect*map_defect+
                        128*std::numeric_limits<double>::epsilon()*(1+source_radius2);
                    if(!std::isfinite(margin) || margin>=source_radius2){++retained_work->strip_support_rejects;return std::nullopt;}
                    for(auto x=old.lo+strip;x<=old.hi-strip;++x) {
                        if(evaluate_cell(x*retained_column_factor,iy).mapped_distance2>=source_radius2-margin) {
                            ++retained_work->strip_support_rejects;return std::nullopt;
                        }
                    }
                    std::array<double,5> derivatives[2];double removed=0;
                    for(int side=0;side<2;++side) {
                        const auto anchor=side?old.hi-strip:old.lo+strip;
                        std::array<double,7> v;
                        for(int k=0;k<7;++k){const auto g=value((anchor+k-3)*retained_column_factor);if(!g)return std::nullopt;v[k]=*g;}
                        derivatives[side]=adaptive_detail::retained_core_derivatives(v,coarse_h);
                        double trap=0;
                        for(int k=0;k<=strip;++k) {
                            const auto x=(side?old.hi-k:old.lo+k)*retained_column_factor;
                            const auto g=value(x);if(!g)return std::nullopt;
                            trap+=(k==0||k==strip?.5:1)*(*g);
                        }
                        const auto first=value((side?old.hi:old.lo)*retained_column_factor);if(!first)return std::nullopt;
                        const auto w=adaptive_detail::bennett_boundary_weights(fractions[side],settings.bennett_delta_c);
                        removed+=coarse_h*(trap+(w.node-.5)*(*first)+w.limb*edge_brightness);
                    }
                    core.trapezoid_area=retained_rows->run_areas[j]*coarse_h-removed;
                    core.derivative1=derivatives[1][0]-derivatives[0][0];
                    core.derivative3=derivatives[1][1]-derivatives[0][1];
                    const double h2=coarse_h*coarse_h;
                    core.uncertainty=h2*(derivatives[0][2]+derivatives[1][2])/12+
                        h2*h2*(derivatives[0][3]+derivatives[1][3])/720+
                        h2*h2*h2*std::abs(derivatives[1][4]-derivatives[0][4])/30240;
                }
                if(!std::isfinite(core.uncertainty) || core.uncertainty*row_spacing>retained_work->target/(4*retained_rows->runs.size()))return std::nullopt;
                const auto a=static_cast<std::int64_t>(std::llround(core.left/incr)),b=static_cast<std::int64_t>(std::llround(core.right/incr));
                if(a<lo+3 || b>hi-3 || a>=b)return std::nullopt;
                double strip_sum=0;
                for(int side=0;side<2;++side) {
                    const auto begin=side?b:lo,end=side?hi:a;
                    for(auto x=begin;x<=end;++x){const auto g=value(x);if(!g)return std::nullopt;strip_sum+=(x==begin||x==end?.5:1)*(*g);}
                }
                const auto second_l=evaluate_cell(lo+1,iy),second_r=evaluate_cell(hi-1,iy);
                const auto boundary=diagnostic_boundary({iy,lo,hi},ls,outside_l,rs,outside_r,
                    std::optional<AdaptiveRunCellState>(second_l),std::optional<AdaptiveRunCellState>(second_r));
                if(!boundary.valid || boundary.error_unresolved)return std::nullopt;
                const auto modeled=adaptive_detail::retained_core_fine_equivalent(core,incr);
                adaptive_detail::CartesianRetainedRun saved;saved.run={iy,lo,hi};saved.core=core;
                saved.area=strip_sum+.5*(weight_cell(lo,iy,ls)+weight_cell(hi,iy,rs))+boundary.area+modeled[0]/incr;
                saved.errors={boundary.error_signed,boundary.error_absolute,double(boundary.error_unresolved),
                    boundary.error_remainder+modeled[1]*row_spacing,0};saved.fractions=boundary.error_fractions;
                if(!(std::isfinite(saved.area)&&saved.area>0))return std::nullopt;
                if(std::getenv("LCBININT_DIAGNOSTIC_AUDIT_BOUNDARY_STRIPS")) {
                    double exact=boundary.area;bool gap=false;
                    for(auto x=lo;x<=hi;++x){const auto state=evaluate_cell(x,iy);if(!inside_cell(state)){gap=true;continue;}exact+=weight_cell(x,iy,state);}
                    const double difference=std::abs(saved.area-exact)*incr*row_spacing;
                    ++retained_work->strip_audit_rows;retained_work->strip_audit_gap_rows+=gap;
                    retained_work->strip_audit_absolute_difference+=difference;
                    retained_work->strip_audit_max_difference=std::max(retained_work->strip_audit_max_difference,difference);
                }
                ++retained_work->retained_boundary_strip_rows;return saved;
            };
            retained_row=[&,retained_min_y,retained_max_y,boundary_strip_row](std::int64_t ix,std::int64_t iy)->std::optional<adaptive_detail::CartesianRetainedRun>{
                if(iy%retained_row_factor) {
                    if(retained_row_factor!=2 || (retained_column_factor!=1 &&
                        !(retained_work->initial_bins==8 && retained_column_factor==2)) || settings.limb_darkening_d!=0 ||
                        !retained_work->local_row_interpolation)return std::nullopt;
                    ++retained_work->row_interpolation_attempts;
                    const auto base=(iy-1)/2;
                    const auto reject=[&](int reason)->std::optional<adaptive_detail::CartesianRetainedRun>{++retained_work->row_interpolation_outcomes[reason];return std::nullopt;};
                    std::array<std::size_t,6> indices;
                    std::array<double,6> areas,left,right;
                    for(int k=0;k<6;++k) {
                        const auto found=retained_index.find(base-2+k);
                        // Tip, gap and junction rows keep ordinary exact scans.
                        if(found==retained_index.end() || found->second.size()!=1)return reject(0);
                        const auto j=found->second.front();indices[k]=j;
                        const auto& r=retained_rows->runs[j].run;
                        if(retained_rows->run_flux_errors[j][2]!=0)return reject(1);
                        if(k>0) {
                            const auto& prev=retained_rows->runs[indices[k-1]].run;
                            if(r.lo>prev.hi || prev.lo>r.hi)return reject(1);
                        }
                        // Convert old row values to the new x-cell units before
                        // interpolating the smooth moment in physical y.
                        areas[k]=retained_rows->run_areas[j]*retained_column_factor;
                        left[k]=(r.lo-retained_rows->run_error_fractions[j][0])*retained_column_factor;
                        right[k]=(r.hi+retained_rows->run_error_fractions[j][1])*retained_column_factor;
                        if(use_limb_darkening) {
                            // Interpolate the smooth LD moment, with the old
                            // Bennett grid-phase defect removed. The constant
                            // limb brightness times geometric width is exact
                            // once the new row's endpoints have been evaluated.
                            areas[k]+=retained_rows->run_flux_errors[j][0]/(incr*row_spacing*retained_row_factor);
                            areas[k]-=edge_brightness*(right[k]-left[k]);
                        }
                    }
                    const auto area=adaptive_detail::retained_row_midpoint(areas);
                    const auto l=adaptive_detail::retained_row_midpoint(left),r=adaptive_detail::retained_row_midpoint(right);
                    if(!(std::isfinite(area[0]) && area[0]>=0 && std::isfinite(l[0]) && std::isfinite(r[0]) && r[0]>l[0]))return reject(2);
                    const double defect=std::max(area[1],(1-settings.limb_darkening_c)*(l[1]+r[1]))*incr*row_spacing;
                    // Reserve only a fraction of the global error budget for
                    // omitted scans; carried errors remain in the same sum.
                    if(!std::isfinite(defect) || defect>retained_work->target/(4*retained_rows->runs.size()))return reject(3);
                    const auto lo=static_cast<std::int64_t>(std::ceil(l[0]));
                    const auto hi=static_cast<std::int64_t>(std::floor(r[0]));
                    if(lo>ix || hi<ix || lo>hi)return reject(4);
                    // Actual endpoint brackets must agree with the predicted
                    // run. Four cheap classifications replace a whole scan.
                    if(!inside_cell(evaluate_cell(lo,iy)) || !inside_cell(evaluate_cell(hi,iy)) ||
                        inside_cell(evaluate_cell(lo-1,iy)) || inside_cell(evaluate_cell(hi+1,iy)))return reject(5);
                    adaptive_detail::CartesianRetainedRun saved;saved.run={iy,lo,hi};saved.area=area[0];
                    saved.fractions={lo-l[0],r[0]-hi};
                    constexpr std::array<double,6> coefficients{0,-1./16,9./16,9./16,-1./16,0};
                    for(int k=0;k<6;++k) {
                        const auto& e=retained_rows->run_flux_errors[indices[k]];
                        saved.errors[0]+=coefficients[k]*e[0]/2;
                        for(int f:{1,3})saved.errors[f]+=std::abs(coefficients[k])*e[f]/2;
                    }
                    saved.errors[3]+=defect;
                    if(use_limb_darkening || retained_column_factor>1) {
                        const auto boundary=diagnostic_boundary({iy,lo,hi},evaluate_cell(lo,iy),evaluate_cell(lo-1,iy),
                            evaluate_cell(hi,iy),evaluate_cell(hi+1,iy),
                            std::optional<AdaptiveRunCellState>(evaluate_cell(lo+1,iy)),std::optional<AdaptiveRunCellState>(evaluate_cell(hi-1,iy)));
                        if(!boundary.valid || boundary.error_unresolved)return reject(2);
                        saved.fractions=boundary.error_fractions;
                        saved.area=use_limb_darkening?area[0]+edge_brightness*(hi-lo+saved.fractions[0]+saved.fractions[1])-
                            boundary.error_signed/(incr*row_spacing):double(hi-lo+1)+boundary.area;
                        saved.errors[0]=boundary.error_signed;saved.errors[1]=boundary.error_absolute;
                        saved.errors[3]+=boundary.error_remainder;
                        if(!(std::isfinite(saved.area)&&saved.area>0))return reject(2);
                    }
                    if(std::getenv("LCBININT_DIAGNOSTIC_AUDIT_INTERPOLATED_ROWS")) {
                        double exact=0;bool gap=false;
                        for(auto x=lo;x<=hi;++x){const auto state=evaluate_cell(x,iy);if(!inside_cell(state)){gap=true;continue;}exact+=weight_cell(x,iy,state);}
                        const auto boundary=diagnostic_boundary({iy,lo,hi},evaluate_cell(lo,iy),evaluate_cell(lo-1,iy),
                            evaluate_cell(hi,iy),evaluate_cell(hi+1,iy),
                            std::optional<AdaptiveRunCellState>(evaluate_cell(lo+1,iy)),std::optional<AdaptiveRunCellState>(evaluate_cell(hi-1,iy)));
                        exact+=boundary.area;
                        ++retained_work->interpolated_audit_rows;retained_work->interpolated_audit_gap_rows+=gap;
                        retained_work->interpolated_audit_max_difference=std::max(retained_work->interpolated_audit_max_difference,std::abs(saved.area-exact)*incr*row_spacing);
                    }
                    saved.core=adaptive_detail::RetainedRowCore{}; // Approximate row: do not infer an exact interior integral from it.
                    ++retained_work->interpolated_rows;retained_work->row_interpolation_outcomes[6]+=hi-lo+1;return saved;
                }
                const auto found=retained_index.find(iy/retained_row_factor);if(found==retained_index.end())return std::nullopt;
                for(auto j:found->second){const auto& r=retained_rows->runs[j].run;
                    if(retained_column_factor>1) {
                        if(found->second.size()!=1 || settings.limb_darkening_d!=0 ||
                            iy/retained_row_factor-retained_min_y<4 || retained_max_y-iy/retained_row_factor<4)return std::nullopt;
                        if(retained_work->local_boundary_strips)return boundary_strip_row(j,ix,iy);
                        if(retained_row_factor!=1)return std::nullopt;
                        const auto& e=retained_rows->run_flux_errors[j];
                        if(e[2]!=0 || std::abs(e[0])+e[3]>retained_work->target/(4*retained_rows->runs.size()))return std::nullopt;
                        const auto& fractions=retained_rows->run_error_fractions[j];
                        const double left=(r.lo-fractions[0])*retained_column_factor,right=(r.hi+fractions[1])*retained_column_factor;
                        const auto lo=static_cast<std::int64_t>(std::ceil(left)),hi=static_cast<std::int64_t>(std::floor(right));
                        if(lo>ix || hi<ix || lo>hi)continue;
                        if(!inside_cell(evaluate_cell(lo,iy)) || !inside_cell(evaluate_cell(hi,iy)) ||
                            inside_cell(evaluate_cell(lo-1,iy)) || inside_cell(evaluate_cell(hi+1,iy)))return std::nullopt;
                        adaptive_detail::CartesianRetainedRun saved;saved.run={iy,lo,hi};
                        saved.area=retained_rows->run_areas[j]*retained_column_factor;
                        saved.errors=e; // Same physical row integral and error.
                        saved.fractions={lo-left,right-hi};
                        ++retained_work->retained_horizontal_rows;return saved;
                    }
                    if(ix<r.lo || ix>r.hi)continue;
                    adaptive_detail::CartesianRetainedRun saved;saved.run={iy,r.lo,r.hi};saved.area=retained_rows->run_areas[j];saved.errors=retained_rows->run_flux_errors[j];saved.fractions=retained_rows->run_error_fractions[j];
                    if(retained_rows->run_cores.size()==retained_rows->runs.size())saved.core=retained_rows->run_cores[j];
                    for(int k:{0,1,3})saved.errors[k]/=retained_row_factor;
                    ++retained_work->reused_rows;return saved;
                }return std::nullopt;
            };
        }
        std::unordered_map<std::int64_t,std::vector<std::size_t>> retained_fine_index;
        if(retained_fine_rows && retained_work) {
            for(std::size_t j=0;j<retained_fine_rows->runs.size();++j)retained_fine_index[retained_fine_rows->runs[j].run.iy].push_back(j);
            auto coarse_callback=std::move(retained_row);
            retained_row=[&,coarse_callback](std::int64_t ix,std::int64_t iy)->std::optional<adaptive_detail::CartesianRetainedRun>{
                const auto found=retained_fine_index.find(iy);
                if(found!=retained_fine_index.end())for(auto j:found->second) {
                    const auto& r=retained_fine_rows->runs[j].run;if(ix<r.lo || ix>r.hi)continue;
                    adaptive_detail::CartesianRetainedRun saved;saved.run=r;saved.area=retained_fine_rows->run_areas[j];
                    saved.errors=retained_fine_rows->run_flux_errors[j];saved.fractions=retained_fine_rows->run_error_fractions[j];
                    if(retained_fine_rows->run_cores.size()==retained_fine_rows->runs.size())saved.core=retained_fine_rows->run_cores[j];
                    ++retained_work->reused_cap_rows;return saved;
                }
                return coarse_callback?coarse_callback(ix,iy):std::nullopt;
            };
        }
        const auto execute = [&](const auto& boundary) {
            return adaptive_detail::fill_cartesian_runs<AdaptiveRunCellState>(
                seeds,evaluate_cell,inside_cell,weight_cell,boundary,
        adaptive_detail::CartesianRunFillLimits {
            maximum_evaluations,
            std::min(kRunMemorySafetyCeiling, evaluation_run_ceiling),
            expected_rows,
        },
        trace_sink,coverage_probes,coverage_domain ? &*coverage_domain : nullptr,
        interior_block ? &interior_block : nullptr,use_limb_darkening?16:32,retained_row?&retained_row:nullptr,
        retained_work ? retained_work->local_boundary_strips :
            std::getenv("LCBININT_DIAGNOSTIC_LOCAL_BOUNDARY_STRIPS")!=nullptr);
        };
        if (boundary_error && (!light_error || use_limb_darkening)) return execute(diagnostic_boundary);
        // A five-argument callback avoids retaining/updating two additional
        // optional neighbour states at every inside cell of the normal scan.
        const auto physical_boundary = [&](const adaptive_detail::CartesianRun& run,
            const AdaptiveRunCellState& left_inside,const AdaptiveRunCellState& left_outside,
            const AdaptiveRunCellState& right_inside,const AdaptiveRunCellState& right_outside) {
            const std::optional<AdaptiveRunCellState> none;
            return diagnostic_boundary(run,left_inside,left_outside,right_inside,right_outside,none,none);
        };
        return execute(physical_boundary);
    };
    if (trace != nullptr) {
        return run_fill(*trace);
    }
    auto& no_trace = adaptive_detail::default_cartesian_run_fill_trace();
    return run_fill(no_trace);
}

struct AdaptiveCartesianRunComponent {
    long double area = 0.0L;
    std::int64_t cells = 0;
    int boundary_edges = 0;
    std::int64_t min_iy = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_iy = std::numeric_limits<std::int64_t>::min();
    int row_count = 0;
    std::size_t first_run_index = std::numeric_limits<std::size_t>::max();
    ComponentFill measurement;
    int refinement_factor = 1;
};

bool refined_run_footprint_is_private(
    const adaptive_detail::CartesianRunFillResult& refined,
    int factor,
    const adaptive_detail::CartesianRunFillResult& coarse,
    std::size_t coarse_component,
    int row_factor=0,
    const std::vector<std::size_t>* allowed_owners=nullptr,
    std::vector<std::size_t>* collided_owners=nullptr)
{
    if(collided_owners)collided_owners->clear();
    if(row_factor==0)row_factor=factor;
    const auto divide_floor = [factor](std::int64_t value) {
        const std::int64_t quotient = value / factor;
        const std::int64_t remainder = value % factor;
        return quotient - (remainder < 0 ? 1 : 0);
    };
    const auto divide_ceil = [factor](std::int64_t value) {
        const std::int64_t quotient = value / factor;
        const std::int64_t remainder = value % factor;
        return quotient + (remainder > 0 ? 1 : 0);
    };
    std::vector<const adaptive_detail::CartesianRunRecord*> coarse_order;
    const bool indexed = !refined.runs.empty() &&
        coarse.runs.size() > 1000000U / refined.runs.size();
    if (indexed) {
        coarse_order.reserve(coarse.runs.size());
        for (const auto& record : coarse.runs) {
            coarse_order.push_back(&record);
        }
        std::sort(
            coarse_order.begin(), coarse_order.end(),
            [](const auto* lhs, const auto* rhs) {
                return lhs->run.iy != rhs->run.iy
                    ? lhs->run.iy < rhs->run.iy
                    : lhs->run.lo < rhs->run.lo;
            });
    }
    const auto overlaps_foreign = [&](std::int64_t iy,
                                      std::int64_t lo,
                                      std::int64_t hi) {
        const auto overlaps = [&](const auto& record) {
            const bool foreign=record.component!=coarse_component &&
                (!allowed_owners || std::find(allowed_owners->begin(),allowed_owners->end(),record.component)==allowed_owners->end());
            const bool overlap=foreign && record.run.hi>=lo && record.run.lo<=hi;
            if(overlap && collided_owners && std::find(collided_owners->begin(),collided_owners->end(),record.component)==collided_owners->end())
                collided_owners->push_back(record.component);
            return overlap;
        };
        if (!indexed) {
            bool found=false;
            for (const auto& record : coarse.runs) {
                if (record.run.iy == iy && overlaps(record)) {
                    found=true;if(!collided_owners)return true;
                }
            }
            return found;
        }
        const auto first = std::lower_bound(
            coarse_order.begin(), coarse_order.end(), iy,
            [](const auto* record, std::int64_t row) {
                return record->run.iy < row;
            });
        bool found=false;
        for (auto item = first;
             item != coarse_order.end() && (*item)->run.iy == iy; ++item) {
            if (overlaps(**item)) {
                found=true;if(!collided_owners)return true;
            }
        }
        return found;
    };
    for (const auto& fine_record : refined.runs) {
        if (fine_record.run.iy % row_factor != 0) {
            continue;
        }
        const std::int64_t coarse_iy = fine_record.run.iy / row_factor;
        const std::int64_t coarse_lo = divide_ceil(fine_record.run.lo);
        const std::int64_t coarse_hi = divide_floor(fine_record.run.hi);
        if (coarse_lo > coarse_hi) {
            continue;
        }
        if (overlaps_foreign(coarse_iy, coarse_lo, coarse_hi) && !collided_owners) {
            return false;
        }
    }
    return !collided_owners || collided_owners->empty();
}

struct RetainedAdaptiveComponent {
    adaptive_detail::CartesianRunFillResult fill;
    double flux=0,signed_error=0,remainder=0;
    int bins=0,row_bins=0,order=2;
    int previous_axis=0;
    double work_estimate=0;
    double inner_signed=0,outer_signed=0,inner_remainder=0,outer_remainder=0,short_guard=0;
    int coarse_start_axis=0;
    std::size_t original_owner=0;
    std::vector<std::size_t> original_owners;
    bool blocked=false,missing_cap_rows=false,retired=false;
    std::array<double,6> outer_parts{}; // cap signed/rem, smooth rem, junction signed/rem, resolved
    std::array<bool,2> cap_weights{};
    int retained_cap_bins=0,retained_cap_x_bins=0;
    adaptive_detail::CartesianRunFillResult cap_rows;
};

// A shared occupied lattice point identifies a duplicate retained image,
// even when the two components currently use different dyadic spacings.
inline bool adaptive_cartesian_retained_overlap(const RetainedAdaptiveComponent& a,const RetainedAdaptiveComponent& b) {
 if(a.fill.runs.empty()||b.fill.runs.empty())return false;
 std::vector<std::size_t> order(b.fill.runs.size());std::iota(order.begin(),order.end(),0);
 std::sort(order.begin(),order.end(),[&](auto i,auto j){return b.fill.runs[i].run.iy<b.fill.runs[j].run.iy;});
 const long double coarse_x=std::min(a.bins,b.bins);
 for(const auto& record:a.fill.runs) {
  const auto& x=record.run;const long double y=static_cast<long double>(x.iy)*b.row_bins/a.row_bins;
  auto it=std::lower_bound(order.begin(),order.end(),y,[&](auto j,long double yy){return b.fill.runs[j].run.iy<yy;});
  for(;it!=order.end() && b.fill.runs[*it].run.iy==y;++it) {
   const auto& z=b.fill.runs[*it].run;
   const long double lo=std::max(x.lo*coarse_x/a.bins,z.lo*coarse_x/b.bins);
   const long double hi=std::min(x.hi*coarse_x/a.bins,z.hi*coarse_x/b.bins);
   if(std::ceil(lo)<=std::floor(hi))return true;
  }
 }
 return false;
}

template <typename ImageMap>
double fill_all_adaptive_cartesian_components_multirun(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<AdaptiveEvaluatedCartesianSeed>& evaluated_seeds,
    double incr,
    double magnification_hint,
    LegacyAreaDiagnostics* diagnostics,
    adaptive_detail::CartesianRunFillTrace* trace = nullptr,
    const std::vector<adaptive_detail::CartesianLatticeSeed>* coverage_probes = nullptr,
    bool source_has_critical=false,
    std::vector<RetainedAdaptiveComponent>* retained_output=nullptr,
    double row_spacing=0,
    const adaptive_detail::CartesianRunFillResult* retained_rows=nullptr,int retained_row_factor=1,int retained_column_factor=1,
    const std::array<std::int64_t,2>* row_clip=nullptr,unsigned physical_tip_mask=3,int cap_weight_mode=0,
    const adaptive_detail::CartesianRunFillResult* retained_fine_rows=nullptr)
{
    if(row_spacing==0)row_spacing=incr;
    auto* adaptive=adaptive_flux_request();
    auto& refinement_work=adaptive_detail::refinement_work_trace();
    if(adaptive && adaptive->depth==0)refinement_work.reset();
    if(adaptive && adaptive->depth==0)adaptive->initial_bins=settings.source_bins;
    if(adaptive && adaptive->depth==0 &&
       (settings.limb_darkening_d!=0 || settings.limb_darkening_c<0 || settings.limb_darkening_c>1))
        adaptive->defect_correction=false;
    if(adaptive && adaptive->mapped.slots.empty() && adaptive->depth==0 &&
       std::isfinite(magnification_hint) && magnification_hint>0) {
        // Point magnification is only an allocation hint, never an error or
        // coverage criterion. Keep the ordinary growth path for underestimates.
        const double expected=2*kPi*magnification_hint*settings.source_bins*settings.source_bins;
        const std::size_t ceiling=262144;
        std::size_t capacity=adaptive->mapped.initial_capacity;
        while(capacity<ceiling && capacity<expected)capacity*=2;
        adaptive->mapped.initial_capacity=capacity;
    }
    // Small budgets retain eager deduplication; their stopping point must not
    // depend on repeated evaluations in the deferred initial pass.
    if(adaptive && adaptive->depth==0 && adaptive->mapped.pending.empty() &&
       ((settings.source_bins!=16 && settings.source_bins!=8) || adaptive->budget<=adaptive->mapped.initial_capacity))
        adaptive->mapped.deferred=false;
    std::vector<RetainedAdaptiveComponent> retained;
    const double source_radius2 = source_radius * source_radius;
    std::vector<adaptive_detail::CartesianLatticeSeed> seeds;
    seeds.reserve(evaluated_seeds.size());
    for (const auto& seed : evaluated_seeds) {
        std::int64_t ix = 0;
        std::int64_t iy = 0;
        if (!rounded_lattice_index(seed.position.x / incr, ix) ||
            !rounded_lattice_index(seed.position.y / row_spacing, iy)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        seeds.push_back({ix, iy});
    }

    const double bins = static_cast<double>(std::max(settings.source_bins, 1));
    const double evaluation_limit_estimate = bins * bins * std::max(
        2000.0,
        4.0 * kPi *
            (std::isfinite(magnification_hint)
                 ? std::abs(magnification_hint)
                 : 0.0));
    const std::int64_t evaluation_limit =
        evaluation_limit_estimate <
                static_cast<double>(std::numeric_limits<std::int64_t>::max())
            ? std::max<std::int64_t>(
                100000,
                static_cast<std::int64_t>(
                    std::ceil(evaluation_limit_estimate)))
            : std::numeric_limits<std::int64_t>::max();
    if (trace != nullptr) {
        trace->fill_level = 0;
    }
    const bool fill_stats=probe_stats_enabled();
    const bool phase_trace=adaptive && adaptive->cost_trace;
    const auto fill_started=(fill_stats || phase_trace) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<SourcePosition> support_positions;
    if (std::getenv("LCBININT_SEED_FULL_COVERAGE_AUDIT")
        || std::getenv("LCBININT_CRITICAL_SUPPORT_CERTIFICATE"))
        for (const auto& seed:evaluated_seeds) support_positions.push_back(seed.position);
    auto run_fill = fill_adaptive_cartesian_run_union(
        mapper, source, source_radius, settings, finite_magnifier,
        seeds, incr, evaluation_limit,
        static_cast<std::size_t>(std::max(settings.source_bins, 1)) * 64U,
        trace,coverage_probes,(std::getenv("LCBININT_CARTESIAN_COVERAGE_CERTIFICATE")!=nullptr || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE")!=nullptr),&support_positions,row_spacing,retained_rows,retained_row_factor,retained_column_factor,row_clip,3,0,retained_fine_rows);
    const auto aggregation_started=phase_trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    if(phase_trace) {
        const double duration=std::chrono::duration<double>(aggregation_started-fill_started).count();
        adaptive->run_fill_seconds+=duration;
        adaptive->fill_queries+=run_fill.counters.mapped_cell_evaluations;
        adaptive->fill_rows+=run_fill.runs.size();
        adaptive->duplicate_seed_starts+=run_fill.counters.duplicate_seeds_avoided;
        for(auto cells:run_fill.component_cells)adaptive->fill_cells+=cells;
        if(adaptive->depth==0)adaptive->initial_fill_seconds+=duration;
    }
    if (fill_stats) {
        auto& c=probe_counters();++c.cartesian_fill_calls;
        c.cartesian_map_queries+=run_fill.counters.mapped_cell_evaluations;
        for (auto n:run_fill.component_cells) c.cartesian_fill_cells+=n;
        c.cartesian_fill_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-fill_started).count();
    }
    if (lightweight_flux_error_mode() && !fill_stats) {
        // Count existing work only. No timers, extra maps, event trace or solves.
        auto& c=probe_counters();++c.cartesian_fill_calls;
        c.cartesian_map_queries+=run_fill.counters.mapped_cell_evaluations;
    }
    if (trace && (std::getenv("LCBININT_CARTESIAN_COVERAGE_CERTIFICATE") || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE"))) {
        trace->coverage_requested=true;trace->coverage_certified=run_fill.coverage_certified;
        trace->coverage_boxes=run_fill.counters.coverage_boxes;
        trace->coverage_unresolved=run_fill.counters.coverage_unresolved;
        trace->coverage_components=run_fill.counters.coverage_components_added;
    }
    if (std::getenv("LCBININT_CARTESIAN_COVERAGE_CERTIFICATE") || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE")) {
        if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
            auto& c=probe_counters();++c.cartesian_coverage_calls;c.cartesian_coverage_certified+=run_fill.coverage_certified;
            c.cartesian_coverage_boxes+=run_fill.counters.coverage_boxes;c.cartesian_coverage_unresolved+=run_fill.counters.coverage_unresolved;
            c.cartesian_coverage_repairs+=run_fill.counters.coverage_components_added;c.cartesian_coverage_seconds+=run_fill.coverage_seconds;
        }
    }
    probe_counters().critical_grid_checks+=run_fill.counters.coverage_probe_cells;
    probe_counters().critical_grid_components+=run_fill.counters.coverage_components_added;
    if (!run_fill.ok()) {
        if (diagnostics != nullptr) {
            ++diagnostics->walk_budget_failures;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (boundary_diagnostics_enabled() && !lightweight_flux_error_mode()) {
        // Single-grid outer-coordinate shape indicator. It is not a bound:
        // sqrt endpoints, topology changes and quadrature remainder need their
        // own treatment. Reuse F(y); do not perform a second resolution fill.
        std::unordered_map<std::int64_t,double> row_areas;
        row_areas.reserve(run_fill.runs.size());
        for (std::size_t j=0;j<run_fill.runs.size();++j)
            row_areas[run_fill.runs[j].run.iy]+=run_fill.run_areas[j];
        long double curvature=0.0;
        std::unordered_set<std::int64_t> zero_halo;
        for (const auto& row : row_areas) {
            if (row.first==std::numeric_limits<std::int64_t>::min() ||
                row.first==std::numeric_limits<std::int64_t>::max()) continue;
            const auto below=row_areas.find(row.first-1),above=row_areas.find(row.first+1);
            const double low=below==row_areas.end() ? 0.0 : below->second;
            const double high=above==row_areas.end() ? 0.0 : above->second;
            curvature+=std::abs(low-2.0*row.second+high);
            if (below==row_areas.end()) zero_halo.insert(row.first-1);
            if (above==row_areas.end()) zero_halo.insert(row.first+1);
        }
        // The finite difference stencil extends one row beyond the support.
        // Omitting these zero-centred stencils misses endpoint curvature.
        for (const auto iy : zero_halo) {
            if (iy==std::numeric_limits<std::int64_t>::min() ||
                iy==std::numeric_limits<std::int64_t>::max()) continue;
            const auto below=row_areas.find(iy-1),above=row_areas.find(iy+1);
            const double low=below==row_areas.end() ? 0.0 : below->second;
            const double high=above==row_areas.end() ? 0.0 : above->second;
            curvature+=std::abs(low+high);
        }
        probe_counters().boundary_outer_curvature_flux_indicator+=
            static_cast<double>(curvature)*incr*row_spacing/24.0;
    }
    if (diagnostics != nullptr) {
        diagnostics->mapped_cell_evaluations +=
            run_fill.counters.mapped_cell_evaluations;
        diagnostics->frontier_intervals_popped +=
            run_fill.counters.frontier_intervals_popped;
        diagnostics->rows_with_multiple_runs +=
            run_fill.counters.rows_with_multiple_runs;
        diagnostics->maximum_runs_in_row = std::max(
            diagnostics->maximum_runs_in_row,
            run_fill.counters.maximum_runs_in_row);
        diagnostics->unique_seed_cells = run_fill.counters.unique_seed_cells;
        diagnostics->duplicate_seed_starts_avoided +=
            run_fill.counters.duplicate_seeds_avoided;
        diagnostics->provisional_components +=
            run_fill.counters.provisional_components;
        diagnostics->merged_components +=
            run_fill.counters.merged_components;
        for (std::size_t index = 0;
             index < run_fill.component_roots.size(); ++index) {
            if (run_fill.component_roots[index] == index) {
                diagnostics->integrated_cells +=
                    run_fill.component_cells[index];
            }
        }
    }

    constexpr std::size_t missing_component =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> component_index(
        run_fill.component_roots.size(), missing_component);
    std::vector<AdaptiveCartesianRunComponent> components;
    for (std::size_t index = 0; index < run_fill.runs.size(); ++index) {
        const auto& record = run_fill.runs[index];
        if (component_index[record.component] == missing_component) {
            component_index[record.component] = components.size();
            components.emplace_back();
        }
        auto& component = components[component_index[record.component]];
        component.area = run_fill.component_areas[record.component];
        component.cells = run_fill.component_cells[record.component];
        component.boundary_edges =
            run_fill.component_boundary_edges[record.component];
        component.min_iy = std::min(component.min_iy, record.run.iy);
        component.max_iy = std::max(component.max_iy, record.run.iy);
        component.row_count = run_fill.component_rows[record.component];
        if (component.first_run_index == missing_component) {
            component.first_run_index = index;
        } else {
            const auto& current = run_fill.runs[component.first_run_index].run;
            if (record.run.iy < current.iy ||
                (record.run.iy == current.iy && record.run.lo < current.lo)) {
                component.first_run_index = index;
            }
        }
    }

    const bool bennett_outer_complex_diagnostic =
        std::getenv("LCBININT_ENABLE_BENNETT_OUTER_SEGMENTS") != nullptr;
    const char* seed_mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
    const bool geometry_outer_guard=std::getenv("LCBININT_CARTESIAN_GEOMETRY_OUTER_GUARD")!=nullptr
        || (seed_mode && std::string(seed_mode)=="critical-seeds");
    // Adaptive defect correction integrates the raw row rule directly.
    // Legacy outer weights at internal segment ends cannot be represented
    // by one component-wide endpoint phase; avoid mixing the two rules.
    const bool bennett_outer_enabled =cap_weight_mode>=0 &&
        !(adaptive && adaptive->defect_correction) &&
        std::getenv("LCBININT_DISABLE_BENNETT_OUTER") == nullptr &&
        !(geometry_outer_guard && source_has_critical) &&
        (std::getenv("LCBININT_ENABLE_BENNETT_OUTER") != nullptr ||
         bennett_outer_complex_diagnostic || components.size() <= 2U);

    struct CartesianOuterBoundaryEstimate {
        double eta = std::numeric_limits<double>::quiet_NaN();
    };
    const auto vertical_boundary_delta = [&](std::size_t run_index,
                                             int direction,
                                             bool permit_partial_crossings)
        -> std::optional<CartesianOuterBoundaryEstimate> {
        if (run_index >= run_fill.runs.size()) {
            return std::nullopt;
        }
        const auto& edge_run = run_fill.runs[run_index].run;
        if (edge_run.lo >= edge_run.hi ||
            edge_run.hi == std::numeric_limits<std::int64_t>::max()) {
            return std::nullopt;
        }
        const double inside_y = static_cast<double>(edge_run.iy) * row_spacing;
        const double outside_y = inside_y + static_cast<double>(direction) * row_spacing;
        const std::uint64_t width_u = static_cast<std::uint64_t>(
            edge_run.hi - edge_run.lo) + 1U;
        if (width_u > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return std::nullopt;
        }

        // Dave's row walker evaluates the boundary crossing for every radial
        // sample in the terminating row and uses the largest crossing as the
        // outer edge.  Keep that same per-segment operation here rather than
        // inferring a tangent from only a few raster samples.  A clean outer
        // endpoint must bracket the limb at every sample.  A backtrack
        // endpoint is allowed to have the neighbouring image branch in some
        // samples; it still needs at least one valid limb bracket.
        double eta = 0.0;
        bool found_crossing = false;
        for (std::int64_t ix = edge_run.lo;; ++ix) {
            const double x = static_cast<double>(ix) * incr;
            const double inside_residual = boundary_root_residual(
                mapper, x, inside_y, source, source_radius2);
            const double outside_residual = boundary_root_residual(
                mapper, x, outside_y, source, source_radius2);
            if (!(std::isfinite(inside_residual) &&
                  std::isfinite(outside_residual) &&
                  inside_residual <= 0.0)) {
                if (!permit_partial_crossings) {
                    return std::nullopt;
                }
                if (ix == edge_run.hi) {
                    break;
                }
                continue;
            }
            if (!(outside_residual > 0.0)) {
                if (!permit_partial_crossings) {
                    return std::nullopt;
                }
                if (ix == edge_run.hi) {
                    break;
                }
                continue;
            }
            const double denominator = outside_residual - inside_residual;
            const double initial = denominator > 0.0
                ? std::clamp(-inside_residual / denominator, 0.0, 1.0)
                : 0.5;
            const double candidate = refine_vertical_boundary_crossing_fraction(
                mapper, source, source_radius2, x,
                inside_y, outside_y, inside_residual, outside_residual,
                initial, row_spacing);
            eta = std::max(eta, candidate);
            found_crossing = true;
            if (ix == edge_run.hi) {
                break;
            }
        }
        return found_crossing
            ? std::optional<CartesianOuterBoundaryEstimate>(
                CartesianOuterBoundaryEstimate {eta})
            : std::nullopt;
    };
    // Apply equation (15) to the one-to-one outer segments returned by the
    // check_bktrack-style walker.  A component may contain several such
    // segments after a split or a merge.  The walker never lets the rule
    // cross a junction, and the physical boundary check below additionally
    // rejects an endpoint whose neighbouring row still contains image cells.
    // Diagnostic-only weights; ordinary integration allocates nothing here.
    std::vector<double> traced_outer_weights;
    std::vector<int> traced_outer_pairs;
    int traced_pair_count=0;
    if (trace != nullptr || lightweight_flux_error_mode()) {
        traced_outer_weights.assign(run_fill.runs.size(), 1.0);
        traced_outer_pairs.assign(run_fill.runs.size(), -1);
    }
    int outer_components_considered = 0;
    int outer_segments_considered = 0;
    int outer_segments_applied = 0;
    int outer_components_rejected_span = 0;
    int outer_components_rejected_boundary = 0;
    if (bennett_outer_enabled) {
        for (std::size_t component_id = 0;
             component_id < components.size(); ++component_id) {
            ++outer_components_considered;
            auto& component = components[component_id];
            const std::int64_t span = component.max_iy >= component.min_iy
                ? component.max_iy - component.min_iy + 1
                : 0;
            // Equation (15) is useful for the long, thin image arcs for which
            // Bennett designed it.  A short blob has no asymptotic outer
            // strip, and the multi-seed/fold regime remains fail-closed.
            if ((components.size() > 2U && !bennett_outer_complex_diagnostic) ||
                (!geometry_outer_guard && run_fill.counters.provisional_components > 8) ||
                span < 4 ||
                component.first_run_index == missing_component) {
                ++outer_components_rejected_span;
                continue;
            }
            const std::size_t root_component =
                run_fill.runs[component.first_run_index].component;
            const auto walk = adaptive_detail::walk_cartesian_outer_segments(
                run_fill.runs, root_component);
            if (!walk.complete || walk.segments.empty()) {
                ++outer_components_rejected_span;
                continue;
            }
            // Provisional starts measure seed density, not image topology.
            // Use actual split/merge junctions for the experimental guard.
            // A clean-arc Eq. (15) rule must not become active merely because
            // the seed algorithm supplied fewer redundant starts.
            if (geometry_outer_guard && walk.junction_nodes!=0U) {
                ++outer_components_rejected_span;
                continue;
            }
            // The experimental multi-component switch is for the new
            // check_bktrack-style topology path, not a broad license to
            // change every clean image arc in a three/five-image event.  The
            // latter showed a small but repeatable bias in the planetary
            // control case, while a junction is the evidence that this
            // segment walker is doing work the old component rule could not.
            if (components.size() > 2U &&
                bennett_outer_complex_diagnostic &&
                walk.junction_nodes == 0U) {
                ++outer_components_rejected_span;
                continue;
            }
            if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
                walk.junction_nodes > 0U) {
                std::fprintf(
                    stderr,
                    "BENNETT_OUTER_CART_WALK component=%zu segments=%zu edges=%zu junctions=%zu complete=%d\n",
                    component_id, walk.segments.size(), walk.adjacency_edges,
                    walk.junction_nodes, walk.complete ? 1 : 0);
            }

            for (std::size_t segment_id = 0;
                 segment_id < walk.segments.size(); ++segment_id) {
                const auto& segment = walk.segments[segment_id];
                ++outer_segments_considered;
                const std::int64_t segment_span =
                    segment.run_indices.size() >= 2U
                    ? run_fill.runs[segment.run_indices.back()].run.iy -
                        run_fill.runs[segment.run_indices.front()].run.iy + 1
                    : 0;
                std::int64_t segment_cells = 0;
                for (const std::size_t run_index : segment.run_indices) {
                    const auto& run = run_fill.runs[run_index].run;
                    if (run.hi < run.lo ||
                        run.hi - run.lo ==
                            std::numeric_limits<std::int64_t>::max()) {
                        segment_cells = 0;
                        break;
                    }
                    segment_cells += run.hi - run.lo + 1;
                }
                const double segment_mean_width =
                    segment.run_indices.empty() || segment_cells <= 0
                    ? 0.0
                    : static_cast<double>(segment_cells) /
                        static_cast<double>(segment.run_indices.size());
                const double segment_aspect = segment_mean_width > 0.0
                    ? static_cast<double>(segment_span) / segment_mean_width
                    : 0.0;
                if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
                    walk.junction_nodes > 0U) {
                    std::fprintf(
                        stderr,
                        "BENNETT_OUTER_CART_SEGMENT component=%zu segment=%zu length=%zu aspect=%.6g lower=%d/%d upper=%d/%d\n",
                        component_id, segment_id, segment.run_indices.size(),
                        segment_aspect,
                        segment.lower_boundary ? 1 : 0,
                        segment.lower_junction ? 1 : 0,
                        segment.upper_boundary ? 1 : 0,
                        segment.upper_junction ? 1 : 0);
                }
                // Equation (15) needs two distinct interior samples at an
                // endpoint.  A short backtrack is retained in the midpoint
                // sum rather than being promoted to a one-point rule.
                if (segment.run_indices.size() < 4U ||
                    segment_span < 4 || (cap_weight_mode!=1 && segment_aspect < 8.0)) {
                    ++outer_components_rejected_span;
                    continue;
                }
                bool applied = false;
                std::optional<double> lower_eta;
                std::optional<double> upper_eta;
                if ((physical_tip_mask&1) && (segment.lower_boundary || segment.lower_junction)) {
                    const auto boundary = vertical_boundary_delta(
                        segment.run_indices.front(), -1,
                        segment.lower_junction);
                    if (boundary.has_value()) {
                        const auto weights =
                            adaptive_detail::bennett_outer_boundary_weights(
                                boundary->eta);
                        const std::size_t first = segment.run_indices.front();
                        const std::size_t second = segment.run_indices[1];
                        component.area +=
                            (weights.first - 1.0) * run_fill.run_areas[first] +
                            (weights.second - 1.0) * run_fill.run_areas[second];
                        if (trace != nullptr || lightweight_flux_error_mode()) {
                            traced_outer_pairs[first]=traced_pair_count;
                            traced_outer_pairs[second]=traced_pair_count++;
                            traced_outer_weights[first] += weights.first - 1.0;
                            traced_outer_weights[second] += weights.second - 1.0;
                        }
                        lower_eta = boundary->eta;
                        applied = true;
                    }
                }
                if ((physical_tip_mask&2) && (segment.upper_boundary || segment.upper_junction)) {
                    const auto boundary = vertical_boundary_delta(
                        segment.run_indices.back(), +1,
                        segment.upper_junction);
                    if (boundary.has_value()) {
                        const auto weights =
                            adaptive_detail::bennett_outer_boundary_weights(
                                boundary->eta);
                        const std::size_t last = segment.run_indices.back();
                        const std::size_t penultimate =
                            segment.run_indices[segment.run_indices.size() - 2];
                        component.area +=
                            (weights.first - 1.0) * run_fill.run_areas[last] +
                            (weights.second - 1.0) *
                                run_fill.run_areas[penultimate];
                        if (trace != nullptr || lightweight_flux_error_mode()) {
                            traced_outer_pairs[last]=traced_pair_count;
                            traced_outer_pairs[penultimate]=traced_pair_count++;
                            traced_outer_weights[last] += weights.first - 1.0;
                            traced_outer_weights[penultimate] += weights.second - 1.0;
                        }
                        upper_eta = boundary->eta;
                        applied = true;
                    }
                }
                if (!applied) {
                    ++outer_components_rejected_boundary;
                    continue;
                }
                ++outer_segments_applied;
                if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr) {
                    std::fprintf(
                        stderr,
                        "BENNETT_OUTER_CART_APPLY component=%zu segment=%zu length=%zu aspect=%.6g lower=%d/%d upper=%d/%d lower_eta=%.9g upper_eta=%.9g\n",
                        component_id, segment_id, segment.run_indices.size(),
                        segment_aspect,
                        segment.lower_boundary ? 1 : 0,
                        segment.lower_junction ? 1 : 0,
                        segment.upper_boundary ? 1 : 0,
                        segment.upper_junction ? 1 : 0,
                        lower_eta.has_value() ? *lower_eta : -1.0,
                        upper_eta.has_value() ? *upper_eta : -1.0);
                }
            }
        }
    }
    if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
        bennett_outer_enabled) {
        std::fprintf(
            stderr,
            "BENNETT_OUTER_CART components=%d segments=%d applied=%d span_reject=%d boundary_reject=%d\n",
            outer_components_considered, outer_segments_considered,
            outer_segments_applied,
            outer_components_rejected_span, outer_components_rejected_boundary);
    }

    if (trace != nullptr) {
        for (auto& event : trace->boundary_diagnostics) {
            if (event.fill_level == trace->fill_level &&
                event.run_index < traced_outer_weights.size()) {
                event.outer_weight = traced_outer_weights[event.run_index];
                event.outer_pair = traced_outer_pairs[event.run_index];
            }
        }
    }

    for (auto& component : components) {
        component.measurement.area = static_cast<double>(component.area);
        component.measurement.rows = component.row_count;
        component.measurement.boundary_rows = component.boundary_edges > 0
            ? component.row_count
            : 0;
        component.measurement.rows_span = component.max_iy >= component.min_iy
            ? static_cast<int>(std::min<long double>(
                static_cast<long double>(component.max_iy) -
                    static_cast<long double>(component.min_iy) + 1.0L,
                static_cast<long double>(1 << 24)))
            : 0;
        component.measurement.width = component.row_count == 0
            ? 0.0
            : static_cast<double>(component.cells) /
                static_cast<double>(component.row_count);
        component.refinement_factor =
            adaptive==nullptr && std::getenv("LCBININT_DIAGNOSTIC_DISABLE_COMPONENT_REFINEMENT") == nullptr
            ? component_refinement_factor(
                component.measurement, settings.source_bins)
            : 1;
    }

    // One index for this completed fill. Reused by component error sums,
    // adopted-grid bookkeeping, row aggregation and topology models.
    std::vector<std::vector<std::size_t>> component_error_runs;
    if(lightweight_flux_error_mode()) {
        component_error_runs.resize(run_fill.component_roots.size());
        for(std::size_t j=0;j<run_fill.runs.size();++j)
            component_error_runs[run_fill.component_roots[run_fill.runs[j].component]].push_back(j);
        for(auto& indices:component_error_runs)std::sort(indices.begin(),indices.end(),[&](std::size_t a,std::size_t b) {
            const auto& x=run_fill.runs[a].run;const auto& y=run_fill.runs[b].run;
            return x.iy!=y.iy?x.iy<y.iy:x.lo<y.lo;
        });
    }
    long double total_area = 0.0L;
    for (std::size_t component_id = 0;
        component_id < components.size(); ++component_id) {
        auto& component = components[component_id];
        double contribution = static_cast<double>(component.area);
        const int factor = component.refinement_factor;
        bool error_used_refined=false;
        double error_component_signed=0,error_component_remainder=0,error_component_limb_density=0;
        double error_limb_correction=0,error_limb_uncertainty=0;
        double adaptive_outer_signed=0,adaptive_outer_remainder=0;
        std::array<double,6> adaptive_outer_parts{};
        bool adaptive_missing_caps=false;
        if(lightweight_flux_error_mode() && component.first_run_index!=missing_component) {
            const auto owner=run_fill.runs[component.first_run_index].component;
            for(const auto j:component_error_runs[owner]) {
                    error_component_signed+=run_fill.run_flux_errors[j][0]*traced_outer_weights[j];
                    error_component_remainder+=run_fill.run_flux_errors[j][3]*std::abs(traced_outer_weights[j]);
                    error_component_limb_density+=run_fill.run_flux_errors[j][4];
                }
        }
        if (factor > 1 && component.first_run_index != missing_component) {
            const std::size_t coarse_component =
                run_fill.runs[component.first_run_index].component;
            const auto refined_seeds =
                adaptive_detail::lift_cartesian_component_run_seeds(
                    run_fill, coarse_component, factor);
            if (refined_seeds.has_value() && !refined_seeds->empty()) {
                FiniteSourceSettings refined_settings = settings;
                refined_settings.source_bins = settings.source_bins * factor;
                const double refined_incr = incr / static_cast<double>(factor);
                const std::int64_t refined_budget =
                    static_cast<std::int64_t>(std::ceil(
                        8.0 * component.measurement.width *
                            static_cast<double>(component.measurement.rows_span) *
                            static_cast<double>(factor) *
                            static_cast<double>(factor) +
                        4096.0));
                if (trace != nullptr) {
                    trace->fill_level = 1;
                }
                const auto& error_before=probe_counters();
                const double prior_signed=error_before.boundary_model_signed_flux_residual;
                const double prior_absolute=error_before.boundary_model_absolute_flux_residual;
                const auto prior_unresolved=error_before.boundary_model_unresolved;
                auto refined = fill_adaptive_cartesian_run_union(
                    mapper, source, source_radius, refined_settings,
                    finite_magnifier, *refined_seeds, refined_incr,
                    refined_budget,
                    static_cast<std::size_t>(std::max(
                        16,
                        2 * component.measurement.rows_span * factor + 16)),
                    trace);
                if(lightweight_flux_error_mode()) {
                    // A trial fill must not contribute until its flux is adopted.
                    auto& c=probe_counters();
                    c.boundary_model_signed_flux_residual=prior_signed;
                    c.boundary_model_absolute_flux_residual=prior_absolute;
                    c.boundary_model_unresolved=prior_unresolved;
                }
                if (refined.ok()) {
                    if (diagnostics != nullptr) {
                        diagnostics->fine_grid_cell_evaluations +=
                            refined.counters.mapped_cell_evaluations;
                    }
                    long double refined_area = 0.0L;
                    for (std::size_t index = 0;
                         index < refined.component_roots.size(); ++index) {
                        if (refined.component_roots[index] == index) {
                            refined_area += refined.component_areas[index];
                        }
                    }
                    const double factor2 =
                        static_cast<double>(factor) *
                        static_cast<double>(factor);
                    const double rescaled =
                        static_cast<double>(refined_area) / factor2;
                    if (std::isfinite(rescaled) && rescaled > 0.0 &&
                        refined_run_footprint_is_private(
                            refined, factor, run_fill, coarse_component)) {
                        contribution = rescaled;
                        if(lightweight_flux_error_mode()) {
                            auto& c=probe_counters();
                            for(const auto j:component_error_runs[coarse_component]) {
                                    const auto& e=run_fill.run_flux_errors[j];
                                    c.boundary_model_signed_flux_residual-=e[0];
                                    c.boundary_model_absolute_flux_residual-=e[1];
                                    c.boundary_model_unresolved-=static_cast<long long>(e[2]);
                                }
                            error_component_signed=0;error_component_remainder=0;error_component_limb_density=0;
                            for(const auto& e:refined.run_flux_errors) {
                                error_component_signed+=e[0];
                                error_component_remainder+=e[3];error_component_limb_density+=e[4];
                                c.boundary_model_signed_flux_residual+=e[0];
                                c.boundary_model_absolute_flux_residual+=e[1];
                                c.boundary_model_unresolved+=static_cast<long long>(e[2]);
                            }
                            const auto outer_error=adaptive_detail::same_grid_outer_error(refined,std::numeric_limits<std::size_t>::max(),nullptr,1-settings.limb_darkening_c);
                            c.boundary_outer_curvature_flux_indicator+=(std::abs(outer_error.signed_residual)+outer_error.uncertainty)*refined_incr*refined_incr;
                            c.boundary_outer_signed_flux_residual+=outer_error.signed_residual*refined_incr*refined_incr;
                            c.boundary_outer_remainder_flux_indicator+=outer_error.uncertainty*refined_incr*refined_incr;
                            error_limb_correction=outer_error.limb_density_correction;error_limb_uncertainty=outer_error.limb_density_uncertainty;
                            c.error_outer_fallbacks+=!outer_error.resolved;
                            ++c.error_refined_components;
                            error_used_refined=true;
                        }
                        if (diagnostics != nullptr) {
                            ++diagnostics->refined_components;
                            diagnostics->refinement_factor = std::max(
                                diagnostics->refinement_factor, factor);
                        }
                    }
                }
                else if (diagnostics != nullptr) {
                    ++diagnostics->walk_budget_failures;
                }
            }
        }
        if(lightweight_flux_error_mode() && !error_used_refined && component.first_run_index!=missing_component) {
            std::array<double,2> physical_phases{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};
            if(adaptive && adaptive->defect_correction) {
                const auto& indices=component_error_runs[run_fill.runs[component.first_run_index].component];
                for(int end:{0,1}) {
                    if(!(physical_tip_mask&(1U<<end)) || indices.size()<4)continue;
                    const auto j=end?indices.back():indices.front();const auto& r=run_fill.runs[j].run;
                    const auto& fractions=run_fill.run_error_fractions[j];
                    const double left=double(r.lo)-fractions[0],right=double(r.hi)+fractions[1];
                    if(const auto phase=adaptive_cartesian_tip_phase(mapper,source,source_radius2,
                        .5*(left+right)*incr,double(r.iy)*row_spacing,incr,row_spacing,(right-left)*incr,end?1:-1))physical_phases[end]=*phase;
                }
            }
            const auto outer_error=adaptive_detail::same_grid_outer_error(run_fill,
                run_fill.runs[component.first_run_index].component,&traced_outer_weights,1-settings.limb_darkening_c,
                &component_error_runs[run_fill.runs[component.first_run_index].component],settings.limb_darkening_d!=0,physical_tip_mask,row_clip==nullptr,
                adaptive && adaptive->defect_correction?1/(incr*row_spacing):0,
                adaptive && adaptive->guard_missing_caps,
                adaptive && adaptive->defect_correction && settings.limb_darkening_d==0,&physical_phases);
            auto& c=probe_counters();
            c.boundary_outer_curvature_flux_indicator+=(std::abs(outer_error.signed_residual)+outer_error.uncertainty)*incr*row_spacing;
            c.boundary_outer_signed_flux_residual+=outer_error.signed_residual*incr*row_spacing;
            c.boundary_outer_remainder_flux_indicator+=outer_error.uncertainty*incr*row_spacing;
            adaptive_missing_caps=outer_error.missing_cap_rows;
            adaptive_outer_signed=outer_error.signed_residual*incr*row_spacing;
            adaptive_outer_remainder=outer_error.uncertainty*incr*row_spacing;
            if(adaptive && (adaptive->row_panel_trace || adaptive->local_vertical_caps)) {
                const double scale=incr*row_spacing;
                adaptive_outer_parts={outer_error.cap_signed*scale,outer_error.cap_remainder*scale,outer_error.smooth_remainder*scale,
                    (outer_error.signed_residual-outer_error.cap_signed)*scale,
                    (outer_error.uncertainty-outer_error.cap_remainder-outer_error.smooth_remainder)*scale,double(outer_error.resolved)};
            }
            error_limb_correction=outer_error.limb_density_correction;error_limb_uncertainty=outer_error.limb_density_uncertainty;
            c.error_outer_fallbacks+=!outer_error.resolved;
        }
        if(lightweight_flux_error_mode()) {
            auto& c=probe_counters();
            c.boundary_component_flux_indicator+=std::abs(error_component_signed)+error_component_remainder;
            c.boundary_component_signed_flux_residual+=error_component_signed;
            c.boundary_component_remainder_flux_indicator+=error_component_remainder;
            if(settings.limb_darkening_c!=0 && settings.limb_darkening_d!=0) {
                const double scale=kPi*source_radius*source_radius*settings.limb_darkening_c*adaptive_detail::linear_ld_table_bias_moment(kLimbDarkeningTableSize);
                c.brightness_table_signed_flux_residual+=scale*std::max(0.,error_component_limb_density+error_limb_correction);
                c.brightness_table_remainder_flux_indicator+=scale*error_limb_uncertainty;
            }
        }
        if(adaptive && component.first_run_index!=missing_component) {
            RetainedAdaptiveComponent state;
            state.flux=contribution*incr*row_spacing;state.bins=settings.source_bins;state.row_bins=int(std::llround(source_radius/row_spacing));
            state.original_owner=run_fill.runs[component.first_run_index].component;state.outer_parts=adaptive_outer_parts;
            state.original_owners={state.original_owner};
            state.missing_cap_rows=adaptive_missing_caps;
            state.signed_error=error_component_signed+adaptive_outer_signed;
            state.remainder=error_component_remainder+adaptive_outer_remainder;
            state.inner_signed=error_component_signed;state.outer_signed=adaptive_outer_signed;
            state.inner_remainder=error_component_remainder;state.outer_remainder=adaptive_outer_remainder;
            // Fewer than eight retained rows cannot fit the endpoint model.
            // Use the represented component flux scale, rather than accepting
            // a tiny curvature fallback as a resolved short-component error.
            if(component.row_count<8){state.short_guard=std::max(0.,std::abs(state.flux)-state.remainder);state.remainder=std::max(state.remainder,std::abs(state.flux));}
            if(adaptive->coarse_start_guard && adaptive->initial_bins==8 && row_clip==nullptr && (state.bins==8 || state.row_bins==8)) {
                // Use the existing minimum endpoint footprint. Below it,
                // or with a witnessed continuation, resolve only the affected
                // component; this scheduling guard is not an error bound.
                // A failed outer fit can also hide a thin continuation at a
                // flood-fill endpoint. Resolve only this component/direction;
                // the existing cache and retained rows survive the trial.
                const auto continuation_witness=[&]()->bool {
                    if(!adaptive->coarse_start_probe)return true;
                    const auto& indices=component_error_runs[state.original_owner];
                    if(indices.size()<2)return true;
                    const auto midpoint=[&](std::size_t j) {
                        const auto& r=run_fill.runs[j].run;
                        const auto& f=run_fill.run_error_fractions[j];
                        return .5*(double(r.lo)-f[0]+double(r.hi)+f[1]);
                    };
                    for(int end:{0,1}) {
                        const auto first=end?indices.size()-1:0;
                        const auto j0=indices[first];const auto y0=run_fill.runs[j0].run.iy;
                        std::size_t second=first;std::size_t siblings=0;
                        while((end?second>0:second+1<indices.size()) &&
                              run_fill.runs[indices[second]].run.iy==y0) {
                            if(++siblings>2)return true;
                            if(end)--second;else ++second;
                        }
                        if(run_fill.runs[indices[second]].run.iy==y0)return true;
                        const auto j1=indices[second];const auto y1=run_fill.runs[j1].run.iy;
                        if(std::abs(static_cast<long double>(y0)-y1)!=1)return true;
                        const double x=(2*midpoint(j0)-midpoint(j1))*incr;
                        const double y=(2*double(y0)-double(y1))*row_spacing;
                        if(!std::isfinite(x)||!std::isfinite(y))return true;
                        const std::array<double,2> key{x,y};
                        const auto [saved,inserted]=adaptive->mapped.access(key,adaptive->new_maps<adaptive->budget);
                        if(!saved)return true;
                        if(inserted){*saved=mapped_lens_distance2(mapper,x,y,source);++adaptive->new_maps;++adaptive->coarse_start_probe_calls;}
                        else ++adaptive->reused_maps;
                        // An inside ray one row beyond the last represented
                        // row witnesses an unresolved continuation. An outside
                        // ray is only a scheduling filter, never a certificate.
                        if(!std::isfinite(*saved)||*saved<=source_radius2)return true;
                    }
                    return false;
                };
                const bool unresolved_outer=adaptive_outer_parts[5]==0 && continuation_witness();
                if(state.bins<16 && state.row_bins<16 && unresolved_outer)
                    state.coarse_start_axis=3;
                else if(state.row_bins<16 && (component.row_count<adaptive->coarse_start_min_rows || unresolved_outer))
                    state.coarse_start_axis=2;
                else if(state.bins<16 && unresolved_outer)state.coarse_start_axis=1;
                if(state.coarse_start_axis) {
                    ++adaptive->coarse_start_guards;
                    state.short_guard=std::max(state.short_guard,std::abs(state.flux));
                    state.remainder=std::max(state.remainder,std::abs(state.flux));
                }
            }
            state.fill.component_roots={0};
            for(auto j:component_error_runs[state.original_owner]) {
                state.work_estimate+=double(run_fill.runs[j].run.hi-run_fill.runs[j].run.lo+1)+4;
                state.fill.runs.push_back({run_fill.runs[j].run,0});
                state.fill.run_areas.push_back(run_fill.run_areas[j]);
                state.fill.run_flux_errors.push_back(run_fill.run_flux_errors[j]);
                state.fill.run_error_fractions.push_back(run_fill.run_error_fractions[j]);
                if(state.fill.runs.size()<=2 && traced_outer_weights[j]!=1)state.cap_weights[0]=true;
                if(traced_outer_weights[j]!=1)state.cap_weights[1]=true;
                if(run_fill.run_cores.size()==run_fill.runs.size())state.fill.run_cores.push_back(run_fill.run_cores[j]);
            }
            if(state.fill.runs.size()>=2){const auto& js=component_error_runs[state.original_owner];state.cap_weights[1]=traced_outer_weights[js[js.size()-1]]!=1 || traced_outer_weights[js[js.size()-2]]!=1;}
            retained.push_back(std::move(state));
        }
        total_area += static_cast<long double>(contribution);
        if (diagnostics != nullptr) {
            diagnostics->boundary_rows += component.measurement.boundary_rows;
        }
    }

    if (diagnostics != nullptr) {
        diagnostics->processed_images = static_cast<int>(components.size());
        diagnostics->interval_count = static_cast<int>(run_fill.runs.size());
    }
    if(phase_trace)adaptive->error_aggregation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-aggregation_started).count();
    if(retained_output)*retained_output=std::move(retained);
    else if(adaptive && adaptive->depth==0) {
        const auto control_started=phase_trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        const double children_seconds_before=adaptive->run_fill_seconds+adaptive->error_aggregation_seconds;
        adaptive->executed=true;adaptive->stop_reason="resolution_limit";
        // Centre-image witnesses were already solved by the seed stage.
        // A subcell image may have no inside point at the coarse lattice.
        // Find only its missing lattice witness, without another lens solve.
        for(const auto& witness:adaptive->centres) {
            bool represented=false;
            for(const auto& c:retained) {
                const double h=source_radius/c.bins;
                for(const auto& r:c.fill.runs)if(std::abs(witness[1]-r.run.iy*h)<=h &&
                    witness[0]>=(r.run.lo-1)*h && witness[0]<=(r.run.hi+1)*h){represented=true;break;}
                if(represented)break;
            }
            if(represented)continue;
            bool settled=false;
            try {
                for(int bins=settings.source_bins*2;bins<=adaptive->max_bins;bins*=2) {
                    const double h=source_radius/bins;
                    const auto ix=std::llround(witness[0]/h),iy=std::llround(witness[1]/h);
                    std::vector<AdaptiveEvaluatedCartesianSeed> local;
                    for(int dy=-1;dy<=1 && local.empty();++dy)for(int dx=-1;dx<=1;++dx) {
                        const std::array<double,2> key{(ix+dx)*h,(iy+dy)*h};double distance;
                        const auto [saved,inserted]=adaptive->mapped.access(key,adaptive->new_maps<adaptive->budget);
                        if(!saved)throw AdaptiveFluxBudgetExhausted{};
                        if(!inserted){distance=*saved;++adaptive->reused_maps;}
                        else {*saved=distance=mapped_lens_distance2(mapper,key[0],key[1],source);++adaptive->new_maps;}
                        if(distance<=source_radius2){local.push_back({{key[0],key[1]},0});break;}
                    }
                    if(local.empty())continue;
                    auto fine_settings=settings;fine_settings.source_bins=bins;
                    std::vector<RetainedAdaptiveComponent> found;const auto before=probe_counters();++adaptive->depth;
                    try {fill_all_adaptive_cartesian_components_multirun(mapper,source,source_radius,fine_settings,finite_magnifier,
                        local,h,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&found);}
                    catch(...){--adaptive->depth;probe_counters()=before;throw;}
                    --adaptive->depth;probe_counters()=before;
                    for(auto& c:found) {
                        bool duplicate=false,overlap=false;
                        for(const auto& prior:retained) {
                            if(prior.original_owner<run_fill.component_roots.size() ||
                               !adaptive_cartesian_retained_overlap(c,prior))continue;
                            overlap=true;
                            if(c.bins==prior.bins && c.row_bins==prior.row_bins && c.fill.runs.size()==prior.fill.runs.size()) {
                                duplicate=std::equal(c.fill.runs.begin(),c.fill.runs.end(),prior.fill.runs.begin(),
                                    [](const auto& a,const auto& b){return a.run.iy==b.run.iy && a.run.lo==b.run.lo && a.run.hi==b.run.hi;});
                                if(duplicate)break;
                            }
                        }
                        if(duplicate)continue;
                        if(overlap){c.short_guard=std::max(c.short_guard,std::abs(c.flux));c.remainder=std::max(c.remainder,std::abs(c.flux));}
                        c.original_owner=run_fill.component_roots.size()+retained.size();
                        c.original_owners={c.original_owner};
                        if(refined_run_footprint_is_private(c.fill,bins/settings.source_bins,run_fill,c.original_owner)) {
                            retained.push_back(std::move(c));++adaptive->small_image_repairs;
                        }
                    }
                    settled=!found.empty();break;
                }
            } catch(const AdaptiveFluxBudgetExhausted&){++adaptive->unresolved_centres;adaptive->stop_reason="evaluation_budget";break;}
            if(!settled)++adaptive->unresolved_centres;
        }
        const auto correction_for=[&](const RetainedAdaptiveComponent& c) {
            if(c.retired)return 0.;
            // Critical/junction models remain scheduling estimates. Apply
            // only regular endpoint defects, retaining the junction defect
            // in admission until its local refinement resolves it.
            const double correction=adaptive->defect_correction?c.signed_error-c.outer_parts[3]:0;
            const double flux=c.flux+correction;
            return std::isfinite(flux)&&flux>0?correction:0.;
        };
        const auto totals=[&]() {
            long double flux=0,signed_sum=0,remainder_sum=0,absolute_sum=0;
            long double applied=0;
            for(const auto& c:retained){const auto correction=correction_for(c);flux+=c.flux+correction;applied+=correction;signed_sum+=c.signed_error-correction;remainder_sum+=c.remainder;absolute_sum+=std::abs(c.signed_error-correction)+c.remainder;}
            adaptive->applied_correction=double(applied);
            adaptive->signed_error=double(signed_sum);adaptive->remainder=double(remainder_sum);
            adaptive->acceptance_indicator=double(absolute_sum);
            return std::array<double,2>{double(flux),double(absolute_sum)};
        };
        const auto record_cost=[&](double flux) {
            if(adaptive->row_panel_trace)for(std::size_t i=0;i<retained.size();++i) {
                const auto& c=retained[i];const double hx=source_radius/c.bins;
                adaptive->row_panel_errors.push_back({double(adaptive->history.size()-1),double(i),c.inner_signed,c.inner_remainder,
                    c.outer_parts[0],c.outer_parts[1],c.outer_parts[2],c.outer_parts[3],c.outer_parts[4],c.short_guard,c.outer_parts[5]});
                for(std::size_t j=0;j<c.fill.runs.size();++j) {
                    const auto& r=c.fill.runs[j].run;const auto& e=c.fill.run_flux_errors[j];
                    adaptive->row_panel_rows.push_back({double(adaptive->history.size()-1),double(i),double(c.bins),double(c.row_bins),double(r.iy),double(r.lo),double(r.hi),hx*c.fill.run_areas[j],e[0],e[3],e[2]});
                }
            }
            if(!adaptive->cost_trace)return;
            adaptive->history_flux.push_back(flux);
            for(std::size_t i=0;i<retained.size();++i){const auto& c=retained[i];double cells=0,edges=0;
                for(const auto& r:c.fill.runs){const double width=double(r.run.hi-r.run.lo+1);cells+=width;edges+=std::min(4.,width);}
                adaptive->cost_components.push_back({double(adaptive->history.size()-1),double(i),double(c.bins),c.flux,c.inner_signed,c.outer_signed,c.inner_remainder,c.outer_remainder,c.short_guard,double(c.fill.runs.size()),cells,edges});
            }
        };
        auto current=totals();adaptive->initial_flux=current[0];
        if(adaptive->relative_target>0 && std::isfinite(adaptive->relative_target) &&
           std::isfinite(current[0]) && current[0]!=0)
            adaptive->target=adaptive->relative_target*std::abs(current[0]);
        adaptive->largest_bins=settings.source_bins;
        for(const auto& c:retained)adaptive->largest_bins=std::max({adaptive->largest_bins,c.bins,c.row_bins});
        adaptive->history.push_back({double(settings.source_bins),current[1],double(adaptive->new_maps),double(adaptive->reused_maps),0});
        record_cost(current[0]);
        refinement_work.initial(double(adaptive->new_maps),current[1],current[0]);
        while(std::isfinite(current[1]) && current[1]>adaptive->target) {
            adaptive->mapped.materialize();
            std::size_t selected=retained.size();double priority=-1,total_priority=0,best_efficiency=-1;
            for(std::size_t i=0;i<retained.size();++i) {
                const auto& c=retained[i];const double score=std::abs(c.signed_error-correction_for(c))+c.remainder;
                total_priority+=score;
                const double efficiency=score*(c.order==1?.5:.75)/std::max(1.,c.work_estimate);
                if(!c.blocked && (c.bins<=adaptive->max_bins/2 || c.row_bins<=adaptive->max_bins/2) && efficiency>best_efficiency){selected=i;priority=score;best_efficiency=efficiency;}
            }
            if(selected==retained.size())break;
            adaptive_detail::RefinementWorkTrial refinement_trial(refinement_work,[&]() {
                return std::array<double,3>{double(adaptive->new_maps),current[1],current[0]};
            });
            refinement_trial.kind=1;
            auto& previous=retained[selected];
            const auto trial_start=adaptive->cost_trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            const auto trial_maps=adaptive->new_maps,trial_hits=adaptive->reused_maps;
            // Preserve the predicted remaining work from the local error
            // order, but add the minimum dyadic layer before re-evaluation.
            // A 4x jump is diagnostic only: grid phase can change its coefficient.
            const double local_target=adaptive->target*priority/std::max(total_priority,priority);
            const int factor=std::getenv("LCBININT_DIAGNOSTIC_ADAPTIVE_JUMP4") && previous.bins<=adaptive->max_bins/4 && priority>local_target*std::pow(2.,previous.order)?4:2;
            int factor_x=factor,factor_y=factor,axis=0;
            if(adaptive->directional_adaptive) {
                const bool corrected=correction_for(previous)!=0;
                const double horizontal=(corrected?0:std::abs(previous.inner_signed))+previous.inner_remainder;
                const double vertical=std::abs(corrected?previous.outer_parts[3]:previous.outer_signed)+previous.outer_remainder;
                if(previous.short_guard>horizontal+vertical){factor_x=1;axis=2;}
                else if(horizontal>2*vertical){factor_y=1;axis=1;}
                else if(vertical>2*horizontal){factor_x=1;axis=2;}
                // Keep an unresolved second direction from starving indefinitely.
                if(axis==previous.previous_axis && previous.order==1){factor_x=factor;factor_y=factor;axis=0;}
                if(previous.missing_cap_rows){factor_x=factor;factor_y=1;axis=1;}
                if(previous.coarse_start_axis==2){factor_x=1;factor_y=factor;axis=2;}
                else if(previous.coarse_start_axis==1){factor_x=factor;factor_y=1;axis=1;}
                else if(previous.coarse_start_axis==3){factor_x=factor;factor_y=factor;axis=0;}
                if(previous.bins*factor_x>adaptive->max_axis_ratio*previous.row_bins*factor_y){factor_y=factor;axis=0;}
                if(previous.row_bins*factor_y>adaptive->max_axis_ratio*previous.bins*factor_x){factor_x=factor;axis=0;}
            }
            if(previous.bins*factor_x>adaptive->max_bins)factor_x=1;
            if(previous.row_bins*factor_y>adaptive->max_bins)factor_y=1;
            if(factor_x==1 && factor_y==1){previous.blocked=true;continue;}
            const auto record_trial=[&](bool adopted){if(adaptive->cost_trace)adaptive->cost_trials.push_back({double(adaptive->history.size()-1),double(selected),double(previous.bins),double(previous.bins*factor_x),std::chrono::duration<double>(std::chrono::steady_clock::now()-trial_start).count(),double(adaptive->new_maps-trial_maps),double(adaptive->reused_maps-trial_hits),double(adopted)});};
            adaptive_detail::CartesianRunFillResult failed_cap_rows;
            if(previous.retained_cap_bins && previous.retained_cap_x_bins==previous.bins*factor_x && previous.retained_cap_bins==previous.row_bins*factor_y)failed_cap_rows=previous.cap_rows;
            if(adaptive->local_vertical_caps)failed_cap_rows.component_roots={0};
            const auto append_cap_rows=[&](const adaptive_detail::CartesianRunFillResult& f) {
                for(std::size_t j=0;j<f.runs.size();++j) {
                    failed_cap_rows.runs.push_back({f.runs[j].run,0});failed_cap_rows.run_areas.push_back(f.run_areas[j]);
                    failed_cap_rows.run_flux_errors.push_back(f.run_flux_errors[j]);failed_cap_rows.run_error_fractions.push_back(f.run_error_fractions[j]);
                    failed_cap_rows.run_cores.push_back(f.run_cores.size()==f.runs.size()?f.run_cores[j]:std::nullopt);
                }
            };
            // Local cap candidate: keep the coarse interior integral,
            // refine only its two physical caps, and replace the cap error.
            // A coarse-start continuation can refine x in those bands too;
            // the retained interior and its error remain at their old spacing.
            // Failed candidates retain mapped work for ordinary refinement.
            const double retained_smooth=std::max(0.,previous.outer_parts[2]);
            const double retained_junction_remainder=std::max(0.,previous.outer_parts[4]);
            const double retained_junction_signed=previous.outer_parts[3];
            const double retained_outer_uncertainty=retained_smooth+retained_junction_remainder;
            if(adaptive->local_vertical_caps && previous.retained_cap_bins==0 && (factor_x==1 || previous.coarse_start_axis==3) && factor_y==2 &&
               settings.limb_darkening_d==0 && previous.fill.runs.size()>=25 &&
               (correction_for(previous)!=0?0:std::abs(previous.inner_signed))+previous.inner_remainder+
               std::abs(retained_junction_signed)+retained_outer_uncertainty<=local_target/2) {
                const auto try_caps=[&]()->bool {
                    const auto& old=previous.fill;const auto n=old.runs.size();
                    for(std::size_t j=0;j<n;++j) {
                        if(old.run_flux_errors[j][2]!=0)return false;
                        if(j && (old.runs[j].run.iy!=old.runs[j-1].run.iy+1 ||
                            old.runs[j].run.lo>old.runs[j-1].run.hi || old.runs[j-1].run.lo>old.runs[j].run.hi))return false;
                    }
                    if(old.runs.front().run.iy<(std::numeric_limits<std::int64_t>::min()+4)/2 ||
                       old.runs.back().run.iy>(std::numeric_limits<std::int64_t>::max()-4)/2)return false;
                    const std::size_t a=4,b=n-5;
                    const int cap_x_factor=previous.coarse_start_axis==3?factor_x:1;
                    const double hx=source_radius/previous.bins,hy=source_radius/previous.row_bins,hf=hy/2,hxf=hx/cap_x_factor;
                    std::array<double,7> vl,vu;double derivative_uncertainty1=0,derivative_uncertainty3=0;
                    constexpr std::array<double,7> d1weights{1,9,45,0,45,9,1},d3weights{1,8,13,0,13,8,1};
                    for(int k=0;k<7;++k) {
                        const auto l=a+k-3,u=b+k-3;
                        vl[k]=hx*old.run_areas[l]+old.run_flux_errors[l][0]/hy;
                        vu[k]=hx*old.run_areas[u]+old.run_flux_errors[u][0]/hy;
                        const double row_uncertainty=(old.run_flux_errors[l][3]+old.run_flux_errors[u][3])/hy;
                        derivative_uncertainty1+=d1weights[k]*row_uncertainty/(60*hy);
                        derivative_uncertainty3+=d3weights[k]*row_uncertainty/(8*hy*hy*hy);
                    }
                    const auto dl=adaptive_detail::retained_core_derivatives(vl,hy),du=adaptive_detail::retained_core_derivatives(vu,hy);
                    const double h2=hy*hy,f2=hf*hf;
                    const double interface_signed=(h2-f2)*(dl[0]-du[0])/12-(h2*h2-f2*f2)*(dl[1]-du[1])/720;
                    const double interface_remainder=(h2-f2)*(dl[2]+du[2]+derivative_uncertainty1)/12+
                        (h2*h2-f2*f2)*(dl[3]+du[3]+derivative_uncertainty3)/720+h2*h2*h2*(std::abs(dl[4])+std::abs(du[4]))/30240;
                    if(!std::isfinite(interface_remainder)||interface_remainder>local_target/4)return false;
                    ++adaptive->cap_trials;
                    // The coarse interior was not re-evaluated. Its smooth and
                    // junction errors survive a cap-only replacement.
                    double flux=0,inner=0,inner_rem=0,outer=interface_signed+retained_junction_signed,
                        outer_rem=interface_remainder+retained_outer_uncertainty;
                    for(auto j=a;j<=b;++j) {
                        const double w=j==a||j==b?.5:1;
                        flux+=w*hx*hy*old.run_areas[j];inner+=w*old.run_flux_errors[j][0];inner_rem+=w*old.run_flux_errors[j][3];
                    }
                    const auto saved_counters=probe_counters();
                    for(int side=0;side<2;++side) {
                        const auto anchor=side?b:a,seed_index=side?n-5:4;
                        const auto& sr=old.runs[seed_index].run;
                        const std::int64_t pad=adaptive->cap_initial_pad;
                        std::array<std::int64_t,2> clip=side?
                            std::array<std::int64_t,2>{old.runs[b].run.iy*2,old.runs.back().run.iy*2+pad}:
                            std::array<std::int64_t,2>{old.runs.front().run.iy*2-pad,old.runs[a].run.iy*2};
                        std::vector<AdaptiveEvaluatedCartesianSeed> local{{{(sr.lo+(sr.hi-sr.lo)/2)*hx,sr.iy*hy},0}};
                        auto fs=settings;fs.source_bins=previous.bins*cap_x_factor;
                        std::vector<RetainedAdaptiveComponent> band;
                        while(true) {
                            std::vector<RetainedAdaptiveComponent> next_band;
                            const auto* known_band=band.empty()?nullptr:&band.front().fill;
                            ++adaptive->depth;
                            try {fill_all_adaptive_cartesian_components_multirun(mapper,source,source_radius,fs,finite_magnifier,
                                local,hxf,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&next_band,hf,
                                &old,2,cap_x_factor,&clip,1U<<side,previous.cap_weights[side]?1:-1,known_band);}
                            catch(...){--adaptive->depth;probe_counters()=saved_counters;throw;}
                            --adaptive->depth;probe_counters()=saved_counters;
                            band=std::move(next_band);
                            if(band.size()!=1){for(const auto& c:band)append_cap_rows(c.fill);return false;}
                            bool touches_outer_clip=false;
                            for(const auto& r:band.front().fill.runs)
                                if(r.run.iy==clip[side?1:0])touches_outer_clip=true;
                            if(!touches_outer_clip)break;
                            auto& edge=clip[side?1:0];
                            if((side && edge>std::numeric_limits<std::int64_t>::max()-4) ||
                               (!side && edge<std::numeric_limits<std::int64_t>::min()+4))return false;
                            edge+=side?4:-4;++adaptive->cap_extensions;
                        }
                        const auto& c=band.front();
                        append_cap_rows(c.fill);
                        if(!refined_run_footprint_is_private(c.fill,previous.bins*cap_x_factor/settings.source_bins,run_fill,
                            previous.original_owner,previous.row_bins*2/settings.source_bins,&previous.original_owners))return false;
                        flux+=c.flux;inner+=c.inner_signed;inner_rem+=c.inner_remainder;outer+=c.outer_signed;outer_rem+=c.outer_remainder;
                        bool found=false;
                        for(std::size_t j=0;j<c.fill.runs.size();++j)if(c.fill.runs[j].run.iy==old.runs[anchor].run.iy*2) {
                            if(found)return false;found=true;
                            flux-=.5*hxf*hf*c.fill.run_areas[j];inner-=.5*c.fill.run_flux_errors[j][0];inner_rem-=.5*c.fill.run_flux_errors[j][3];
                        }
                        if(!found)return false;
                    }
                    const double indicator=std::abs(adaptive->defect_correction?interface_signed+retained_junction_signed:inner+outer)+inner_rem+outer_rem;
                    if(!(std::isfinite(flux)&&flux>0&&std::isfinite(indicator)&&inner_rem>=0&&outer_rem>=0&&indicator<=local_target))return false;
                    previous.flux=flux;previous.inner_signed=inner;previous.inner_remainder=inner_rem;
                    previous.outer_signed=outer;previous.outer_remainder=outer_rem;
                    previous.signed_error=inner+outer;previous.remainder=inner_rem+outer_rem;
                    previous.outer_parts={outer-interface_signed-retained_junction_signed,
                        outer_rem-interface_remainder-retained_outer_uncertainty,
                        interface_remainder+retained_smooth,interface_signed+retained_junction_signed,
                        retained_junction_remainder,1};
                    previous.retained_cap_bins=previous.row_bins*2;previous.retained_cap_x_bins=previous.bins*cap_x_factor;previous.cap_rows=std::move(failed_cap_rows);
                    previous.blocked=false;previous.short_guard=0;previous.coarse_start_axis=0;
                    return true;
                };
                const auto trials_before=adaptive->cap_trials;
                bool adopted=false;
                try {adopted=try_caps();}catch(const AdaptiveFluxBudgetExhausted&){adaptive->stop_reason="evaluation_budget";record_trial(false);break;}
                if(adopted) {
                    record_trial(true);
                    ++adaptive->cap_adopted;++adaptive->refinements;
                    adaptive->largest_bins=std::max({adaptive->largest_bins,previous.retained_cap_bins,previous.retained_cap_x_bins});
                    adaptive->refinement_grid_bins.push_back({double(selected),double(previous.bins),double(previous.row_bins),double(previous.retained_cap_x_bins),double(previous.retained_cap_bins),previous.retained_cap_x_bins==previous.bins?2.:0.});
                    current=totals();adaptive->history.push_back({double(previous.bins),current[1],double(adaptive->new_maps),double(adaptive->reused_maps),double(selected)});
                    record_cost(current[0]);continue;
                }
                if(adaptive->cap_trials>trials_before)++adaptive->cap_rejected;
            }
            const auto lifted=adaptive_detail::lift_cartesian_component_run_seeds(previous.fill,0,factor_x,factor_y);
            if(!lifted || lifted->empty()){previous.blocked=true;continue;}
            const int next_bins=previous.bins*factor_x,next_row_bins=previous.row_bins*factor_y;
            const double next_h=source_radius/next_bins,next_row_h=source_radius/next_row_bins;
            std::vector<AdaptiveEvaluatedCartesianSeed> next_seeds;next_seeds.reserve(lifted->size());
            for(const auto& seed:*lifted)next_seeds.push_back({{seed.ix*next_h,seed.iy*next_row_h},0});
            auto next_settings=settings;next_settings.source_bins=next_bins;
            std::vector<RetainedAdaptiveComponent> children;
            const auto counters_before=probe_counters();
            struct CoarseInterpolationScope {
                bool& flag;bool previous;
                ~CoarseInterpolationScope(){flag=previous;}
            } interpolation_scope{adaptive->local_row_interpolation,adaptive->local_row_interpolation};
            if(previous.coarse_start_axis && factor_x==1)adaptive->local_row_interpolation=false;
            ++adaptive->depth;
            try {
                fill_all_adaptive_cartesian_components_multirun(mapper,source,source_radius,next_settings,finite_magnifier,
                    next_seeds,next_h,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&children,next_row_h,
                    // Reuse aligned rows and separately validated smooth cores;
                    // unresolved endpoint fits do not invalidate the interior.
                    (previous.coarse_start_axis==0 || factor_x==1 || adaptive->local_boundary_strips) &&
                    !std::getenv("LCBININT_DIAGNOSTIC_DISABLE_ROW_REUSE") &&
                        (factor_x==1 || adaptive->local_boundary_strips ||
                            (factor_y==1 && std::getenv("LCBININT_DIAGNOSTIC_LOCAL_HORIZONTAL_ROWS")))?&previous.fill:nullptr,factor_y,factor_x,nullptr,3,0,
                    failed_cap_rows.runs.empty()?nullptr:&failed_cap_rows);
            } catch(const AdaptiveFluxBudgetExhausted&) {
                --adaptive->depth;probe_counters()=counters_before;
                adaptive->stop_reason="evaluation_budget";record_trial(false);break;
            } catch(...) {--adaptive->depth;throw;}
            --adaptive->depth;
            // Trial error counters must not be mistaken for adopted flux.
            // Work counts survive in the explicit adaptive context.
            probe_counters()=counters_before;
            const auto collect_children=[&]() {
                RetainedAdaptiveComponent next;next.bins=next_bins;next.row_bins=next_row_bins;next.previous_axis=axis;next.original_owner=previous.original_owner;next.fill.component_roots={0};
                if(children.size()==1 && !std::getenv("LCBININT_DIAGNOSTIC_DISABLE_SINGLE_CHILD_MOVE")) {
                    // The ordinary refinement of one connected component normally
                    // has one child. Transfer its retained rows instead of growing
                    // and copying five vectors that were just constructed.
                    next=std::move(children.front());
                    next.bins=next_bins;next.row_bins=next_row_bins;next.previous_axis=axis;
                    next.original_owner=previous.original_owner;
                } else for(const auto& child:children) {
                    for(int k=0;k<2;++k)next.cap_weights[k]=next.cap_weights[k]||child.cap_weights[k];
                    next.flux+=child.flux;next.signed_error+=child.signed_error;next.remainder+=child.remainder;next.work_estimate+=child.work_estimate;
                    for(int k=0;k<6;++k)next.outer_parts[k]+=child.outer_parts[k];
                    next.coarse_start_axis=std::max(next.coarse_start_axis,child.coarse_start_axis);
                    next.inner_signed+=child.inner_signed;next.outer_signed+=child.outer_signed;next.inner_remainder+=child.inner_remainder;next.outer_remainder+=child.outer_remainder;next.short_guard+=child.short_guard;
                    for(std::size_t j=0;j<child.fill.runs.size();++j) {
                        next.fill.runs.push_back({child.fill.runs[j].run,0});next.fill.run_areas.push_back(child.fill.run_areas[j]);
                        next.fill.run_flux_errors.push_back(child.fill.run_flux_errors[j]);next.fill.run_error_fractions.push_back(child.fill.run_error_fractions[j]);
                        if(child.fill.run_cores.size()==child.fill.runs.size())next.fill.run_cores.push_back(child.fill.run_cores[j]);
                    }
                }
                next.original_owners=previous.original_owners;
                return next;
            };
            auto next=collect_children();
            std::vector<std::size_t> merge_donors,allowed_owners=previous.original_owners;
            double previous_priority=priority;
            bool merge_failed=false,merge_budget=false;
            const auto has_repair_owner=[&](const RetainedAdaptiveComponent& c) {
                const auto synthetic=[&](std::size_t owner){return owner>=run_fill.component_roots.size();};
                return synthetic(c.original_owner) || std::any_of(c.original_owners.begin(),c.original_owners.end(),synthetic);
            };
            bool repair_merge=has_repair_owner(previous);
            if(!repair_merge)for(std::size_t i=0;i<retained.size();++i)
                if(i!=selected && !retained[i].retired && has_repair_owner(retained[i])){repair_merge=true;break;}
            // Ordinary initial-16 refinement already checks privacy once
            // below. Owner closure is needed only for coarse-start stitching
            // or separately repaired images, not every smooth refinement.
            if((settings.source_bins==8 || repair_merge) && !children.empty()) {
                // Disconnected coarse runs may belong to one connected fine
                // image. Close the owner set incrementally; exact completed
                // fine rows and all mapped rays survive every expansion.
                for(std::size_t attempt=0;attempt<retained.size();++attempt) {
                    std::vector<std::size_t> collided;
                    refined_run_footprint_is_private(next.fill,next_bins/settings.source_bins,run_fill,
                        previous.original_owner,next_row_bins/settings.source_bins,&allowed_owners,&collided);
                    // Repaired subcell images have no owner in the original
                    // coarse fill. Include their current occupied footprint:
                    // two initially separate repairs can join on a finer grid.
                    for(std::size_t donor=0;donor<retained.size();++donor) {
                        const auto& c=retained[donor];
                        if(donor==selected || c.retired ||
                           std::find(merge_donors.begin(),merge_donors.end(),donor)!=merge_donors.end())continue;
                        const auto synthetic=[&](std::size_t owner){return owner>=run_fill.component_roots.size();};
                        if(!synthetic(c.original_owner) &&
                           !std::any_of(c.original_owners.begin(),c.original_owners.end(),synthetic))continue;
                        if(adaptive_cartesian_retained_overlap(next,c))collided.push_back(c.original_owner);
                    }
                    if(collided.empty())break;
                    std::vector<std::size_t> added;
                    for(const auto owner:collided) {
                        std::size_t donor=retained.size();
                        for(std::size_t i=0;i<retained.size();++i) {
                            const auto& c=retained[i];
                            if(i!=selected && !c.retired &&
                               (c.original_owner==owner || std::find(c.original_owners.begin(),c.original_owners.end(),owner)!=c.original_owners.end())) {donor=i;break;}
                        }
                        if(donor==retained.size()){merge_failed=true;break;}
                        const auto& c=retained[donor];
                        // Never discard an already finer donor integral.
                        if(c.bins>next_bins || c.row_bins>next_row_bins || c.retained_cap_bins>next_row_bins || c.retained_cap_x_bins>next_bins ||
                           next_bins%c.bins || next_row_bins%c.row_bins){merge_failed=true;break;}
                        if(std::find(merge_donors.begin(),merge_donors.end(),donor)!=merge_donors.end())continue;
                        added.push_back(donor);merge_donors.push_back(donor);
                        previous_priority+=std::abs(c.signed_error-correction_for(c))+c.remainder;
                        if(c.original_owners.empty())allowed_owners.push_back(c.original_owner);
                        else allowed_owners.insert(allowed_owners.end(),c.original_owners.begin(),c.original_owners.end());
                        const auto seeds=adaptive_detail::lift_cartesian_component_run_seeds(c.fill,0,next_bins/c.bins,next_row_bins/c.row_bins);
                        if(!seeds || seeds->empty()){merge_failed=true;break;}
                        for(const auto& seed:*seeds)next_seeds.push_back({{seed.ix*next_h,seed.iy*next_row_h},0});
                    }
                    if(merge_failed || added.empty()){merge_failed=true;break;}
                    auto completed=std::move(next.fill);children.clear();++adaptive->depth;
                    try {
                        fill_all_adaptive_cartesian_components_multirun(mapper,source,source_radius,next_settings,finite_magnifier,
                            next_seeds,next_h,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&children,next_row_h,
                            nullptr,1,1,nullptr,3,0,&completed);
                    } catch(const AdaptiveFluxBudgetExhausted&) {
                        --adaptive->depth;probe_counters()=counters_before;adaptive->stop_reason="evaluation_budget";merge_failed=merge_budget=true;break;
                    } catch(...) {--adaptive->depth;probe_counters()=counters_before;throw;}
                    --adaptive->depth;probe_counters()=counters_before;next=collect_children();
                    if(children.empty()){merge_failed=true;break;}
                }
                next.original_owners=allowed_owners;
            }
            if(merge_failed || children.empty() || !(std::isfinite(next.flux)&&next.flux>0&&std::isfinite(next.signed_error)&&std::isfinite(next.remainder)&&next.remainder>=0)
                || !refined_run_footprint_is_private(next.fill,next_bins/settings.source_bins,run_fill,previous.original_owner,next_row_bins/settings.source_bins,&next.original_owners)) {
                if(merge_budget){record_trial(false);break;}
                if(merge_failed){previous.remainder=std::max(previous.remainder,std::abs(previous.flux));previous.short_guard=std::max(previous.short_guard,std::abs(previous.flux));current=totals();}
                previous.blocked=true;adaptive->stop_reason="unresolved_component";record_trial(false);continue;
            }
            const double next_priority=std::abs(next.signed_error-correction_for(next))+next.remainder;
            next.order=next_priority>previous_priority/factor?1:2;
            record_trial(true);
            adaptive->refinement_grid_bins.push_back({double(selected),double(previous.bins),double(previous.row_bins),double(next_bins),double(next_row_bins),double(axis)});
            for(const auto donor:merge_donors) {
                auto& c=retained[donor];c.flux=c.signed_error=c.remainder=0;c.short_guard=0;
                c.inner_signed=c.outer_signed=c.inner_remainder=c.outer_remainder=0;c.outer_parts={};
                c.fill={};c.blocked=true;c.retired=true;++adaptive->coarse_start_merges;
            }
            previous=std::move(next);++adaptive->refinements;
            adaptive->largest_bins=std::max({adaptive->largest_bins,next_bins,next_row_bins});current=totals();
            adaptive->history.push_back({double(next_bins),current[1],double(adaptive->new_maps),double(adaptive->reused_maps),double(selected)});
            record_cost(current[0]);
        }
        adaptive->met=adaptive->unresolved_centres==0 && std::isfinite(current[1])&&current[1]<=adaptive->target;
        if(adaptive->unresolved_centres)adaptive->stop_reason="unresolved_centre_image";
        if(adaptive->met)adaptive->stop_reason="estimated_target_met";
        for(const auto& c:retained){if(c.retired)continue;const double correction=correction_for(c);if(adaptive->local_vertical_caps)adaptive->mixed_component_grids.push_back({double(c.bins),double(c.row_bins),double(c.retained_cap_bins)});adaptive->components.push_back({double(std::max({c.bins,c.row_bins,c.retained_cap_bins,c.retained_cap_x_bins})),c.flux+correction,c.signed_error-correction,c.remainder,double(c.fill.runs.size())});adaptive->component_grid_bins.push_back({double(c.bins),double(c.row_bins)});}
        total_area=current[0]/(incr*row_spacing);
        if(phase_trace)adaptive->adaptive_control_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-control_started).count()-(adaptive->run_fill_seconds+adaptive->error_aggregation_seconds-children_seconds_before);
    }
    return static_cast<double>(total_area);
}
