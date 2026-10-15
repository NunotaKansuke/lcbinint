using detail::PolarRealRoots;
using detail::polar_real_polynomial_roots;
using detail::polar_continue_stationary_roots;
using detail::polar_checked_limb_partition;

// Closed nine-node Newton-Cotes with a four-node Stieltjes extension.
// The returned rule is exact through degree nine; its independent embedded
// reference is exact through degree seventeen. The extension polynomial is
// y^4-(501707/2010794)y^2+376379/48259056, y=x-1/2.
struct PolarEmbeddedNineRule {
    static constexpr std::array<double,13> x={0,.03862309348069189,.125,.25,
        .30858876383313893,.375,.5,.625,.6914112361668611,.75,.875,.9613769065193081,1};
    static constexpr std::array<int,9> index={0,2,3,5,6,7,9,10,12};
    static constexpr std::array<double,9> low={989.0/28350,5888.0/28350,-928.0/28350,
        10496.0/28350,-4540.0/28350,10496.0/28350,-928.0/28350,5888.0/28350,989.0/28350};
    static constexpr std::array<double,13> high={.010679517161151978,.06390411938465332,
        .10711837802953139,.14670594721734698,-.029457518196350035,.1433748736147592,
        .11534936557781439,.1433748736147592,-.029457518196350035,.14670594721734698,
        .10711837802953139,.06390411938465332,.010679517161151978};
    static double interpolate(const std::array<double,13>& values,double u) {
        static const auto weights=[] {
            std::array<double,13> result{};
            for(int j=0;j<13;++j) {
                double product=1;
                for(int k=0;k<13;++k) if(k!=j) product*=x[j]-x[k];
                result[j]=1/product;
            }
            return result;
        }();
        double numerator=0,denominator=0;
        for(int j=0;j<13;++j) {
            if(std::abs(u-x[j])<=32*std::numeric_limits<double>::epsilon()) return values[j];
            const double term=weights[j]/(u-x[j]);
            numerator+=term*values[j];denominator+=term;
        }
        return numerator/denominator;
    }
    static std::array<double,3> integrate(const std::array<double,13>& values,double width) {
        double lower=0,higher=0,absolute=0;
        for(int j=0;j<9;++j) {lower+=low[j]*values[index[j]];absolute+=std::abs(low[j]*values[index[j]]);}
        for(int j=0;j<13;++j) {higher+=high[j]*values[j];absolute+=std::abs(high[j]*values[j]);}
        return {width*lower,2*width*std::abs(lower-higher),256*std::numeric_limits<double>::epsilon()*width*absolute};
    }
};

// Uniform-node barycentric interpolation avoids rebuilding N Lagrange
// products for every retained physical observation.
template<std::size_t N>
double polar_interpolate_equispaced(const std::array<double,N>& values,double u)
{
    static constexpr auto weights=[] {
        std::array<double,N> result{};result[0]=1;
        for(std::size_t j=1;j<N;++j) result[j]=-result[j-1]*(N-j)/j;
        return result;
    }();
    double numerator=0,denominator=0;
    for(std::size_t j=0;j<N;++j) {
        if(std::abs(u-j)<=32*std::numeric_limits<double>::epsilon()) return values[j];
        const double term=weights[j]/(u-j);
        numerator+=term*values[j];denominator+=term;
    }
    return numerator/denominator;
}

// Included after the lens mapper and coarse polar support definitions.
// Radial source-limb roots replace cut-cell counting. The r dr measure is
// exact for Uniform; endpoint-weighted quadrature integrates linear LD.
// Angular Boole panels retain all previously evaluated radial columns.
struct AdaptivePolarPanel {
    std::int64_t a, b;
    int depth;
    double flux, angular_error, radial_error;
    int error_kind=0;
    bool cap=false,transformed=false;
    double t_lo=0.0,t_hi=0.0,t_scale=0.0;
    double tip=0.0,left_width=0.0,width=0.0;
    double tip_error=0.0;
    double reference_flux=0.0,quadrature_error=0.0;
    double uniform_fold=0.0,dark_fold=0.0,contact_bracket=0.0;
    bool incompatible=false;
    int left_runs=0,right_runs=0;
    bool contact_disabled=false;
    int base_depth=0;
    std::shared_ptr<std::vector<AdaptivePolarPanel>> pieces;
    double coverage_guard=0.0;
    double legacy_quadrature_error=0.0;
    bool high_rule=false;
};


double adaptive_polar_from_support(
    const BinaryLensMapper& mapper, SourcePosition source, double rho,
    const FiniteSourceSettings& settings, const PolarAreaDiagnostics& support,
    const std::vector<SourcePosition>& seeds, double coarse_magnification, AdaptivePolarRequest& work)
{
    work.executed = true;
    auto& refinement_work=adaptive_detail::refinement_work_trace();
    refinement_work.reset();
    work.angular_error_model="independent_angular_and_ld_flood_fill_polar";
    work.discovery_maps=work.maps;
    // Keep the lower per-column cost when discovery already consumed a
    // substantial part of the explicit map budget. Both radial rules use
    // the same tolerance and contribute their own indicated error.
    // A discarded discovery attempt is charged to maps too. Reserve half
    // the total budget for integration before selecting the cheaper rule;
    // otherwise a successful retry would unnecessarily force Simpson.
    const bool gauss_radial=work.maps < work.budget/2;
    work.radial_quadrature=settings.limb_darkening_c==0.0 ? "exact_uniform"
        : gauss_radial ? "witnessed_gauss_chebyshev_u_1" : "witnessed_simpson_budget_fallback";
    work.effective_angular_ratio=work.initial_angular_ratio;
    work.initial_magnification = coarse_magnification;
    work.initial_phi_bins = support.phi_bins;
    // Reserve 8% absolute-density / 32% relative-column error locally.
    // The witnessed radial indicator is still charged in full to final
    // angular+radial admission; this never changes the requested tolerance.
    double radial_allocation=4.0;
    if(const char* allocation=std::getenv("LCBININT_DIAGNOSTIC_POLAR_RADIAL_ALLOCATION")) {
        radial_allocation=std::atof(allocation);
        if(!std::isfinite(radial_allocation) || radial_allocation<1 || radial_allocation>4)
            throw std::invalid_argument("diagnostic radial allocation must be in [1,4]");
    }
    const AdaptivePolarMapper counted{mapper, work};
    const double norm = source_flux(rho, settings);
    double target = norm * (work.relative_tolerance > 0.0
        ? work.relative_tolerance * std::max(1.0, std::abs(coarse_magnification))
        : work.absolute_tolerance);
    const auto radial_relative_target = [&]() {
        // The angular controller updates target after discovering the flux.
        // Dividing that new target by the old coarse flux can inflate every
        // later column's radial allowance by orders of magnitude. Relative
        // requests already supply the correct dimensionless allocation.
        return work.relative_tolerance > 0.0 ? work.relative_tolerance
            : target/(norm*std::max(1.0,std::abs(work.initial_magnification)));
    };
    const double rho2 = rho * rho;
    const long double wide_separation = mapper.separation.real();
    const long double wide_source_x = static_cast<long double>(source.x) + mapper.source_origin_shift;
    const long double wide_source_y = source.y;
    const detail::PolarLimbPolynomial limb_polynomial(wide_separation, mapper.m1, mapper.m2,
        wide_source_x, wide_source_y, rho2);
    const bool wide_partition_shadow=std::getenv("LCBININT_DIAGNOSTIC_POLAR_PARTITION_SHADOW")!=nullptr;
    const double upper_radius = std::abs(wide_separation)
        + std::hypot(wide_source_x, wide_source_y) + rho + 4.0;
    const double root_tolerance = std::max(rho * 1e-11,
        rho * std::min(1e-3, target / (norm * std::max(1.0, std::abs(coarse_magnification)))) * 0.002);
    std::set<int> active;
    const int stride = std::max(1, static_cast<int>(work.initial_angular_ratio / 2.0));
    const int groups = static_cast<int>((static_cast<std::int64_t>(support.phi_bins) + stride - 1) / stride);
    const double panel_dphi = 2.0 * kPi / groups;
    for (const auto& run : support.support_runs) {
        const int group = static_cast<int>(static_cast<std::int64_t>(run.iphi) * groups / support.phi_bins);
        for (int d = -1; d <= 1; ++d) active.insert(static_cast<int>((static_cast<std::int64_t>(group) + d + groups) % groups));
    }
    // A centre image can be thinner than a coarse radial cell. Preserve its
    // actual angle as a refinement guard even when flood fill finds no cell.
    std::vector<std::array<double, 3>> seed_angles;
    seed_angles.reserve(seeds.size());
    for (const auto& seed : seeds) {
        const double r = std::hypot(seed.x, seed.y);
        if (!(r > 0.0)) continue;
        double phi = std::atan2(seed.y, seed.x);
        if (phi < 0.0) phi += 2.0 * kPi;
        const int index = std::min(groups - 1, static_cast<int>(phi / panel_dphi));
        for (int d = -1; d <= 1; ++d) active.insert(static_cast<int>((static_cast<std::int64_t>(index) + d + groups) % groups));
        const double ux = -seed.y / r, uy = seed.x / r;
        double dx = ux, dy = uy;
        const double rx = seed.x / r, ry = seed.y / r;
        double jrx = rx, jry = ry;
        for (int lens = 0; lens < 2; ++lens) {
            const double mass = lens == 0 ? mapper.m1 : mapper.m2;
            const double x = seed.x - (lens == 0 ? mapper.separation.real() : 0.0), y = seed.y;
            const double den = x*x + y*y, dden = 2.0 * (x*ux + y*uy);
            dx -= mass * (ux*den - x*dden) / (den*den);
            dy -= mass * (uy*den - y*dden) / (den*den);
            const double radial_dden = 2.0 * (x*rx + y*ry);
            jrx -= mass * (rx*den - x*radial_dden) / (den*den);
            jry -= mass * (ry*den - y*radial_dden) / (den*den);
        }
        // The image ellipse is tilted: angular extent follows the inverse
        // Jacobian, allowing its radius to change with angle. A fixed-radius
        // tangential derivative can severely underestimate a thin arc's span.
        const double determinant = jrx * dy - jry * dx;
        const double angular_extent = rho * std::hypot(jrx, jry) / std::max(r * std::abs(determinant), 1e-300);
        seed_angles.push_back({phi, std::min(kPi, angular_extent), r});
    }
    if (active.empty() || !(target > 0.0)) {
        work.stop_reason = "unresolved_support";
        work.error_indicator = std::numeric_limits<double>::infinity();
        return coarse_magnification;
    }
    const std::int64_t ticks = std::int64_t{1} << (work.max_angular_depth + 2);
    // Equal panels close the circle exactly. A shorter final stride would
    // make the refinement lattice asymmetric under phi -> 2*pi-phi.
    const std::int64_t circle = ticks * groups;
    struct FootprintWitness { std::int64_t key=-1; double weight=0.0; };
    struct FootprintGroup {
        FootprintWitness maximum{},minimum{};
        int maximum_count=-1,minimum_count=-1;
        std::vector<FootprintWitness> representatives;
    };
    std::unordered_map<int,FootprintGroup> witnesses;
    auto coarse_runs=support.support_runs;
    std::sort(coarse_runs.begin(),coarse_runs.end(),[](const auto& a,const auto& b) {
        return a.iphi!=b.iphi ? a.iphi<b.iphi : a.left<b.left;
    });
    for (std::size_t i=0;i<coarse_runs.size();) {
        const int iphi=coarse_runs[i].iphi;
        int count=0;double weight=0;
        while (i<coarse_runs.size() && coarse_runs[i].iphi==iphi) {
            int left=coarse_runs[i].left,right=coarse_runs[i].right;++i;
            while (i<coarse_runs.size() && coarse_runs[i].iphi==iphi && static_cast<std::int64_t>(coarse_runs[i].left)<=static_cast<std::int64_t>(right)+1) {
                right=std::max(right,coarse_runs[i].right);++i;
            }
            ++count;weight+=(static_cast<double>(right)-left+1.0)*(static_cast<double>(right)+left+1.0);
        }
        const double position=(static_cast<double>(iphi)+.5)*groups/support.phi_bins;
        const int group=std::min(groups-1,static_cast<int>(position));
        const FootprintWitness point{static_cast<std::int64_t>(std::llround(position*ticks))%circle,weight};
        auto& entry=witnesses[group];
        if (entry.maximum.key<0 || weight>entry.maximum.weight) entry.maximum=point;
        if (entry.minimum.key<0 || weight<entry.minimum.weight) entry.minimum=point;
        // Retain both coarse run-count extrema as actual geometric probes.
        // Counts from the coarse lattice are proposal data, not certificates.
        if (count>entry.maximum_count) {
            entry.maximum_count=count;
            if (entry.representatives.empty()) entry.representatives.resize(2);
            entry.representatives[0]=point;
        }
        if (entry.minimum_count<0 || count<entry.minimum_count) {
            entry.minimum_count=count;
            if (entry.representatives.empty()) entry.representatives.resize(2);
            entry.representatives[1]=point;
        }
    }
    for (auto& item:witnesses) {
        auto& entry=item.second;
        entry.representatives.push_back(entry.maximum);
        entry.representatives.push_back(entry.minimum);
        std::sort(entry.representatives.begin(),entry.representatives.end(),[](const auto& a,const auto& b) {return a.key<b.key;});
        entry.representatives.erase(std::unique(entry.representatives.begin(),entry.representatives.end(),
            [](const auto& a,const auto& b) {return a.key==b.key;}),entry.representatives.end());
    }
    struct BoundaryPredictions {
        std::array<std::array<double,2>,6> values{};
        std::size_t count=0;
        void push_back(std::array<double,2> value) {
            if (count<values.size()) values[count++]=value;
        }
    };
    struct Column {
        double flux = 0.0, error = 0.0; int runs = 0;
        mutable std::array<double,16> critical_r{},critical_f{};
        mutable std::size_t critical_count=0;
        mutable PolarRealRoots stationary;
        mutable bool critical_ready=false;
        double phi=0.0;
        BoundaryPredictions boundaries;
    };
    std::unordered_map<std::uint64_t, Column> columns;
    columns.reserve(512);
    // Execution order jumps between adaptive panels. Preserve the nearest
    // physical angular neighbour instead of using the last executed column.
    // unordered_map references remain stable across rehashing.
    struct AngularColumnIndex {
        using Entry=std::pair<double,const Column*>;
        std::vector<Entry> entries;
        AngularColumnIndex(){entries.reserve(512);}
        bool empty()const{return entries.empty();}
        auto begin(){return entries.begin();}
        auto end(){return entries.end();}
        auto lower_bound(double phi) {
            return std::lower_bound(entries.begin(),entries.end(),phi,
                [](const Entry& entry,double value){return entry.first<value;});
        }
        void emplace(double phi,const Column* column) {
            const auto at=lower_bound(phi);
            if(at==end() || at->first!=phi)entries.insert(at,{phi,column});
        }
    };
    AngularColumnIndex angular_columns;
    PolarRealRoots previous_extrema;
    bool have_previous_extrema=false;
    BoundaryPredictions previous_boundaries;
    double previous_phi=0.0;
    const auto column_at = [&](double phi) -> const Column& {
        if(!(phi>=0.0 && phi<2.0*kPi))phi=std::fmod(phi,2.0*kPi);
        if (phi<0.0) phi+=2.0*kPi;
        if (phi==0.0) phi=0.0;
        std::uint64_t key;
        std::memcpy(&key,&phi,sizeof(key));
        const auto found = columns.find(key);
        if (found != columns.end()) { ++work.reused_columns; return found->second; }
        if (!angular_columns.empty()) {
            auto right=angular_columns.lower_bound(phi);
            auto left=right==angular_columns.begin() ? std::prev(angular_columns.end()) : std::prev(right);
            if (right==angular_columns.end()) right=angular_columns.begin();
            const auto periodic_difference=[](double d) {
                return d>kPi?d-2*kPi:(d<-kPi?d+2*kPi:d);
            };
            const double dl=periodic_difference(phi-left->first);
            const double dr=periodic_difference(phi-right->first);
            const auto neighbour=std::abs(dl)<=std::abs(dr) ? left : right;
            const double delta=std::abs(dl)<=std::abs(dr) ? dl : dr;
            previous_extrema=neighbour->second->stationary;
            previous_boundaries=neighbour->second->boundaries;
            previous_phi=phi-delta;
            have_previous_extrema=true;
        }
        const double ux = std::cos(phi), uy = std::sin(phi);
        BoundaryPredictions current_boundaries;
        const auto residual = [&](double r) {
            const auto mapped = map_lens_real(counted, r * ux, r * uy);
            const double value = distance_squared(mapped, source) - rho2;
            return std::isfinite(value) ? value : std::numeric_limits<double>::infinity();
        };
        const auto radial_state = [&](double r) {
            count_adaptive_polar_map(work);
            const double image_x=r*ux,image_y=r*uy;
            SourcePosition mapped{image_x,image_y};
            double jxx=1.0,jxy=0.0;
            for (int lens = 0; lens < 2; ++lens) {
                const double mass = lens == 0 ? mapper.m1 : mapper.m2;
                const double x = image_x - (lens == 0 ? mapper.separation.real() : 0.0);
                const double y = image_y, inv=1.0/(x*x+y*y);
                mapped.x-=mass*x*inv;
                mapped.y-=mass*y*inv;
                const double weighted_inverse_squared=mass*inv*inv;
                jxx+=(x*x-y*y)*weighted_inverse_squared;
                jxy+=2.0*x*y*weighted_inverse_squared;
            }
            mapped.x-=mapper.source_origin_shift;
            const double jyy=2.0-jxx,dx=jxx*ux+jxy*uy,dy=jxy*ux+jyy*uy;
            const double tx=r*(-jxx*uy+jxy*ux),ty=r*(-jxy*uy+jyy*ux);
            return std::array<double, 4>{distance_squared(mapped, source) - rho2,
                2.0 * ((mapped.x - source.x) * dx + (mapped.y - source.y) * dy), dx * dx + dy * dy,
                2.0*((mapped.x-source.x)*tx+(mapped.y-source.y)*ty)};
        };
        const auto root = [&](double a, double b, double fa, double fb) {
            double x = a + (b - a) * (-fa / (fb - fa));
            double distance=std::numeric_limits<double>::infinity();
            const double delta_phi=std::remainder(phi-previous_phi,2.0*kPi);
            for (std::size_t index=0;index<previous_boundaries.count;++index) {
                const auto& old=previous_boundaries.values[index];
                const double guess=old[0]+old[1]*delta_phi;
                if (guess>a && guess<b && std::abs(guess-.5*(a+b))<distance) {
                    x=guess;distance=std::abs(guess-.5*(a+b));
                }
            }
            if (std::isfinite(distance)) ++work.boundary_predictions;
            double uncertainty = b - a;
            for (int it = 0; it < 40; ++it) {
                if (!(x > a && x < b)) x = 0.5 * (a + b);
                const auto state = radial_state(x);
                const double f = state[0], df = state[1];
                const double step = std::abs(f / df);
                if (std::isfinite(step) && step <= std::max(root_tolerance, 8.0 * std::numeric_limits<double>::epsilon() * std::abs(x))) {
                    // A small Newton step alone does not bound a boundary.
                    // Verify actual inside/outside probes, as Cartesian does.
                    const double radius=std::max({2.0*step,.25*root_tolerance,
                        32.0*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(x))});
                    // The current physical residual is already known.
                    // Reuse x as one bracket endpoint and evaluate only past
                    // the Newton proposal; a sign change still verifies the
                    // physical inside/outside bracket before returning.
                    const double other=f==0.0 ? std::clamp(x+std::copysign(radius,df),a,b) :
                        std::clamp(x-std::copysign(radius,f/df),a,b);
                    const double fother=residual(other);
                    const double lo=std::min(x,other),hi=std::max(x,other);
                    if (std::isfinite(f) && std::isfinite(fother) && (f<=0.0)!=(fother<=0.0)) {
                        // The physical bracket already bounds the root.
                        // Use the available Newton correction as the returned
                        // coordinate instead of leaving a tolerance-sized
                        // residual in the LD endpoint factorization.
                        const double polished=std::clamp(x-f/df,lo,hi);
                        current_boundaries.push_back({polished,-state[3]/df});
                        ++work.boundary_bracket_checks;
                        return std::pair<double,double>{polished,std::max(polished-lo,hi-polished)};
                    }
                    ++work.boundary_bracket_retries;
                }
                if ((f <= 0.0) == (fa <= 0.0)) { a = x; fa = f; }
                else { b = x; fb = f; }
                uncertainty = b - a;
                x -= f / df;
                if (uncertainty <= root_tolerance) {
                    current_boundaries.push_back({.5*(a+b),0.0});
                    return std::pair<double, double>{0.5 * (a + b), uncertainty};
                }
            }
            current_boundaries.push_back({.5*(a+b),0.0});
            return std::pair<double, double>{0.5 * (a + b), uncertainty};
        };
        const auto polynomial = limb_polynomial.coefficients(ux, uy);
        std::array<long double, 7> derivative{};
        for (int i = 1; i <= 6; ++i) derivative[i - 1] = i * polynomial[i];
        const double upper = upper_radius;
        PolarRealRoots extrema;
        std::array<int,18> partition_counts{};
        bool direct_partition=false;
        bool wide_partition_used=false;
        if(have_previous_extrema) {
            ++work.limb_partition_checks;
            direct_partition=polar_checked_limb_partition(polynomial,upper,previous_extrema,partition_counts,&wide_partition_used);
            if (!direct_partition) {
                // Extrema are proposals here: completeness is checked on the
                // full sextic below, not assumed from Newton convergence.
                PolarRealRoots predicted;
                bool ordered=true;
                for (long double old : previous_extrema) {
                    long double x=old;
                    for (int iteration=0; iteration<8; ++iteration) {
                        long double f=derivative[5],df=0;
                        for (int j=4;j>=0;--j) {df=df*x+f;f=f*x+derivative[j];}
                        const long double next=x-f/df;
                        if (!std::isfinite(next) || !(next>0 && next<upper)) break;
                        const bool stable=static_cast<double>(next)==static_cast<double>(x);
                        x=next;
                        if(stable) break;
                    }
                    if (!(x>0 && x<upper) || (!predicted.empty() && x<=predicted.back())) {
                        ordered=false;break;
                    }
                    predicted.push_back(x);
                }
                if(ordered) {
                    direct_partition=polar_checked_limb_partition(polynomial,upper,predicted,partition_counts,&wide_partition_used);
                    // Stationary continuation independently checks these
                    // proposals, including after an inconclusive limb filter.
                    previous_extrema=predicted;
                }
            }
        }
        const auto solve_stationary=[&]() {
            extrema.count=0;
            if(have_previous_extrema && polar_continue_stationary_roots(derivative,upper,previous_extrema,extrema))
                ++work.stationary_continuations;
            else {
                extrema.count=0;
                polar_real_polynomial_roots(derivative,5,0.0L,upper,extrema);
                ++work.stationary_full_solves;
            }
        };
        if(direct_partition) extrema=previous_extrema;else solve_stationary();
        std::array<double, 18> cuts{};
        std::array<double,18> cut_values{};
        std::size_t cut_count=0;
        const auto prepare_cuts=[&]() {
            cut_count=1;cuts[0]=0.0;cut_values[0]=std::numeric_limits<double>::infinity();
            for(long double r:extrema) cuts[cut_count++]=static_cast<double>(r);
            cuts[cut_count++]=upper;
            for(std::size_t j=1;j<cut_count;++j) cut_values[j]=residual(cuts[j]);
        };
        prepare_cuts();
        if(direct_partition) {
            for(std::size_t j=1;j<cut_count;++j)
                if(((cut_values[j-1]<=0)!=(cut_values[j]<=0))!=(partition_counts[j-1]==1)) {
                    direct_partition=false;break;
                }
            if(!direct_partition) {solve_stationary();prepare_cuts();}
            else {
                ++work.limb_partition_accepts;
                if(wide_partition_shadow && wide_partition_used) {
                    PolarRealRoots checked;
                    polar_real_polynomial_roots(derivative,5,0,upper,checked);
                    int physical_changes=0,predicted_changes=0;bool inside=false;
                    for(long double r:checked) {
                        const auto mapped=map_binary_lens_real(mapper,static_cast<double>(r)*ux,static_cast<double>(r)*uy);
                        const bool next_inside=distance_squared(mapped,source)<=rho2;
                        physical_changes+=next_inside!=inside;inside=next_inside;
                    }
                    const auto mapped=map_binary_lens_real(mapper,upper*ux,upper*uy);
                    physical_changes+=(distance_squared(mapped,source)<=rho2)!=inside;
                    for(std::size_t j=0;j+1<cut_count;++j)predicted_changes+=partition_counts[j];
                    if(physical_changes!=predicted_changes)
                        throw std::runtime_error("polar wide partition physical-root-count disagreement");
                }
            }
        }
        previous_extrema=extrema;have_previous_extrema=true;
        Column result;
        result.phi=phi;result.critical_ready=!direct_partition;
        struct RadialPanel { double a, b, fa, fm, fb, fq1, fq3, flux, error; int depth; };
        double start = -1.0, start_error = 0.0;
        double previous = cuts.front(), fprevious = std::numeric_limits<double>::infinity();
        for (std::size_t i = 1; i < cut_count; ++i) {
            const double next = cuts[i], fnext = cut_values[i];
            if (i+1<cut_count && !direct_partition) {
                const auto index=result.critical_count++;
                result.critical_r[index]=next;result.critical_f[index]=fnext;
            }
            if ((fprevious <= 0.0) != (fnext <= 0.0)) {
                const auto edge = root(previous, next, fprevious, fnext);
                if (fnext <= 0.0) { start = edge.first; start_error = edge.second; }
                else if (start >= 0.0) {
                    ++result.runs;
                    ++work.boundary_runs;
                    const double end = edge.first, end_error = edge.second, width = end - start;
                    // Linear LD has intensity <= 1. The area swept by a
                    // boundary uncertainty delta is bounded by r*delta +
                    // delta^2/2, including when delta is comparable to r.
                    result.error += (std::abs(start)+.5*start_error)*start_error
                        + (std::abs(end)+.5*end_error)*end_error;
                    result.flux += (1.0 - settings.limb_darkening_c) * 0.5 * (end - start) * (end + start);
                    if (settings.limb_darkening_c != 0.0 && width > 0.0) {
                        if(gauss_radial) {
                            // The actual source-limb roots define x in [-1,1].
                            // Divide out its endpoint weight sqrt(1-x^2),
                            // evaluating the full lens map at every node.
                            // Nested Gauss-Chebyshev U rules reuse all 1 -> 3 -> 7
                            // -> 15 -> 31 nodes; no image/ellipse flux model.
                            // P6=(r-start)(r-end) Q4. Cancelling the limb
                            // endpoint weight and secondary r^2 denominator
                            // gives g=c*span^2/rho*sqrt(Q4/(|u|^2*|ru-a|^2)).
                            // This is the full rational lens equation, not an
                            // image approximation. Physical witnesses below
                            // measure deflation/root/coefficient uncertainty.
                            std::array<long double,7> quotient{};
                            const long double root_sum=static_cast<long double>(start)+end;
                            const long double root_product=static_cast<long double>(start)*end;
                            for(int j=4;j>=0;--j)
                                quotient[j]=polynomial[j+2]+root_sum*quotient[j+1]-root_product*quotient[j+2];
                            const long double center=.5L*root_sum,span=.5L*(static_cast<long double>(end)-start);
                            // Shift Q4 to the local interval to avoid repeated
                            // cancellation of powers of the global radius.
                            for(int j=3;j>=0;--j) for(int k=j;k<4;++k) quotient[k]+=center*quotient[k+1];
                            long double power=1;
                            for(int j=0;j<5;++j) {quotient[j]*=power;power*=span;}
                            const long double direction_norm=static_cast<long double>(ux)*ux+static_cast<long double>(uy)*uy;
                            const long double model_scale=settings.limb_darkening_c*span*span/rho;
                            struct WeightedSample {double value=0,square=0,slope=0;};
                            std::array<WeightedSample,9> initial{};
                            std::vector<WeightedSample> expanded;
                            WeightedSample* samples=initial.data();
                            int divisor=2,depth=0;
                            static constexpr double cs[7]={.9238795325112867561,.7071067811865475244,
                                .3826834323650897717,0,-.3826834323650897717,-.7071067811865475244,-.9238795325112867561};
                            static constexpr double sn[7]={.3826834323650897717,.7071067811865475244,
                                .9238795325112867561,1,.9238795325112867561,.7071067811865475244,.3826834323650897717};
                            const auto evaluate=[&](double cosine,double sine,bool with_derivative=true) {
                                long double q=quotient[4],dq=0;
                                if(with_derivative) {
                                    for(int j=3;j>=0;--j) {dq=dq*cosine+q;q=q*cosine+quotient[j];}
                                } else {
                                    for(int j=3;j>=0;--j)q=q*cosine+quotient[j];
                                }
                                const long double r=center+span*cosine;
                                const long double x=r*ux-wide_separation,y=r*uy;
                                const long double den=direction_norm*(x*x+y*y);
                                if(!(q>0 && den>0) || !std::isfinite(q) || !std::isfinite(den))
                                    throw std::runtime_error("unresolved polar LD polynomial deflation");
                                const double value=static_cast<double>(model_scale)*std::sqrt(static_cast<double>(q/den));
                                double slope=0;
                                if(with_derivative) {
                                    const long double dden=2*direction_norm*span*(x*ux+y*uy);
                                    slope=.5*value*static_cast<double>(dq/q-dden/den);
                                }
                                return WeightedSample{value,sine*sine,slope};
                            };
                            const auto physical_value=[&](double cosine,double sine) {
                                const double radius=start+.5*width*(1+cosine);
                                const double mu=std::sqrt(std::max(0.0,-residual(radius)/rho2));
                                return settings.limb_darkening_c*radius*mu*.5*width/sine;
                            };
                            samples[1]=evaluate(0,1,false);
                            // Non-nested physical samples detect agreement of
                            // embedded rules that share the same blind spot.
                            // Reuse the witnesses through every refinement;
                            // they are checks, never an image-flux model.
                            static constexpr double witness_x[4]={-1.0/3,1.0/3,-.8,.8};
                            const double witness_sine=std::sqrt(8.0/9);
                            std::array<double,4> witness{};
                            int witness_count=0;
                            double model_floor=0;
                            const auto extend_witnesses=[&](int count) {
                                for(;witness_count<count;++witness_count) {
                                    const int k=witness_count;
                                    const double sine=k<2 ? witness_sine : .6;
                                    witness[k]=physical_value(witness_x[k],sine);
                                    model_floor=std::max(model_floor,4*kPi*
                                        std::abs(witness[k]-evaluate(witness_x[k],sine,false).value));
                                }
                            };
                            extend_witnesses(2);
                            const auto integral=[&](int stride) {
                                double value=0;
                                for(int j=stride;j<divisor;j+=stride) value+=samples[j].square*samples[j].value;
                                return value*kPi*stride/divisor;
                            };
                            double flux=integral(1);
                            struct WitnessProjection {std::array<double,3> value{},slope{};double inverse_remainder=0;};
                            static const auto initial_projections=[] {
                                std::array<WitnessProjection,4> result{};
                                for(int k=0;k<4;++k) {
                                    const double x=witness_x[k];
                                    std::array<double,3> terms{};double denominator=0;
                                    for(int j=1;j<4;++j) {
                                        const double square=sn[2*j-1]*sn[2*j-1];
                                        terms[j-1]=(j%2 ? -1.0 : 1.0)*square/(x-cs[2*j-1]);
                                        denominator+=terms[j-1];
                                    }
                                    for(int j=1;j<4;++j) {
                                        const double node=cs[2*j-1],square=sn[2*j-1]*sn[2*j-1];
                                        const double dx=x-node,basis=terms[j-1]/denominator;
                                        result[k].value[j-1]=basis*basis*(1-3*dx*node/square);
                                        result[k].slope[j-1]=basis*basis*dx;
                                    }
                                    const double u=8*x*x*x-4*x;result[k].inverse_remainder=1/(u*u);
                                }
                                return result;
                            }();
                            const auto indicator=[&]() {
                                if(divisor==2) {
                                    // Symmetric witnesses cancel the linear
                                    // Hermite term exactly. Its derivative is
                                    // only needed if the rule grows to 3 nodes.
                                    const double absolute=.5*kPi*std::abs(samples[1].value);
                                    return 9*kPi*std::abs(.5*(witness[0]+witness[1])-samples[1].value)+
                                        model_floor+400*std::numeric_limits<double>::epsilon()*absolute;
                                }
                                // Gauss U_(n) integrates degree 2n-1. The
                                // values and derivatives at its n nodes form
                                // that Hermite model; two actual witnesses
                                // supply the first even remainder U_n(x)^2.
                                // Orthogonality gives integral pi/2, so this
                                // estimates the error of the returned rule.
                                std::array<double,4> remainders{};
                                double absolute=0;
                                for(int j=1;j<divisor;++j) absolute+=samples[j].square*std::abs(samples[j].value);
                                absolute*=kPi/divisor;
                                for(int k=0;k<witness_count;++k) {
                                    if(divisor==4) {
                                        double prediction=0;
                                        for(int j=1;j<4;++j) prediction+=initial_projections[k].value[j-1]*samples[j].value+
                                            initial_projections[k].slope[j-1]*samples[j].slope;
                                        remainders[k]=(witness[k]-prediction)*initial_projections[k].inverse_remainder;
                                        continue;
                                    }
                                    double denominator=0;
                                    for(int j=1;j<divisor;++j) {
                                        const double node=divisor==4 ? cs[2*j-1] : divisor==8 ? cs[j-1] : std::cos(kPi*j/divisor);
                                        denominator+=(j%2 ? -1.0 : 1.0)*samples[j].square/(witness_x[k]-node);
                                    }
                                    double prediction=0;
                                    for(int j=1;j<divisor;++j) {
                                        const double node=divisor==4 ? cs[2*j-1] : divisor==8 ? cs[j-1] : std::cos(kPi*j/divisor);
                                        const double dx=witness_x[k]-node;
                                        const double basis=(j%2 ? -1.0 : 1.0)*samples[j].square/(dx*denominator);
                                        const double diagonal=3*node/(2*samples[j].square);
                                        prediction+=basis*basis*((1-2*dx*diagonal)*samples[j].value+dx*samples[j].slope);
                                    }
                                    double old=1,u=2*witness_x[k];
                                    for(int n=2;n<divisor;++n) {const double next=2*witness_x[k]*u-old;old=u;u=next;}
                                    remainders[k]=(witness[k]-prediction)/(u*u);
                                }
                                // U_n^2 has weighted second moment 1/2.
                                // Two symmetric pairs recover the even
                                // quadratic remainder, including cancellation
                                // of the leading term at either witness pair.
                                const double quadrature=4*kPi*std::abs((9*.5*(remainders[0]+remainders[1])+
                                    25*.5*(remainders[2]+remainders[3]))/34);
                                return quadrature+model_floor+
                                    400*std::numeric_limits<double>::epsilon()*absolute;
                            };
                            double error=indicator();
                            const double local_target=radial_allocation*std::max(target/(2*kPi)*.02,
                                std::abs(result.flux+flux)*radial_relative_target()*.08);
                            while(error>local_target) {
                                // Refining the polynomial rule cannot remove
                                // disagreement with actual mapped samples.
                                if(model_floor>local_target) {++work.radial_limit_hits;break;}
                                if(depth>=work.max_radial_depth) {++work.radial_limit_hits;break;}
                                if(divisor==2)samples[1]=evaluate(0,1);
                                const int next_divisor=2*divisor;
                                if(next_divisor<=8) {
                                    // Move reused nodes backwards before
                                    // filling odd positions in the stack buffer.
                                    for(int j=divisor-1;j>=1;--j)initial[2*j]=samples[j];
                                    for(int j=1;j<next_divisor;j+=2) {
                                        const int index=next_divisor==4 ? 2*j-1 : j-1;
                                        initial[j]=evaluate(cs[index],sn[index]);
                                    }
                                    samples=initial.data();
                                } else {
                                    std::vector<WeightedSample> next(next_divisor+1);
                                    for(int j=1;j<next_divisor;++j) {
                                        if(j%2==0)next[j]=samples[j/2];
                                        else {
                                            const double theta=kPi*j/next_divisor;
                                            next[j]=evaluate(std::cos(theta),std::sin(theta));
                                        }
                                    }
                                    expanded=std::move(next);samples=expanded.data();
                                }
                                divisor=next_divisor;
                                extend_witnesses(4);
                                flux=integral(1);error=indicator();++depth;++work.radial_refinements;
                                work.largest_radial_depth=std::max(work.largest_radial_depth,depth);
                            }
                            result.flux+=flux;result.error+=error;
                        } else {
                        const auto integrand = [&](double t) {
                            if (t == 0.0 || t == 0.5 * kPi) return 0.0;
                            const double st = std::sin(t);
                            const double radius = start + width * st * st;
                            const double mu = std::sqrt(std::max(0.0, -residual(radius) / rho2));
                            return settings.limb_darkening_c * radius * mu * width * std::sin(2.0 * t);
                        };
                        const auto make_simpson=[&](double a,double b,double fa,double fm,double fb,int depth) {
                            const double h=b-a,q1=integrand(a+.25*h),q3=integrand(a+.75*h);
                            const double coarse=h*(fa+4.0*fm+fb)/6.0;
                            const double fine=h*(fa+4.0*q1+2.0*fm+4.0*q3+fb)/12.0;
                            const double defect=(fine-coarse)/15.0;
                            const std::array<double,5> values{fa,q1,fm,q3,fb};
                            static constexpr double weights[5]={1,-4,6,-4,1};
                            double witness_error=0;
                            for(double fraction:{1.0/3,2.0/3}) {
                                double numerator=0,denominator=0;
                                for(int j=0;j<5;++j) {
                                    const double weight=weights[j]/(fraction-.25*j);
                                    numerator+=weight*values[j];denominator+=weight;
                                }
                                const double mismatch=std::abs(integrand(a+fraction*h)-numerator/denominator);
                                witness_error=std::isfinite(mismatch) ? std::max(witness_error,h*mismatch)
                                    : std::numeric_limits<double>::infinity();
                            }
                            const double rounding=50*std::numeric_limits<double>::epsilon()*h*
                                std::max({std::abs(fa),std::abs(q1),std::abs(fm),std::abs(q3),std::abs(fb)});
                            return RadialPanel{a,b,fa,fm,fb,q1,q3,fine+defect,
                                8.0*std::max({std::abs(defect),witness_error,rounding}),depth};
                        };
                        std::vector<RadialPanel> panels;
                        auto first=make_simpson(0.0,.5*kPi,0.0,integrand(.25*kPi),0.0,0);
                        panels.push_back(first);
                        double flux = first.flux, error = first.error;
                            const double local_target = radial_allocation*std::max(target / (2.0 * kPi) * 0.02,
                            std::abs(result.flux + flux) * radial_relative_target() * 0.08);
                        for (;;) {
                            if (error <= local_target) break;
                            auto it = std::max_element(panels.begin(), panels.end(), [&](const auto& a, const auto& b) {
                                return (a.depth < work.max_radial_depth ? a.error : -1.0) < (b.depth < work.max_radial_depth ? b.error : -1.0);
                            });
                            if (it->depth >= work.max_radial_depth) { ++work.radial_limit_hits; break; }
                            const auto p = *it; const double m = 0.5 * (p.a + p.b);
                            auto left=make_simpson(p.a,m,p.fa,p.fq1,p.fm,p.depth+1);
                            auto right=make_simpson(m,p.b,p.fm,p.fq3,p.fb,p.depth+1);
                            flux += left.flux + right.flux - p.flux;
                            error = std::max(0.0, error + left.error + right.error - p.error);
                            *it = left; panels.push_back(right); ++work.radial_refinements;
                            work.largest_radial_depth = std::max(work.largest_radial_depth, p.depth + 1);
                        }
                        result.flux += flux; result.error += error;
                        }
                    }
                    start = -1.0;
                }
            }
            previous = next; fprevious = fnext;
        }
        ++work.columns;
        result.stationary=extrema;
        result.boundaries=current_boundaries;
        previous_boundaries=std::move(current_boundaries);
        previous_phi=phi;
        const auto& stored=columns.emplace(key,result).first->second;
        angular_columns.emplace(phi,&stored);
        return stored;
    };
    const auto column = [&](std::int64_t key) -> const Column& {
        key%=circle;if (key<0) key+=circle;
        return column_at((static_cast<double>(key)/ticks)*panel_dphi);
    };
    const auto ensure_critical=[&](const Column& col) {
        if(col.critical_ready) return;
        ++work.lazy_stationary_solves;
        const double ux=std::cos(col.phi),uy=std::sin(col.phi);
        const auto polynomial=limb_polynomial.coefficients(ux,uy);
        std::array<long double,7> derivative{};
        for(int j=1;j<=6;++j) derivative[j-1]=j*polynomial[j];
        PolarRealRoots actual;
        if(polar_continue_stationary_roots(derivative,upper_radius,col.stationary,actual)) ++work.stationary_continuations;
        else {
            actual.count=0;polar_real_polynomial_roots(derivative,5,0.0L,upper_radius,actual);
            ++work.stationary_full_solves;
        }
        col.stationary=actual;col.critical_count=0;
        for(long double wide_r:actual) {
            const double r=static_cast<double>(wide_r);
            const auto mapped=map_lens_real(counted,r*ux,r*uy);
            const auto j=col.critical_count++;
            col.critical_r[j]=r;col.critical_f[j]=distance_squared(mapped,source)-rho2;
        }
        col.critical_ready=true;
    };
    const auto make_cap = [&](const AdaptivePolarPanel& base,std::int64_t a,std::int64_t b,int depth) {
        const double lo=static_cast<double>(a)/ticks*panel_dphi;
        const double hi=static_cast<double>(b)/ticks*panel_dphi,h=hi-lo;
        const bool born=base.right_runs>base.left_runs;
        const int minimum_runs=std::min(base.left_runs,base.right_runs);
        const auto distance=[&](double phi) { return std::max(0.0,born ? phi-base.tip : base.tip-phi); };
        const auto model=[&](double phi) {
            const double u=distance(phi);
            return base.uniform_fold*std::sqrt(u)+base.dark_fold*u;
        };
        const auto primitive=[&](double u) {
            return 2.0/3.0*base.uniform_fold*std::pow(u,1.5)+.5*base.dark_fold*u*u;
        };
        const std::array<std::int64_t,5> keys{a,a+(b-a)/4,(a+b)/2,a+3*(b-a)/4,b};
        std::array<double,5> residual{},radial{};
        std::array<int,5> runs{};
        bool incompatible=false;
        for (int j=0;j<5;++j) {
            const double phi=lo+h*j/4;
            const auto& col=column(keys[j]);
            residual[j]=col.flux-model(phi);radial[j]=col.error;runs[j]=col.runs;
            const int expected=minimum_runs+(distance(phi)>0 ? 1 : 0);
            incompatible |= col.runs<minimum_runs || col.runs>minimum_runs+1 || (col.runs!=expected && std::abs(phi-base.tip)>base.contact_bracket);
        }
        const double coarse=h*(residual[0]+4*residual[2]+residual[4])/6;
        const double fine=h*(residual[0]+4*residual[1]+2*residual[2]+4*residual[3]+residual[4])/12;
        const double defect=(fine-coarse)/15;
        double error=64*std::abs(defect)+base.tip_error*h/(base.width-base.left_width);
        // The known single fold has already been integrated analytically.
        // A run-count change here is expected and does not invalidate the
        // residual quadrature. Its range includes smooth background slopes
        // that Simpson integrates exactly; h*range is not their integration
        // error. Retain the defect, contact-position uncertainty, independent
        // discovery witnesses, and parent/child change checks instead.
        for (const auto& seed:seed_angles)
            if (seed[0]>=lo && seed[0]<=hi) (void)column_at(seed[0]);
        const auto footprint=witnesses.find(static_cast<int>(base.a/ticks));
        if (footprint!=witnesses.end()) for (const auto& witness:footprint->second.representatives) {
        if (witness.key>=a && witness.key<=b) {
            const double u=4.0*static_cast<double>(witness.key-a)/(b-a);
            const double predicted=polar_interpolate_equispaced(residual,u);
            const auto& col=column(witness.key);
            const double phi=static_cast<double>(witness.key)/ticks*panel_dphi;
            const int expected=minimum_runs+(distance(phi)>0 ? 1 : 0);
            incompatible |= col.runs<minimum_runs || col.runs>minimum_runs+1 ||
                (col.runs!=expected && std::abs(phi-base.tip)>base.contact_bracket);
            error=std::max(error,4*h*std::abs(col.flux-model(phi)-predicted));
            ++work.coverage_witness_checks;
        }
        }
        for(auto it=angular_columns.lower_bound(lo);it!=angular_columns.end() && it->first<=hi;++it) {
            const double phi=it->first,u=4*(phi-lo)/h;
            const double predicted=polar_interpolate_equispaced(residual,u);
            const auto& col=*it->second;
            const int expected=minimum_runs+(distance(phi)>0 ? 1 : 0);
            incompatible |= col.runs<minimum_runs || col.runs>minimum_runs+1 ||
                (col.runs!=expected && std::abs(phi-base.tip)>base.contact_bracket);
            error=std::max(error,4*h*std::abs(col.flux-model(phi)-predicted));
        }
        auto result=base;result.a=a;result.b=b;result.depth=depth;result.pieces.reset();
        result.flux=fine+defect+std::abs(primitive(distance(hi))-primitive(distance(lo)));
        incompatible |= !(result.flux>=0.0) || !std::isfinite(result.flux);
        result.angular_error=result.quadrature_error=error;
        result.radial_error=1.2*h*(radial[0]+4*radial[1]+2*radial[2]+4*radial[3]+radial[4])/12;
        result.error_kind=4;result.incompatible=incompatible;
        return result;
    };
    const auto make_single_cap = [&](const AdaptivePolarPanel& base,double fractionlo,double fractionhi,int depth) {
        const double tlo=base.t_scale*fractionlo,thi=base.t_scale*fractionhi;
        const bool born=base.right_runs>base.left_runs;
        const double dt=thi-tlo,h=thi*thi-tlo*tlo;
        const double phi0=base.tip+(born ? tlo*tlo : -tlo*tlo);
        const double phi1=base.tip+(born ? thi*thi : -thi*thi);
        const double lo=std::min(phi0,phi1),hi=std::max(phi0,phi1);
        std::array<double,9> g{},radial{};bool incompatible=false;
        for(int j=0;j<9;++j) {
            const double t=base.t_scale*(fractionlo+(fractionhi-fractionlo)*j/8);
            if(t==0) continue;
            const double phi=base.tip+(born ? t*t : -t*t);
            const auto& col=column_at(phi);
            incompatible |= col.runs>1 || (col.runs!=1 && t*t>base.contact_bracket);
            g[j]=2*t*col.flux;radial[j]=2*t*col.error;
        }
        std::array<double,13> physical{},physical_radial{};
        for(int j=0;j<9;++j) {
            physical[PolarEmbeddedNineRule::index[j]]=g[j];
            physical_radial[PolarEmbeddedNineRule::index[j]]=radial[j];
        }
        for(int j:{1,4,8,11}) {
            const double t=tlo+dt*PolarEmbeddedNineRule::x[j];
            const auto& col=column_at(base.tip+(born ? t*t : -t*t));
            incompatible|=col.runs>1 || (col.runs!=1 && t*t>base.contact_bracket);
            physical[j]=2*t*col.flux;physical_radial[j]=2*t*col.error;
        }
        const auto embedded=PolarEmbeddedNineRule::integrate(physical,dt);
        double error=embedded[1]+embedded[2]+base.tip_error*h/(base.t_scale*base.t_scale);
        const auto check_location=[&](double phi,const Column& col) {
            const double t=std::sqrt(std::abs(phi-base.tip));
            incompatible |= col.runs>1 || (col.runs!=1 && t*t>base.contact_bracket);
            if(t>0 && dt>0) {
                const double u=(t-tlo)/dt;
                const double predicted=PolarEmbeddedNineRule::interpolate(physical,u);
                error=std::max(error,4*dt*std::abs(2*t*col.flux-predicted));
            }
            ++work.coverage_witness_checks;
        };
        // A known centre image must agree with the retained quadrature.
        // Once sampled, its proposal width is not an integration error.
        for(const auto& seed:seed_angles) if(seed[0]>=lo && seed[0]<=hi)
            check_location(seed[0],column_at(seed[0]));
        const auto footprint=witnesses.find(static_cast<int>(base.a/ticks));
        if(footprint!=witnesses.end()) for(const auto& witness:footprint->second.representatives) {
            const double phi=static_cast<double>(witness.key)/ticks*panel_dphi;
            if(phi>=lo && phi<=hi) check_location(phi,column(witness.key));
        }
        for(auto it=angular_columns.lower_bound(lo);it!=angular_columns.end() && it->first<=hi;++it)
            check_location(it->first,*it->second);
        auto result=base;result.transformed=true;result.t_lo=fractionlo;result.t_hi=fractionhi;result.depth=depth;result.pieces.reset();
        result.flux=embedded[0];result.angular_error=result.quadrature_error=error;
        result.radial_error=0;
        for(int j=0;j<9;++j) result.radial_error+=1.2*dt*std::abs(PolarEmbeddedNineRule::low[j])*
            physical_radial[PolarEmbeddedNineRule::index[j]];
        incompatible|=!(result.flux>=0) || !std::isfinite(result.flux);
        result.error_kind=4;result.incompatible=incompatible;return result;
    };
    const auto try_contact = [&](const AdaptivePolarPanel& p,AdaptivePolarPanel& result) {
        if (p.cap || p.contact_disabled || p.error_kind!=1 || p.b-p.a<8) return false;
        ++work.contact_attempts;
        std::array<const Column*,5> nodes{&column(p.a),&column(p.a+(p.b-p.a)/4),
            &column((p.a+p.b)/2),&column(p.a+3*(p.b-p.a)/4),&column(p.b)};
        const int first=nodes.front()->runs,last=nodes.back()->runs;
        // A single birth/death supplies one square-root endpoint term.
        // Existing radial images stay in the directly sampled residual.
        if (std::abs(first-last)!=1) return false;
        int transitions=0,edge=0;
        for (int j=1;j<5;++j) {
            if (nodes[j]->runs!=first && nodes[j]->runs!=last) return false;
            if (nodes[j]->runs!=nodes[j-1]->runs) { ++transitions;edge=j; }
        }
        if (transitions!=1) return false;
        const auto& low=*nodes[edge-1];const auto& high=*nodes[edge];
        ensure_critical(low);ensure_critical(high);
        int crossings=0;double radius=0,flo=0,fhi=0;
        for (std::size_t j=0;j<low.critical_count;++j) {
            std::size_t nearest=0;double distance=std::numeric_limits<double>::infinity();
            for (std::size_t k=0;k<high.critical_count;++k)
                if (std::abs(high.critical_r[k]-low.critical_r[j])<distance) {
                    distance=std::abs(high.critical_r[k]-low.critical_r[j]);nearest=k;
                }
            if (high.critical_count && (low.critical_f[j]<=0)!=(high.critical_f[nearest]<=0)) {
                ++crossings;radius=.5*(low.critical_r[j]+high.critical_r[nearest]);
                flo=low.critical_f[j];fhi=high.critical_f[nearest];
            }
        }
        if (crossings!=1 || !std::isfinite(flo) || !std::isfinite(fhi)) return false;
        const double phi0=static_cast<double>(p.a)/ticks*panel_dphi;
        const double h=static_cast<double>(p.b-p.a)/ticks*panel_dphi;
        double lo=phi0+h*(edge-1)/4,hi=phi0+h*edge/4,maximum=0;
        for (const auto node:nodes) maximum=std::max(maximum,std::abs(node->flux));
        if (!(maximum>0.0)) return false;
        PolarRealRoots old;
        for (std::size_t j=0;j<low.critical_count;++j) old.push_back(low.critical_r[j]);
        int last_side=0;
        for (int iteration=0;iteration<48;++iteration) {
            if (4*(hi-lo)*maximum<=.02*target) break;
            const double fraction=std::clamp(-flo/(fhi-flo),.05,.95);
            const double phi=lo+(hi-lo)*fraction;
            if (!(phi>lo && phi<hi)) return false;
            const double ux=std::cos(phi),uy=std::sin(phi);
            const auto coefficients=limb_polynomial.coefficients(ux,uy);
            std::array<long double,7> derivative{};
            for (int j=1;j<=6;++j) derivative[j-1]=j*coefficients[j];
            PolarRealRoots extrema;
            if (!polar_continue_stationary_roots(derivative,upper_radius,old,extrema)) {
                extrema.count=0;polar_real_polynomial_roots(derivative,5,0,upper_radius,extrema);
            }
            old=extrema;++work.contact_geometry_columns;
            int runs=0;bool inside=false;double critical_value=0,distance=std::numeric_limits<double>::infinity(),next_radius=radius;
            for (long double wide_r:extrema) {
                const double r=static_cast<double>(wide_r);
                const auto mapped=map_lens_real(counted,r*ux,r*uy);
                const double f=distance_squared(mapped,source)-rho2;
                if (!std::isfinite(f)) return false;
                if (!inside && f<=0) ++runs;
                inside=f<=0;
                if (std::abs(r-radius)<distance) { distance=std::abs(r-radius);critical_value=f;next_radius=r; }
            }
            if (distance==std::numeric_limits<double>::infinity() || (runs!=first && runs!=last)) return false;
            radius=next_radius;
            if (runs==first) { lo=phi;flo=critical_value;if (last_side==-1) fhi*=.5;last_side=-1; }
            else { hi=phi;fhi=critical_value;if (last_side==1) flo*=.5;last_side=1; }
        }
        if (4*(hi-lo)*maximum>.02*target) return false;
        auto base=p;base.cap=true;base.tip=.5*(lo+hi);base.left_width=phi0-base.tip;base.width=phi0+h-base.tip;
        base.left_runs=first;base.right_runs=last;base.tip_error=4*(hi-lo)*maximum;
        // F(r,phi)=0 and F_r=0 at an ordinary angular contact. Locally
        // F = -A*u + B*(r-r_tip)^2/2, with u the inward angular distance.
        // The uniform flux is 2*r_tip*sqrt(2*A/B)*sqrt(u); the linear-LD
        // contribution is pi*r_tip*A/(rho*sqrt(2*B))*u. Integrating these
        // known endpoint terms separates the old rule's cusp bias from an
        // actual disagreement with the transformed quadrature.
        const double ux=std::cos(base.tip),uy=std::sin(base.tip);
        const auto mapped=map_lens_real(counted,radius*ux,radius*uy);
        double jxx=1,jxy=0,ddx=0,ddy=0;
        for (int lens=0;lens<2;++lens) {
            const double mass=lens==0 ? mapper.m1 : mapper.m2;
            const double x=radius*ux-(lens==0 ? mapper.separation.real() : 0.0),y=radius*uy;
            const double den=x*x+y*y,den2=den*den,den3=den2*den,dden=2*(x*ux+y*uy);
            jxx+=mass*(x*x-y*y)/den2;jxy+=2*mass*x*y/den2;
            ddx+=mass*(2*ux*dden+2*x)/den2-2*mass*x*dden*dden/den3;
            ddy+=mass*(2*uy*dden+2*y)/den2-2*mass*y*dden*dden/den3;
        }
        const double dx=jxx*ux+jxy*uy,dy=jxy*ux+(2-jxx)*uy;
        const double tx=radius*(-jxx*uy+jxy*ux),ty=radius*(-jxy*uy+(2-jxx)*ux);
        const double sx=mapped.x-source.x,sy=mapped.y-source.y;
        const double A=std::abs(2*(sx*tx+sy*ty)),B=2*(dx*dx+dy*dy+sx*ddx+sy*ddy);
        if (!(A>0 && B>0) || !std::isfinite(A) || !std::isfinite(B)) return false;
        const double uniform=(1-settings.limb_darkening_c)*2*radius*std::sqrt(2*A/B);
        const double dark=settings.limb_darkening_c*kPi*radius*A/(rho*std::sqrt(2*B));
        const double extent=last>first ? base.width : -base.left_width;
        const double exact=2.0/3.0*uniform*std::pow(extent,1.5)+.5*dark*extent*extent;
        static constexpr double weights[5]={7.0/90,32.0/90,12.0/90,32.0/90,7.0/90};
        double sampled=0;
        for (int j=0;j<5;++j) {
            const double phi=phi0+h*j/4;
            const double u=std::max(0.0,last>first ? phi-base.tip : base.tip-phi);
            sampled+=h*weights[j]*(uniform*std::sqrt(u)+dark*u);
        }
        if (!std::isfinite(exact) || !std::isfinite(sampled)) return false;
        base.uniform_fold=uniform;base.dark_fold=dark;base.contact_bracket=hi-lo;
        if(std::min(first,last)==0) {
            bool outside_clear=true;
            const auto outside=[&](double phi) {return last>first ? phi<base.tip : phi>base.tip;};
            for(const auto& seed:seed_angles) if(seed[0]>=phi0 && seed[0]<=phi0+h && outside(seed[0]))
                outside_clear &= column_at(seed[0]).runs==0;
            for(auto it=angular_columns.lower_bound(phi0);it!=angular_columns.end() && it->first<=phi0+h;++it)
                if(outside(it->first)) outside_clear &= it->second->runs==0;
            const auto footprint=witnesses.find(static_cast<int>(p.a/ticks));
            if(footprint!=witnesses.end()) for(const auto& witness:footprint->second.representatives)
                if(witness.key>=p.a && witness.key<=p.b) {
                    const double phi=static_cast<double>(witness.key)/ticks*panel_dphi;
                    if(outside(phi)) outside_clear &= column(witness.key).runs==0;
                }
            if(outside_clear) {base.t_scale=std::sqrt(extent);result=make_single_cap(base,0,1,p.depth);}
            else result.incompatible=true;
            if(result.incompatible) result=make_cap(base,p.a,p.b,p.depth);
        } else result=make_cap(base,p.a,p.b,p.depth);
        if (result.incompatible) return false;
        // Retain the old physical-node comparison for residual caps.
        // Isolated transformed caps have their own embedded integral and
        // independent centre/discovery probes; the discarded physical rule
        // is not an estimate of their quadrature error.
        result.reference_flux=p.flux+exact-sampled;
        if(!result.transformed) result.angular_error=std::max(result.angular_error,4.0*std::abs(result.flux-result.reference_flux));
        result.base_depth=p.depth;
        result.pieces=std::make_shared<std::vector<AdaptivePolarPanel>>();
        auto first_piece=result;first_piece.pieces.reset();result.pieces->push_back(first_piece);
        ++work.angular_cap_panels;
        return true;
    };
    const auto make_panel = [&](std::int64_t a, std::int64_t b, int depth) {
        const auto& fa = column(a);const auto& fm = column((a + b) / 2);const auto& fb = column(b);
        const auto& q1 = column(a + (b - a) / 4);const auto& q3 = column(a + 3 * (b - a) / 4);
        const double h = static_cast<double>(b - a) / ticks * panel_dphi;
        const double coarse = h * (fa.flux + 4.0 * fm.flux + fb.flux) / 6.0;
        const double fine = h * (fa.flux + 4.0 * q1.flux + 2.0 * fm.flux + 4.0 * q3.flux + fb.flux) / 12.0;
        const double defect = (fine - coarse) / 15.0;
        const double radial = h * (fa.error + 4.0 * q1.error + 2.0 * fm.error + 4.0 * q3.error + fb.error) / 12.0;
        double angular_error = 64.0 * std::abs(defect);
        double coverage_guard=0.0;
        int error_kind=0;
        if (fa.runs != fm.runs || fm.runs != fb.runs || q1.runs != fm.runs || q3.runs != fm.runs) {
            error_kind=1;
            // A branch birth/death invalidates smooth Simpson cancellation.
            // Bound the sampled variation, including the emerging branch,
            // without charging the unchanged flux of overlapping images.
            // At an isolated angular cap the minimum is zero, so this guard
            // is ten times larger than the previous 0.1*h*maximum guard.
            const auto extrema = std::minmax({fa.flux, fm.flux, fb.flux, q1.flux, q3.flux});
            angular_error = std::max(angular_error, h * (extrema.second - extrema.first));
        }
        for (const auto& seed : seed_angles) {
            // A rounded tick near 2*pi can wrap to zero. Compare the
            // physical angle so its interpolation coordinate stays within
            // this panel even when the inverse-Jacobian width is large.
            const double seed_lo=static_cast<double>(a)/ticks*panel_dphi;
            if (seed[0]>=seed_lo && seed[0]<=seed_lo+h) {
                const auto& observed=column_at(seed[0]);
                const int minimum=std::min({fa.runs,q1.runs,fm.runs,q3.runs,fb.runs});
                const int maximum=std::max({fa.runs,q1.runs,fm.runs,q3.runs,fb.runs});
                if(minimum>0 && minimum==maximum && observed.runs==minimum) {
                    // Check the centre independently. The geometric width
                    // only proposed this probe; after detection, use its
                    // quadrature discrepancy instead of its whole flux.
                    const double u=4*(seed[0]-static_cast<double>(a)/ticks*panel_dphi)/h;
                    const std::array<double,5> f{fa.flux,q1.flux,fm.flux,q3.flux,fb.flux};
                    const double predicted=polar_interpolate_equispaced(f,u);
                    const double discrepancy=4*h*std::abs(observed.flux-predicted);
                    coverage_guard=std::max(coverage_guard,discrepancy);
                    if(discrepancy>angular_error) {angular_error=discrepancy;error_kind=3;}
                    ++work.coverage_witness_checks;
                } else {
                    error_kind=2;
                    angular_error=std::max(angular_error,h*std::abs(observed.flux));
                }
            }
        }
        // Preserve a positive column from discovery through every bisection.
        // Compare with the quartic through the existing five nodes. This is
        // an independent location, not another dyadic convergence test.
        const auto footprint=witnesses.find(static_cast<int>(a/ticks));
        if (footprint!=witnesses.end()) for (const auto& witness:footprint->second.representatives) {
        if (witness.key>=a && witness.key<=b) {
            const double u=4.0*static_cast<double>(witness.key-a)/(b-a);
            const std::array<double,5> nodes{fa.flux,q1.flux,fm.flux,q3.flux,fb.flux};
            const double predicted=polar_interpolate_equispaced(nodes,u);
            const auto& observed=column(witness.key);
            ++work.coverage_witness_checks;
            double discrepancy=4.0*h*std::abs(observed.flux-predicted);
            const int minimum=std::min({fa.runs,q1.runs,fm.runs,q3.runs,fb.runs});
            const int maximum=std::max({fa.runs,q1.runs,fm.runs,q3.runs,fb.runs});
            if (observed.runs<minimum || observed.runs>maximum) {
                // A discovered branch or hole must reach quadrature nodes.
                discrepancy=std::max(discrepancy,h*std::abs(observed.flux));
                ++work.coverage_branch_guards;
            }
            coverage_guard=std::max(coverage_guard,discrepancy);
            if (discrepancy>angular_error) { angular_error=discrepancy;error_kind=3; }
        }
        }
        // Every earlier physical observation remains a witness after a split.
        // In particular, non-dyadic validation points must not disappear
        // merely because they are absent from the child Boole stencil.
        const double lo=static_cast<double>(a)/ticks*panel_dphi,hi=static_cast<double>(b)/ticks*panel_dphi;
        for(auto it=angular_columns.lower_bound(lo);it!=angular_columns.end() && it->first<=hi;++it) {
            const auto& observed=*it->second;
            if(&observed==&fa || &observed==&q1 || &observed==&fm || &observed==&q3 || &observed==&fb) continue;
            const double u=4*(it->first-lo)/h;
            const std::array<double,5> nodes{fa.flux,q1.flux,fm.flux,q3.flux,fb.flux};
            const double predicted=polar_interpolate_equispaced(nodes,u);
            angular_error=std::max(angular_error,4*h*std::abs(observed.flux-predicted));
            if(observed.runs<std::min({fa.runs,q1.runs,fm.runs,q3.runs,fb.runs}) ||
               observed.runs>std::max({fa.runs,q1.runs,fm.runs,q3.runs,fb.runs})) {
                error_kind=1;
                angular_error=std::max(angular_error,h*std::max({fa.flux,q1.flux,fm.flux,q3.flux,fb.flux,observed.flux}));
            }
        }
        auto result=AdaptivePolarPanel{a, b, depth, fine + defect, angular_error, 1.2 * radial, error_kind};
        result.coverage_guard=coverage_guard;
        result.legacy_quadrature_error=64*std::abs(defect);
        return result;
    };
    struct LargerError { bool operator()(const AdaptivePolarPanel& a, const AdaptivePolarPanel& b) const { return a.angular_error < b.angular_error; } };
    std::priority_queue<AdaptivePolarPanel, std::vector<AdaptivePolarPanel>, LargerError> pending;
    double flux = 0.0, angular_error = 0.0, radial_error = 0.0;
    std::array<double,5> angular_by_kind{};
    bool initialized = false;
    double frozen_error = 0.0;
    const auto requested_target = [&]() {
        return work.relative_tolerance > 0.0
            ? work.relative_tolerance * (flux > 0.0 ? flux
                : norm * std::max(1.0, std::abs(coarse_magnification)))
            : norm * work.absolute_tolerance;
    };
    try {
        std::deque<int> discovery(active.begin(), active.end());
        while (!discovery.empty()) {
            const int i = discovery.front(); discovery.pop_front();
            const auto a = i * ticks;
            const auto b = (i + 1LL) * ticks;
            auto p = make_panel(a, b, 0);
            flux += p.flux; angular_error += p.angular_error; radial_error += p.radial_error;
            angular_by_kind[p.error_kind]+=p.angular_error;
            pending.push(p);
            for (int side : {-1, 1}) {
                if (column(side < 0 ? a : b).flux <= 0.0) continue;
                const int next = static_cast<int>((static_cast<std::int64_t>(i) + side + groups) % groups);
                if (active.insert(next).second) discovery.push_back(next);
            }
        }
        initialized = true;
        if (!std::isfinite(flux)) {
            work.stop_reason = "unresolved_radial_support";
            work.angular_error = work.radial_error = work.error_indicator =
                std::numeric_limits<double>::infinity();
            return coarse_magnification;
        }
        if (work.relative_tolerance > 0.0 && flux > 0.0) {
            work.initial_magnification = flux / norm;
            target = work.relative_tolerance * std::abs(flux);
        }
        // A known narrow image can lie between every initial Simpson node.
        // Seed columns still carry a positive refinement guard. Refine them
        // until quadrature sees the image instead of returning the coarse
        // cut-cell answer merely because the initial integral is zero.
        refinement_work.initial(double(work.maps),angular_error+radial_error,flux);
        while ((flux <= 0.0 || angular_error + radial_error > requested_target()) && !pending.empty()) {
            adaptive_detail::RefinementWorkTrial refinement_trial(refinement_work,[&]() {
                return std::array<double,3>{double(work.maps),angular_error+radial_error,flux};
            });
            refinement_trial.kind=1;
            target = requested_target();
            if (work.radial_limit_hits && radial_error >= target && angular_error < 0.1 * target) {
                work.stop_reason = "radial_resolution_limit";
                break;
            }
            const auto p = pending.top(); pending.pop();
            if (p.cap) {
                refinement_trial.kind=4;
                // Keep a transformed contact as one physical angular tile.
                // Its inner adaptive pieces can then be replaced atomically
                // by the original rule if convergence stalls. Replacing a
                // single child by the whole tile would double-count flux.
                auto replacement=p;
                auto& pieces=*p.pieces;
                auto it=std::max_element(pieces.begin(),pieces.end(),[](const auto& a,const auto& b) {
                    return a.angular_error<b.angular_error;
                });
                const auto piece=*it;
                bool fallback=pieces.size()>=32 || piece.depth>=work.max_angular_depth;
                if (!fallback) {
                    const auto midpoint=(piece.a+piece.b)/2;
                    const double tm=.5*(piece.t_lo+piece.t_hi);
                    const auto left=piece.transformed ? make_single_cap(piece,piece.t_lo,tm,piece.depth+1) : make_cap(piece,piece.a,midpoint,piece.depth+1);
                    const auto right=piece.transformed ? make_single_cap(piece,tm,piece.t_hi,piece.depth+1) : make_cap(piece,midpoint,piece.b,piece.depth+1);
                    fallback=left.incompatible || right.incompatible ||
                        (piece.depth-p.base_depth>=3 && left.angular_error+right.angular_error>.8*piece.angular_error);
                    if (!fallback) {
                        replacement.flux+=left.flux+right.flux-piece.flux;
                        replacement.quadrature_error=std::max(0.0,replacement.quadrature_error+left.angular_error+right.angular_error-piece.angular_error);
                        replacement.angular_error=std::max(replacement.quadrature_error,4.0*std::abs(replacement.flux-p.flux));
                        replacement.radial_error=std::max(0.0,replacement.radial_error+left.radial_error+right.radial_error-piece.radial_error);
                        replacement.depth=std::max(replacement.depth,piece.depth+1);
                        *it=left;pieces.push_back(right);
                        ++work.angular_refinements;
                        work.largest_angular_depth=std::max(work.largest_angular_depth,piece.depth+1);
                        fallback=replacement.quadrature_error<.25*target && replacement.angular_error>.25*target;
                    }
                }
                if (fallback) {
                    replacement=make_panel(p.a,p.b,p.base_depth);
                    replacement.contact_disabled=true;
                    ++work.contact_fallbacks;
                }
                flux+=replacement.flux-p.flux;
                angular_error=std::max(0.0,angular_error+replacement.angular_error-p.angular_error);
                radial_error=std::max(0.0,radial_error+replacement.radial_error-p.radial_error);
                angular_by_kind[p.error_kind]-=p.angular_error;
                angular_by_kind[replacement.error_kind]+=replacement.angular_error;
                pending.push(replacement);continue;
            }
            AdaptivePolarPanel contact;
            if (try_contact(p,contact)) {
                refinement_trial.kind=3;
                flux+=contact.flux-p.flux;
                angular_error=std::max(0.0,angular_error+contact.angular_error-p.angular_error);
                radial_error=std::max(0.0,radial_error+contact.radial_error-p.radial_error);
                angular_by_kind[p.error_kind]-=p.angular_error;angular_by_kind[contact.error_kind]+=contact.angular_error;
                pending.push(contact);continue;
            }
            if (p.depth >= work.max_angular_depth) {
                refinement_trial.kind=5;
                ++work.angular_limit_hits;
                frozen_error += p.angular_error + p.radial_error;
                // Further subdivision elsewhere cannot reduce these capped
                // panels. Keep their flux/error in the totals and stop before
                // spending the entire map budget on unrelated smooth panels.
                if (frozen_error >= target) break;
                continue;
            }
            if(!p.high_rule) {
                const double width=static_cast<double>(p.b-p.a)/ticks*panel_dphi;
                const double lo=static_cast<double>(p.a)/ticks*panel_dphi;
                std::array<double,9> values{},uncertainty{};
                int runs=-1;bool compatible=true;
                for(int j=0;j<9;++j) {
                    const auto& col=column(p.a+j*(p.b-p.a)/8);
                    if(runs<0)runs=col.runs;
                    if(runs==0 || col.runs!=runs)compatible=false;
                    values[j]=col.flux;uncertainty[j]=col.error;
                }
                if(compatible) {
                    double guard=0;
                    for(auto it=angular_columns.lower_bound(lo);it!=angular_columns.end() && it->first<=lo+width;++it) {
                        compatible&=it->second->runs==runs;
                        guard=std::max(guard,4*width*std::abs(it->second->flux-
                            polar_interpolate_equispaced(values,8*(it->first-lo)/width)));
                    }
                    static constexpr double boole[5]={7.0/90,32.0/90,12.0/90,32.0/90,7.0/90};
                    double lower=0,higher=0,radial=0,absolute=0;
                    for(int j=0;j<5;++j){lower+=boole[j]*values[2*j];radial+=boole[j]*uncertainty[2*j];}
                    for(int j=0;j<9;++j){higher+=PolarEmbeddedNineRule::low[j]*values[j];absolute+=std::abs(PolarEmbeddedNineRule::low[j]*values[j]);}
                    compatible&=lower>=0 && std::isfinite(lower);
                    if(compatible) {
                        auto refined=p;refined.high_rule=true;refined.flux=width*lower;
                        refinement_trial.kind=2;
                        const double quadrature=2*width*std::abs(lower-higher);
                        refined.angular_error=quadrature+guard+256*std::numeric_limits<double>::epsilon()*width*absolute;
                        refined.coverage_guard=guard;refined.radial_error=1.2*width*radial;
                        refined.error_kind=guard>quadrature?3:0;
                        ++work.corrected_error_checks;
                        if(refined.angular_error<p.angular_error)++work.corrected_error_reductions;
                        flux+=refined.flux-p.flux;
                        angular_error=std::max(0.0,angular_error+refined.angular_error-p.angular_error);
                        radial_error=std::max(0.0,radial_error+refined.radial_error-p.radial_error);
                        angular_by_kind[p.error_kind]-=p.angular_error;
                        angular_by_kind[refined.error_kind]+=refined.angular_error;
                        pending.push(refined);continue;
                    }
                }
            }
            const auto m = (p.a + p.b) / 2;
            auto left = make_panel(p.a, m, p.depth + 1);
            auto right = make_panel(m, p.b, p.depth + 1);
            left.contact_disabled=right.contact_disabled=p.contact_disabled;
            flux += left.flux + right.flux - p.flux;
            angular_error = std::max(0.0, angular_error + left.angular_error + right.angular_error - p.angular_error);
            angular_by_kind[p.error_kind]-=p.angular_error;
            angular_by_kind[left.error_kind]+=left.angular_error;
            angular_by_kind[right.error_kind]+=right.angular_error;
            radial_error = std::max(0.0, radial_error + left.radial_error + right.radial_error - p.radial_error);
            pending.push(left); pending.push(right);
            ++work.angular_refinements;
            work.largest_angular_depth = std::max(work.largest_angular_depth, p.depth + 1);
        }
        if (!(flux > 0.0) || !std::isfinite(flux)) {
            work.met = false;
            work.stop_reason = "unresolved_radial_support";
            work.angular_error = work.radial_error = work.error_indicator =
                std::numeric_limits<double>::infinity();
            return coarse_magnification;
        }
        work.met = angular_error + radial_error <= requested_target();
        if (work.stop_reason != "radial_resolution_limit")
            work.stop_reason = work.met ? "estimated_target_met" : "resolution_limit";
    } catch (const AdaptiveFluxBudgetExhausted&) {
        work.stop_reason = "map_budget";
        if (!initialized) {
            work.angular_error = work.radial_error = std::numeric_limits<double>::infinity();
            work.error_indicator = std::numeric_limits<double>::infinity();
            return coarse_magnification;
        }
    }
    work.smooth_angular_error=std::max(0.0,angular_by_kind[0])/norm;
    work.topology_angular_error=std::max(0.0,angular_by_kind[1])/norm;
    work.seed_angular_error=std::max(0.0,angular_by_kind[2])/norm;
    work.coverage_angular_error=std::max(0.0,angular_by_kind[3])/norm;
    work.cap_angular_error=std::max(0.0,angular_by_kind[4])/norm;
    // Use the reported components for final acceptance as well. Repeated
    // panel replacement can make the separate running total drift slightly
    // from the sum of its nonnegative components at very tight tolerances.
    work.angular_error=work.smooth_angular_error+work.topology_angular_error+
        work.seed_angular_error+work.coverage_angular_error+work.cap_angular_error;
    work.radial_error = radial_error / norm;
    work.unrefinable_error = frozen_error / norm;
    work.error_indicator = work.angular_error + work.radial_error;
    if(std::getenv("LCBININT_DIAGNOSTIC_POLAR_PANEL_TRACE")) {
        const auto emit=[&](const AdaptivePolarPanel& p) {
            double lo=static_cast<double>(p.a)/ticks*panel_dphi;
            double hi=static_cast<double>(p.b)/ticks*panel_dphi;
            if(p.transformed) {
                const double sign=p.right_runs>p.left_runs ? 1.0 : -1.0;
                lo=p.tip+sign*std::pow(p.t_scale*p.t_lo,2);
                hi=p.tip+sign*std::pow(p.t_scale*p.t_hi,2);
                if(lo>hi) std::swap(lo,hi);
            }
            std::fprintf(stderr,"POLAR_PANEL %.17g %.17g %.17g %.17g %.17g %d %d %d\n",
                lo,hi,p.flux,p.angular_error,p.legacy_quadrature_error,p.error_kind,p.depth,p.transformed);
        };
        auto copy=pending;
        while(!copy.empty()) {
            const auto& p=copy.top();
            if(p.cap && p.pieces) {for(const auto& piece:*p.pieces) emit(piece);}
            else emit(p);
            copy.pop();
        }
    }
    const double final_budget=work.relative_tolerance>0.0
        ? work.relative_tolerance*std::abs(flux/norm) : work.absolute_tolerance;
    if(work.met && !(work.error_indicator<=final_budget)) {
        work.met=false;work.stop_reason="resolution_limit";
    }
    return flux / norm;
}
