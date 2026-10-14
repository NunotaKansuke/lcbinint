#pragma once

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cstring>

namespace lcbinint::magnification {

// Calibration instrumentation for the certificate seeding stage.
//
// The production cached-binary path uses the source-centre images followed by
// the component certificate.  Each certificate probe is a full quintic solve.
//
// Whether that is affordable or wasteful is a measurement, not an opinion, and
// the shipping build reports neither of the two numbers it needs: what share of
// an evaluation the probes are, and whether the answer moves when a stage is
// removed.  Both are provided here, gated on environment variables, and both
// default to exactly the shipping behaviour when unset.
//
// This is deliberately not a public option.  A knob that can silently weaken
// the completeness argument does not belong in the API; it belongs in the
// calibration harness, which has to run the weakened variants in order to
// report what they cost.  ``probe_policy()`` is read once per process.

// Per-thread tallies.  Zero-initialised, and only written while
// ``probe_stats_enabled()``; the counters are thread-local because the row loop
// is an OpenMP parallel for, and a harness that wants a total either runs the
// library single-threaded or sums per thread.
struct ProbeCounters {
    // Point-image solves, which is the unit of probe cost: everything else in
    // the seeding stage is arithmetic on the results.
    long long continuation_newton_stagnations = 0;
    long long continuation_inclusion_failures = 0;
    long long continuation_duplicate_failures = 0;
    long long continuation_attempts = 0;
    long long continuation_accepted = 0;
    double continuation_seconds = 0;
    long long certified_solves = 0;
    // What the certificate proposed against what was actually spent.  The
    // traversal stops a direction as soon as it departs from its baseline.
    // This counts the nominal source-limb ladder; lazy local-scale refinement
    // is charged to certified_solves and can exceed the nominal offered count.
    long long certified_offered = 0;
    long long certified_extrema = 0;
    // Support descriptors built, and disks whose support the ladder failed to
    // prove.  The second is the accuracy cost of an ablation: a support that
    // stops being proven is refused, not silently returned wrong.
    long long certifications = 0;
    long long unproven = 0;
    // Wall seconds, split so the polyline scan that builds the descriptor is
    // separable from the solves that consume it -- they are reduced by
    // different means.
    double certified_seconds = 0.0;
    double certify_seconds = 0.0;
    // Shadow classifiers do not change the roots, seeds, or support decision.
    long long discriminant_calls = 0;
    long long discriminant_resolved = 0;
    long long discriminant_filtered = 0;
    long long discriminant_unresolved = 0;
    long long discriminant_mismatches = 0;
    double discriminant_seconds = 0.0;
    long long representative_groups = 0;
    long long representative_boxes_resolved = 0;
    long long representative_skipped = 0;
    long long representative_mismatches = 0;
    long long source_cell_cache_hits = 0;
    long long source_cell_enclosures = 0;
    long long source_cell_reuses = 0;
    long long source_cell_fallbacks = 0;
    long long polar_fill_calls=0,polar_map_queries=0,polar_seed_positions=0,polar_fill_cells=0;
    double polar_fill_seconds=0;
    long long polar_diagonal_runs=0;
    long long polar_coverage_calls=0,polar_coverage_boxes=0,polar_coverage_unresolved=0,polar_coverage_repairs=0,polar_coverage_certified=0;
    double polar_coverage_seconds=0;
    long long cartesian_fill_calls=0,cartesian_map_queries=0,cartesian_fill_cells=0;
    double cartesian_fill_seconds=0;
    long long cartesian_coverage_calls=0,cartesian_coverage_certified=0,cartesian_coverage_boxes=0,cartesian_coverage_unresolved=0,cartesian_coverage_repairs=0;
    double cartesian_coverage_seconds=0;
    long long critical_support_calls=0,critical_support_proven=0,critical_support_unresolved=0,critical_support_phase_queries=0,critical_support_proof_queries=0;
    double critical_support_seconds=0;
    long long critical_arc_seeds = 0;
    long long critical_uncovered_probes = 0;
    long long critical_incremental_solves = 0;
    long long critical_centre_solves = 0;
    long long critical_grid_checks = 0;
    long long critical_grid_components = 0;
    long long critical_phase_solves = 0;
    long long critical_phase_tracked = 0;
    long long critical_newton_iterations = 0;
    long long critical_tracking_failures = 0;
    double critical_build_seconds = 0.0;
    double representative_box_seconds = 0.0;
    double representative_count_seconds = 0.0;
    long long representative_link_checks = 0;
    long long representative_polyline_clear = 0;
    long long representative_verified_skipped = 0;
    long long representative_conditional_skipped = 0;
    long long representative_link_boxes = 0;
    double representative_link_seconds = 0.0;
    // Local boundary model residuals (raw image flux units, summed per thread).
    // These exclude interior/outer-coordinate errors and are not error bounds.
    long long boundary_model_edges = 0;
    long long boundary_model_unresolved = 0;
    long long boundary_model_extra_evaluations = 0;
    long long boundary_model_reused_evaluations = 0;
    double boundary_outer_curvature_flux_indicator = 0.0;
    double boundary_component_flux_indicator = 0.0;
    double boundary_outer_signed_flux_residual=0;
    double boundary_component_signed_flux_residual=0;
    double boundary_component_remainder_flux_indicator=0;
    double boundary_outer_remainder_flux_indicator=0;
    double brightness_table_signed_flux_residual=0;
    double brightness_table_remainder_flux_indicator=0;
    long long error_refined_components=0,error_outer_fallbacks=0;
    double boundary_model_absolute_flux_residual = 0.0;
    double boundary_model_signed_flux_residual = 0.0;
};

ProbeCounters& probe_counters();
bool probe_stats_enabled();
bool discriminant_diagnostics_enabled();
bool boundary_diagnostics_enabled();
// Scoped, thread-local shared flux indicator. Does not enable expensive profiling.
bool& lightweight_flux_error_mode();
// One-call retained adaptive context. No persistent geometry cache or default
// routing change. Lattice values survive every local component replacement.
struct AdaptiveFluxRequest {
 struct Hash {std::size_t operator()(const std::array<double,2>& v)const {
  const auto a=std::hash<double>{}(v[0]),b=std::hash<double>{}(v[1]);return a^(b+0x9e3779b9+(a<<6)+(a>>2));
 }};
 struct LatticeCache {
  struct Slot {
   std::array<double,2> key;double value;
   Slot() {} // The insertion path initializes the key and value before use.
  };
  std::vector<Slot> slots;
  // Initial evaluations remain available without hashing every new point.
  // Build the searchable cache only when refinement or lookup needs it.
  std::vector<Slot> pending;
  bool deferred=std::getenv("LCBININT_DIAGNOSTIC_DISABLE_DEFER_LATTICE_CACHE")==nullptr;
  // Empty/full and a hash fingerprint live together in compact control bytes.
  // Most probes need no payload cache line or floating-key comparison.
  std::vector<unsigned char> control;
  std::size_t size=0,initial_capacity=16384;
  bool fast_hash=true;
  std::size_t hash(const std::array<double,2>& key)const {
   if(!fast_hash)return Hash{}(key);
   // Exact floating coordinates remain the identity. Canonicalize signed zero
   // because the key equality treats +0 and -0 as the same coordinate.
   static_assert(sizeof(double)==sizeof(std::uint64_t),"lattice hash requires 64-bit double");
   std::uint64_t x=0,y=0;
   if(key[0]!=0)std::memcpy(&x,&key[0],sizeof(x));
   if(key[1]!=0)std::memcpy(&y,&key[1],sizeof(y));
   auto h=x+0x9e3779b97f4a7c15ULL+((y<<17)|(y>>47));
   h=(h^(h>>30))*0xbf58476d1ce4e5b9ULL;
   h=(h^(h>>27))*0x94d049bb133111ebULL;
   return h^(h>>31);
  }
  static unsigned char tag(std::size_t h) {
   return static_cast<unsigned char>((h>>56)%255+1);
  }
  void grow() {
   auto old=std::move(slots);auto old_control=std::move(control);
   slots.resize(old.empty()?initial_capacity:old.size()*2);
   control.assign(slots.size(),0);
   for(std::size_t i=0;i<old.size();++i)if(old_control[i]) {
    const auto h=hash(old[i].key);auto j=h&(slots.size()-1);
    while(control[j])j=(j+1)&(slots.size()-1);
    slots[j]=old[i];control[j]=tag(h);
   }
  }
  void materialize() {
   if(!deferred)return;
   deferred=false;
   auto records=std::move(pending);
   size=0;
   if(records.empty())return;
   // The number of retained records is known. Allocate once and bulk-build
   // without per-record capacity checks or intermediate rehashes.
   std::size_t capacity=std::max<std::size_t>(initial_capacity,2);
   while(records.size()*10>capacity*7)capacity*=2;
   slots.resize(capacity);
   control.assign(capacity,0);
   for(const auto& record:records) {
    const auto h=hash(record.key);const auto wanted=tag(h);
    auto j=h&(capacity-1);
    while(control[j]) {
     if(control[j]==wanted && slots[j].key==record.key)break;
     j=(j+1)&(capacity-1);
    }
    if(!control[j])++size;
    slots[j]=record;control[j]=wanted;
   }
  }
  std::pair<double*,bool> access(const std::array<double,2>& key,bool insert) {
   if(deferred) {
    if(!insert)materialize();
    else {
     if(pending.capacity()==0)pending.reserve(std::min<std::size_t>(initial_capacity,16384));
     pending.emplace_back();pending.back().key=key;++size;
     return {&pending.back().value,true};
    }
   }
   if(slots.empty()){if(!insert)return {nullptr,false};grow();}
   if(insert && (size+1)*10>slots.size()*7)grow();
   const auto h=hash(key);const auto wanted=tag(h);auto j=h&(slots.size()-1);
   while(control[j]) {
    if(control[j]==wanted && slots[j].key==key)return {&slots[j].value,false};
    j=(j+1)&(slots.size()-1);
   }
   if(!insert)return {nullptr,false};
   slots[j].key=key;control[j]=wanted;++size;return {&slots[j].value,true};
  }
 } mapped;

 // Optional relative tolerance, normalized once using the retained initial flux.
 // Zero preserves the existing absolute-flux target contract.
 double relative_target=0,initial_flux=0;
 double target=0,signed_error=0,remainder=0,acceptance_indicator=0;
 int max_bins=512,depth=0,largest_bins=0,max_axis_ratio=4,initial_bins=0;
 std::size_t budget=1000000,new_maps=0,reused_maps=0,refinements=0;
 bool met=false,executed=false;
 std::string stop_reason="not_executed";
 std::vector<std::array<double,2>> component_grid_bins;
 std::vector<std::array<double,6>> refinement_grid_bins;
 bool cost_trace=false;
 double run_fill_seconds=0,error_aggregation_seconds=0,adaptive_control_seconds=0;
 double seed_seconds=0,seed_root_seconds=0,initial_fill_seconds=0,row_boundary_seconds=0;
 std::size_t fill_queries=0,fill_rows=0,fill_cells=0,duplicate_seed_starts=0;
 std::size_t seed_root_calls=0;
 bool row_panel_trace=false;
 bool guard_missing_caps=std::getenv("LCBININT_DIAGNOSTIC_DISABLE_MISSING_CAP_GUARD")==nullptr;
 bool directional_adaptive=true,local_row_interpolation=true,local_boundary_strips=true;
 bool local_vertical_caps=true;
 bool defect_correction=true;
 // Coarse-start models are tentative until their endpoint footprint is
 // separated or one local dyadic layer resolves an unmodeled outer shape.
 bool coarse_start_guard=std::getenv("LCBININT_DIAGNOSTIC_DISABLE_COARSE_START_GUARD")==nullptr;
 bool coarse_start_probe=true;
 int coarse_start_min_rows=8; // Same two four-row endpoint footprint as the error model.
 std::size_t coarse_start_guards=0,coarse_start_probe_calls=0,coarse_start_merges=0;
 double applied_correction=0;
 int cap_initial_pad=4; // 0 is a diagnostic forced-extension probe.
 std::size_t cap_trials=0,cap_adopted=0,cap_rejected=0,reused_cap_rows=0,cap_extensions=0;
 std::vector<std::array<double,3>> mixed_component_grids;
 // stage,component,bins,row bins,iy,lo,hi,raw physical row integral,signed horizontal error,remainder,unresolved
 std::vector<std::array<double,11>> row_panel_rows;
 // stage,component,inner signed/remainder,cap signed/remainder,smooth remainder,junction signed/remainder,short guard,resolved
 std::vector<std::array<double,11>> row_panel_errors;
 std::vector<double> history_flux;
 // snapshot: stage,component,bins,flux,inner signed,outer signed,inner remainder,outer remainder,short guard,runs,cells,endpoint cells
 std::vector<std::array<double,12>> cost_components;
 // trial: stage,component,old bins,new bins,seconds,new maps,reused maps,adopted
 std::vector<std::array<double,8>> cost_trials;
 std::vector<std::array<double,5>> history,components;
 std::vector<std::array<double,2>> centres;
 std::size_t small_image_repairs=0,unresolved_centres=0,reused_rows=0;
 std::size_t interpolated_rows=0,row_interpolation_attempts=0;
 std::array<std::size_t,7> row_interpolation_outcomes{};
 std::size_t retained_horizontal_rows=0;
 std::size_t retained_boundary_strip_rows=0;
 std::size_t strip_audit_rows=0,strip_audit_gap_rows=0;
 double strip_audit_absolute_difference=0,strip_audit_max_difference=0;
 std::size_t strip_support_checks=0,strip_support_rejects=0;
 std::size_t interpolated_audit_rows=0,interpolated_audit_gap_rows=0;
 double interpolated_audit_max_difference=0;
};
struct AdaptiveFluxBudgetExhausted : std::runtime_error {
 AdaptiveFluxBudgetExhausted():std::runtime_error("adaptive lattice evaluation budget exhausted"){}
};
AdaptiveFluxRequest*& adaptive_flux_request();

// Experimental seed reduction; no default/public integration guarantee.
bool representative_probes_enabled();
// Deliberately unproven count-only ablation, never a support certificate.
bool representative_count_ablation_enabled();
bool representative_polyline_links_enabled();
bool representative_verified_links_enabled();
void reset_probe_counters();

// Diagnostic controls for the certificate probe ladder.  There is deliberately
// no switch for restoring the retired ring or branch-grid heuristic.
struct ProbePolicy {
    // How deep the certified offset ladder goes, counted from the coarse end,
    // and whether the tangent directions are probed at all.  Dropping the
    // tangents is expected to lose cusps; it is kept as a knob so the corpus
    // can be shown to notice.
    int offsets = 8;
    bool tangents = true;
    bool normals = true;
};

const ProbePolicy& probe_policy();

// Parsed form of LCBININT_PROBE_POLICY, for the harness to echo back.
std::string probe_policy_description();

} // namespace lcbinint::magnification
