#include "bind_lc.hpp"
#include "bind_obs.hpp"
#include "bind_jax_ir.hpp"

#include "lcbinint/magnification/probe_diagnostics.hpp"
#include "lcbinint/magnification/binary_image_discriminant.hpp"
#include "lcbinint/magnification/bennett_integration.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/critical_phase_enclosure.hpp"
#include "lcbinint/magnification/binary_boundary_enclosure.hpp"
#include "lcbinint/magnification/critical_support_certificate.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_lcbinint, m)
{
    m.doc() = "Python bindings for the lcbinint C++ core";

    register_lc_submodule(m);
    register_obs_submodule(m);
    register_jax_ir_submodule(m);

    // Calibration instrumentation for the seeding stage.  Inert unless
    // LCBININT_PROBE_STATS is set, and thread-local, so a harness that wants a
    // total runs the library single-threaded; see probe_diagnostics.hpp.
    m.def("probe_counters", []() {
        const auto& counters = lcbinint::magnification::probe_counters();
        py::dict out;
        out["enabled"] = lcbinint::magnification::probe_stats_enabled();
        out["policy"] = lcbinint::magnification::probe_policy_description();
        out["continuation_newton_stagnations"] = counters.continuation_newton_stagnations;
        out["continuation_inclusion_failures"] = counters.continuation_inclusion_failures;
        out["continuation_duplicate_failures"] = counters.continuation_duplicate_failures;
        out["continuation_attempts"] = counters.continuation_attempts;
        out["continuation_accepted"] = counters.continuation_accepted;
        out["continuation_seconds"] = counters.continuation_seconds;
        out["certified_solves"] = counters.certified_solves;
        out["certified_offered"] = counters.certified_offered;
        out["certified_extrema"] = counters.certified_extrema;
        out["certifications"] = counters.certifications;
        out["unproven"] = counters.unproven;
        out["certified_seconds"] = counters.certified_seconds;
        out["certify_seconds"] = counters.certify_seconds;
        out["representative_enabled"] = lcbinint::magnification::representative_probes_enabled();
        out["representative_groups"] = counters.representative_groups;
        out["representative_boxes_resolved"] = counters.representative_boxes_resolved;
        out["representative_skipped"] = counters.representative_skipped;
        out["representative_mismatches"] = counters.representative_mismatches;
        out["source_cell_cache_hits"] = counters.source_cell_cache_hits;
        out["source_cell_enclosures"] = counters.source_cell_enclosures;
        out["source_cell_reuses"] = counters.source_cell_reuses;
        out["source_cell_fallbacks"] = counters.source_cell_fallbacks;
        out["polar_fill_calls"] = counters.polar_fill_calls;
        out["polar_map_queries"] = counters.polar_map_queries;
        out["polar_seed_positions"] = counters.polar_seed_positions;
        out["polar_fill_cells"] = counters.polar_fill_cells;
        out["polar_fill_seconds"] = counters.polar_fill_seconds;
        out["polar_coverage_calls"] = counters.polar_coverage_calls;
        out["polar_coverage_boxes"] = counters.polar_coverage_boxes;
        out["polar_coverage_unresolved"] = counters.polar_coverage_unresolved;
        out["polar_coverage_repairs"] = counters.polar_coverage_repairs;
        out["polar_coverage_certified"] = counters.polar_coverage_certified;
        out["polar_coverage_seconds"] = counters.polar_coverage_seconds;
        out["cartesian_fill_calls"] = counters.cartesian_fill_calls;
        out["cartesian_map_queries"] = counters.cartesian_map_queries;
        out["cartesian_fill_cells"] = counters.cartesian_fill_cells;
        out["cartesian_fill_seconds"] = counters.cartesian_fill_seconds;
        out["cartesian_coverage_calls"] = counters.cartesian_coverage_calls;
        out["cartesian_coverage_certified"] = counters.cartesian_coverage_certified;
        out["cartesian_coverage_boxes"] = counters.cartesian_coverage_boxes;
        out["cartesian_coverage_unresolved"] = counters.cartesian_coverage_unresolved;
        out["cartesian_coverage_repairs"] = counters.cartesian_coverage_repairs;
        out["cartesian_coverage_seconds"] = counters.cartesian_coverage_seconds;
        out["critical_support_calls"] = counters.critical_support_calls;
        out["critical_support_proven"] = counters.critical_support_proven;
        out["critical_support_unresolved"] = counters.critical_support_unresolved;
        out["critical_support_phase_queries"] = counters.critical_support_phase_queries;
        out["critical_support_proof_queries"] = counters.critical_support_proof_queries;
        out["critical_support_seconds"] = counters.critical_support_seconds;
        out["polar_diagonal_runs"] = counters.polar_diagonal_runs;
        out["critical_arc_seeds"] = counters.critical_arc_seeds;
        out["critical_uncovered_probes"] = counters.critical_uncovered_probes;
        out["critical_incremental_solves"] = counters.critical_incremental_solves;
        out["critical_centre_solves"] = counters.critical_centre_solves;
        out["critical_grid_checks"] = counters.critical_grid_checks;
        out["critical_grid_components"] = counters.critical_grid_components;
        out["critical_phase_solves"] = counters.critical_phase_solves;
        out["critical_phase_tracked"] = counters.critical_phase_tracked;
        out["critical_newton_iterations"] = counters.critical_newton_iterations;
        out["critical_tracking_failures"] = counters.critical_tracking_failures;
        out["critical_build_seconds"] = counters.critical_build_seconds;
        out["representative_box_seconds"] = counters.representative_box_seconds;
        out["representative_count_seconds"] = counters.representative_count_seconds;
        out["representative_link_checks"] = counters.representative_link_checks;
        out["representative_polyline_clear"] = counters.representative_polyline_clear;
        out["representative_verified_skipped"] = counters.representative_verified_skipped;
        out["representative_conditional_skipped"] = counters.representative_conditional_skipped;
        out["representative_link_boxes"] = counters.representative_link_boxes;
        out["representative_link_seconds"] = counters.representative_link_seconds;
        out["representative_count_ablation"] = lcbinint::magnification::representative_count_ablation_enabled();
        out["discriminant_enabled"] = lcbinint::magnification::discriminant_diagnostics_enabled();
        out["discriminant_calls"] = counters.discriminant_calls;
        out["discriminant_resolved"] = counters.discriminant_resolved;
        out["discriminant_filtered"] = counters.discriminant_filtered;
        out["discriminant_unresolved"] = counters.discriminant_unresolved;
        out["discriminant_mismatches"] = counters.discriminant_mismatches;
        out["discriminant_seconds"] = counters.discriminant_seconds;
        out["boundary_enabled"] = lcbinint::magnification::boundary_diagnostics_enabled();
        out["boundary_model_edges"] = counters.boundary_model_edges;
        out["boundary_model_unresolved"] = counters.boundary_model_unresolved;
        out["boundary_model_extra_evaluations"] = counters.boundary_model_extra_evaluations;
        out["boundary_model_reused_evaluations"] = counters.boundary_model_reused_evaluations;
        out["boundary_outer_curvature_flux_indicator"] = counters.boundary_outer_curvature_flux_indicator;
        out["boundary_model_absolute_flux_residual"] = counters.boundary_model_absolute_flux_residual;
        out["boundary_model_signed_flux_residual"] = counters.boundary_model_signed_flux_residual;
        return out;
    }, "Per-thread probe tallies since the last reset.");

    m.def("reset_probe_counters", &lcbinint::magnification::reset_probe_counters,
          "Zero the calling thread's probe tallies.");
    m.def("_binary_centre_seed_certificate", [](double s,double q,double x,double y,double rho) {
        return lcbinint::magnification::binary_centre_seed_certificate(s,q,{x,y},rho);
    });
    m.def("_binary_rounded_geometry_count", [](double s,double q,double x,double y) {
        const auto count=lcbinint::magnification::binary_image_count_rounded_geometry(s,q,{x,y});
        return count.image_count;
    });
    m.def("_binary_critical_support_geometry_tubes", [](double s,double q) {
        py::list out;
        for (const auto& t:lcbinint::magnification::cached_critical_support_geometry(s,q).audit_tubes(48)) {
            py::dict item;item["index"]=t.index;item["bins"]=t.bins;item["root"]=t.root;
            item["centre"]=py::make_tuple(t.centre.x,t.centre.y);item["radius"]=t.radius;
            item["source_lower"]=py::make_tuple(t.source_lower.x,t.source_lower.y);
            item["source_upper"]=py::make_tuple(t.source_upper.x,t.source_upper.y);out.append(item);
        }
        return out;
    });
    m.def("_binary_critical_cluster_enclosure", [](double s,double q,int index,int bins,py::list proposals) {
        if (py::len(proposals)!=4) throw py::value_error("four root proposals required");
        std::array<lcbinint::SourcePosition,4> roots;
        for (std::size_t j=0;j<4;++j) {
            const auto pair=py::cast<py::tuple>(proposals[j]);roots[j]={py::cast<double>(pair[0]),py::cast<double>(pair[1])};
        }
        py::list out;
        for (const auto& c:lcbinint::magnification::binary_critical_cluster_enclosure(s,q,index,bins,roots)) {
            py::dict item;item["centre"]=py::make_tuple(c.centre.x,c.centre.y);item["radius"]=c.radius;
            item["roots"]=c.roots;item["source_lower"]=py::make_tuple(c.source_lower.x,c.source_lower.y);
            item["source_upper"]=py::make_tuple(c.source_upper.x,c.source_upper.y);out.append(item);
        }
        return out;
    });
    m.def("_binary_critical_support_certificate", [](double s,double q,double x,double y,double rho) {
        const auto proof=lcbinint::magnification::cached_critical_support_certificate(s,q,{x,y},rho);
        py::dict out;out["resolved"]=proof.resolved;out["phase_queries"]=proof.phase_queries;
        out["proof_queries"]=proof.proof_queries;out["source_visits"]=proof.source_visits;
        out["outside_roots"]=proof.outside_roots;out["inside_roots"]=proof.inside_roots;
        out["monotone_roots"]=proof.monotone_roots;out["point_failures"]=proof.point_failures;
        out["match_failures"]=proof.match_failures;
        py::list unresolved;
        for (const auto& u:proof.unresolved) unresolved.append(py::make_tuple(u[0],u[1],u[2]));
        out["unresolved"]=unresolved;
        py::list witnesses;
        for (const auto& w:proof.witnesses)
            witnesses.append(py::make_tuple(w.position.x,w.position.y,w.inside_radius));
        out["witnesses"]=witnesses;return out;
    });
    m.def("_boundary_pair_error", [](double a,double b,double alo,double ahi,double blo,double bhi) {
        const auto p=lcbinint::magnification::boundary_pair_error(a,b,alo,ahi,blo,bhi);
        py::dict out;out["bounded"]=p.bounded;out["signed_lower"]=p.signed_lower;out["signed_upper"]=p.signed_upper;
        out["absolute_upper"]=p.absolute_upper;out["separate_absolute_upper"]=p.separate_absolute_upper;return out;
    });
    m.def("_binary_boundary_cap", [](double s,double q,double sx,double sy,double rho,
        double xlo,double xhi,double ylo,double yhi,int columns,bool extent) {
        const auto p=lcbinint::magnification::binary_boundary_cap(s,q,{sx,sy},rho,xlo,xhi,ylo,yhi,columns,extent);
        py::dict out;out["bounded"]=p.bounded;out["area_lower"]=p.area_lower;out["area_upper"]=p.area_upper;
        out["columns"]=p.columns;out["monotone_columns"]=p.monotone_columns;out["evaluations"]=p.evaluations;
        out["extent_checked"]=p.extent_checked;out["lower_face_outside"]=p.lower_face_outside;
        out["left_face_outside"]=p.left_face_outside;out["right_face_outside"]=p.right_face_outside;return out;
    },py::arg("s"),py::arg("q"),py::arg("sx"),py::arg("sy"),py::arg("rho"),py::arg("xlo"),py::arg("xhi"),py::arg("ylo"),py::arg("yhi"),py::arg("columns")=32,py::arg("check_extent")=false);
    m.def("_binary_boundary_strip", [](double s,double q,double sx,double sy,double rho,
        double llo,double lhi,double rlo,double rhi,double ylo,double yhi,int depth) {
        const auto p=lcbinint::magnification::binary_boundary_strip(s,q,{sx,sy},rho,llo,lhi,rlo,rhi,ylo,yhi,depth);
        py::dict out;out["resolved"]=p.resolved;out["area_lower"]=p.area_lower;out["area_upper"]=p.area_upper;
        out["certified_substrips"]=p.certified_substrips;out["unresolved_substrips"]=p.unresolved_substrips;return out;
    },py::arg("s"),py::arg("q"),py::arg("sx"),py::arg("sy"),py::arg("rho"),py::arg("left_lo"),py::arg("left_hi"),py::arg("right_lo"),py::arg("right_hi"),py::arg("ylo"),py::arg("yhi"),py::arg("depth")=4);
    m.def("_binary_boundary_tube", [](double s,double q,double sx,double sy,double rho,
                                     double xlo,double xhi,double ylo,double yhi) {
        const auto proof=lcbinint::magnification::binary_boundary_tube(
            s,q,{sx,sy},rho,xlo,xhi,ylo,yhi);
        py::dict out;out["resolved"]=proof.resolved;
        out["slope_lower"]=proof.slope_lower;out["slope_upper"]=proof.slope_upper;
        out["curvature_lower"]=proof.curvature_lower;out["curvature_upper"]=proof.curvature_upper;
        out["midpoint_area_error_upper"]=proof.midpoint_area_error_upper;return out;
    });
    m.def("_binary_boundary_enclosure", [](double s,double q,double sx,double sy,double rho,
                                          double inside_x,double outside_x,double y,double proposal) {
        const auto proof=lcbinint::magnification::binary_boundary_enclosure(
            s,q,{sx,sy},rho,inside_x,outside_x,y,proposal);
        py::dict out;out["resolved"]=proof.resolved;out["lower"]=proof.lower;
        out["upper"]=proof.upper;out["iterations"]=proof.iterations;return out;
    });
    m.def("_binary_critical_phase_proposals", [](double s,double q,double phase) {
        py::list output;
        for (const auto& p : lcbinint::magnification::binary_critical_phase_proposals(s,q,phase))
            output.append(py::make_tuple(p.x,p.y));
        return output;
    });
    m.def("_binary_critical_phase_enclosure", [](double s,double q,int index,int bins,
                                               py::list proposals,double fraction) {
        if (py::len(proposals)!=4) throw py::value_error("four root proposals required");
        std::array<lcbinint::SourcePosition,4> roots;
        for (std::size_t j=0;j<4;++j) {
            const auto pair=py::cast<py::tuple>(proposals[j]);
            roots[j]={py::cast<double>(pair[0]),py::cast<double>(pair[1])};
        }
        const auto proof=lcbinint::magnification::binary_critical_phase_enclosure(
            s,q,index,bins,roots,fraction);
        py::dict output;output["resolved"]=proof.resolved;
        py::list radii,margins,source_lower,source_upper;
        for (std::size_t j=0;j<4;++j) {
            radii.append(proof.radii[j]);margins.append(proof.margins[j]);
            source_lower.append(py::make_tuple(proof.source_lower[j].x,proof.source_lower[j].y));
            source_upper.append(py::make_tuple(proof.source_upper[j].x,proof.source_upper[j].y));
        }
        output["radii"]=radii;output["margins"]=margins;
        output["source_lower"]=source_lower;output["source_upper"]=source_upper;
        return output;
    },py::arg("s"),py::arg("q"),py::arg("index"),py::arg("bins"),
      py::arg("roots"),py::arg("fraction")=1.0);
    m.def("_binary_critical_curve_diagnostic", [](double s,double q,int bins) {
        lcbinint::magnification::FiniteSourceSettings settings;
        settings.caustic_bins=bins;
        lcbinint::magnification::FiniteSourceMagnifier magnifier(settings);
        py::list output;
        for (const auto& branch : magnifier.binary_critical_curve_branches_cached(s,q)) {
            py::list points;
            for (const auto& p : branch) points.append(py::make_tuple(p.x,p.y));
            output.append(points);
        }
        return output;
    },py::arg("s"),py::arg("q"),py::arg("bins")=1400);
    m.def("_binary_image_count_discriminant",
        [](double s, double q, double x, double y, bool interval_only) {
            const auto count = lcbinint::magnification::binary_image_count_discriminant(
                s,q,{x,y},!interval_only);
            py::dict out;
            out["image_count"] = count.image_count;
            out["status"] = lcbinint::magnification::discriminant_status_name(count.status);
            out["filtered"] = count.filtered;
            return out;
        }, py::arg("s"), py::arg("q"), py::arg("x"), py::arg("y"),
        py::arg("interval_only") = false,
        "Diagnostic real-quintic filtered sign; -1 means a root solve is required.");
    m.def("_binary_source_box",[](double s,double q,double x,double y,double rho) {
        const auto box=lcbinint::magnification::binary_source_box(s,q,{x,y},rho);
        py::dict out;
        out["lower"]=py::make_tuple(box.lower.x,box.lower.y);
        out["upper"]=py::make_tuple(box.upper.x,box.upper.y);
        out["image_count"]=box.image_count;
        out["cache_hits"]=box.cache_hits;
        out["enclosure_evaluations"]=box.enclosure_evaluations;
        return out;
    },py::arg("s"),py::arg("q"),py::arg("x"),py::arg("y"),py::arg("rho"),
       "Diagnostic cached source-cell certificate; a miss preserves partial refinement.");
    m.def("_bennett_boundary_model_residual",
        [](double delta, double node, double next_node, double limb, double delta_c) {
            const auto value = lcbinint::magnification::detail::bennett_boundary_model_residual(
                delta,node,next_node,limb,delta_c);
            py::dict out;
            out["resolved"] = value.resolved;
            out["signed_residual"] = value.signed_residual;
            out["sqrt_coefficient"] = value.sqrt_coefficient;
            out["linear_coefficient"] = value.linear_coefficient;
            return out;
        }, py::arg("delta"), py::arg("node"), py::arg("next_node"),
        py::arg("limb"), py::arg("delta_c")=0.15,
        "Local model defect only; excludes the unknown remainder and is not an error bound.");
}
