#include "bind_triple_algebraic.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#ifndef LCBININT_TRIPLE_STANDALONE_MODULE
#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#endif
#include <pybind11/complex.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace triple=lcbinint::algebraic_boundary::triple;
namespace {
template<bool Gradient> py::dict compute(const std::array<triple::Complex,3>& lenses,
    const std::array<double,3>& masses,triple::Complex source,double rho,double limb,
    double atol,double rtol,int event_budget,bool value_mesh) {
    for(double m:masses)if(!(m>0) || !std::isfinite(m))throw std::invalid_argument("triple algebraic requires positive masses");
    if(std::abs(masses[0]+masses[1]+masses[2]-1)>1e-12)
        throw std::invalid_argument("triple masses must sum to one");
    const auto frame=triple::primary_frame<Gradient>(lenses,source,rho,masses[1],masses[2]);
    triple::IntegrationOptions options;options.absolute_tolerance=atol;options.relative_tolerance=rtol;
    options.event_box_budget=event_budget;
    if constexpr(Gradient)options.gradient_value_mesh=value_mesh;
    triple::IntegrationResult<Gradient> result;
    std::string regular_stop="not_attempted";
    {
        py::gil_scoped_release release;
        triple::RadialEventPlan plan;
#ifndef LCBININT_TRIPLE_STANDALONE_MODULE
        if(std::isfinite(rtol) && rtol>=1e-4 &&
                std::isfinite(atol) && atol>=0 && std::isfinite(limb) && limb>=0 && limb<=1) {
            static thread_local triple::PreparedRegularSupport prepared;
            plan=prepared.plan(frame.geometry,rtol,event_budget,!Gradient);regular_stop=plan.reason;
        }
#endif
        result=triple::integrate<Gradient>(frame.geometry,limb,options,plan.completed?&plan:nullptr);
    }
    py::dict out;
    out["mu"]=result.magnification;out["error_estimate"]=result.error_estimate;
    out["value_converged"]=result.converged;out["support_complete"]=result.support_complete;
    out["gradient_reliable"]=result.gradient_reliable;
    out["gradient_valid"]=result.gradient_valid;
    out["gradient_precision_checked"]=result.gradient_precision_checked;
    out["gradient_policy"]=Gradient?(value_mesh?"value_mesh":"strict"):"none";
    out["smooth_source_used"]=result.smooth_source_used;
    out["smooth_source_samples"]=result.smooth_source_samples;
    out["smooth_source_reused_samples"]=result.smooth_source_reused_samples;out["stop"]=result.reason;
    out["node_evaluations"]=result.value_evaluations;
    out["parameter_derivative_evaluations"]=result.parameter_derivative_evaluations;
    out["quad_samples"]=result.quad_samples;out["refinements"]=result.refinements;
    out["component_domains"]=result.component_domains;
    out["h_splits"]=result.h_splits;out["event_boxes"]=result.support.boxes;
    out["event_resultant_nodes"]=result.support.resultant_nodes;
    out["event_resultant_events"]=result.support.resultant_events;
    out["event_resultant_solves"]=result.support.resultant_solves;
    out["event_quad_boxes"]=result.support.quad_boxes;
    out["last_event_box"]=result.support.last_box;
    out["radial_events"]=result.support.events.size();
    out["event_quad_refinements"]=result.support.quad_refinements;
    out["event_double_refinements"]=result.support.double_refinements;
    out["support_route"]=result.support.reason;
    out["regular_support_stop"]=regular_stop;
    out["bounded_components"]=result.support.excluded_images.size();
    out["bounded_component_value"]=result.support.bounded_component_value;
    out["bounded_component_error"]=result.support.bounded_component_error;
    out["continued_radii"]=result.continued_radii;out["global_radii"]=result.global_radii;
    if constexpr(Gradient) {
        out["gradient"]=frame.pullback(result.normalized.df0);
        out["gradient_parameters"]=std::vector<std::string>{"source_x","source_y","rho","m1","m2",
            "lens0_x","lens0_y","lens1_x","lens1_y","lens2_x","lens2_y"};
        out["canonical_gradient"]=result.normalized.df0;
        out["canonical_gradient_error_estimate"]=result.normalized.gradient_error;
    }
    return out;
}
}
void register_triple_algebraic(py::module_& module) {
    module.def("triple_algebraic",[](const std::array<triple::Complex,3>& lenses,
        const std::array<double,3>& masses,triple::Complex source,double rho,double limb,
        bool with_gradient,double atol,double rtol,int event_budget,const std::string& gradient_policy) {
        if(gradient_policy!="strict" && gradient_policy!="value_mesh")throw std::invalid_argument("gradient_policy must be strict or value_mesh");
        const bool value_mesh=gradient_policy=="value_mesh";
        return with_gradient?compute<true>(lenses,masses,source,rho,limb,atol,rtol,event_budget,value_mesh):
                             compute<false>(lenses,masses,source,rho,limb,atol,rtol,event_budget,value_mesh);
    },py::arg("lenses"),py::arg("masses"),py::arg("source"),py::arg("rho"),
      py::arg("limb")=0.,py::arg("with_gradient")=false,
      py::arg("atol")=1e-6,py::arg("rtol")=1e-5,py::arg("event_box_budget")=1000000,py::arg("gradient_policy")="value_mesh",
      "Algebraic triple finite-source integration. Positive normalized masses; linear limb darkening. "
      "strict checks value and derivative errors; value_mesh refines on flux only. Inspect value_converged, gradient_valid, gradient_reliable and gradient_precision_checked.");
}
#ifdef LCBININT_TRIPLE_STANDALONE_MODULE
PYBIND11_MODULE(_triple_algebraic_test,module){register_triple_algebraic(module);}
#endif
