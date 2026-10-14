#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/fused_gradient.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"

namespace py=pybind11;
using lcbinint::holonomic::FusedGradientOptions;
using lcbinint::holonomic::LensParams;
using lcbinint::holonomic::epoch_fused_gradient;

struct PreparedEpoch {
    lcbinint::holonomic::FusedGradientResult result;
    bool seed_support_proven = false;
    std::size_t center_image_count = 0;
    std::size_t augmented_seed_count = 0;
};

// Keep the phase-ordered caustic cache and the point-image continuation cache
// alive for the whole light curve. Center images alone are not complete when
// the finite source disk straddles a caustic: a fold pair can exist only on a
// subregion of the disk and be absent at its center.
class BinarySeededFusedGradient {
public:
    BinarySeededFusedGradient(double separation, double mass_ratio,
                              double source_radius, int caustic_bins)
        : separation_(separation), mass_ratio_(mass_ratio),
          source_radius_(source_radius), finite_(make_settings(caustic_bins)) {
        if (!(std::isfinite(separation_) && separation_ > 0.0 &&
              std::isfinite(mass_ratio_) && mass_ratio_ > 0.0 &&
              std::isfinite(source_radius_) && source_radius_ > 0.0)) {
            throw std::invalid_argument(
                "binary seeder requires positive finite separation, mass ratio and source radius");
        }
        finite_.binary_caustic_branches(separation_, mass_ratio_);
    }

    PreparedEpoch evaluate(double xs, double ys, double limb_darkening,
                           double mu_atol, double mu_rtol, bool barycentric) {
        const lcbinint::SourcePosition source{xs, ys};
        const auto seed_set = lcbinint::holonomic::make_caustic_aware_binary_seeds(
            finite_, point_, separation_, mass_ratio_, source, source_radius_);
        std::vector<std::array<double, 2>> seeds;
        seeds.reserve(seed_set.radial_seeds.size());
        for (const auto& image : seed_set.radial_seeds) {
            seeds.push_back({image.re, image.im});
        }

        PreparedEpoch prepared;
        prepared.seed_support_proven = seed_set.support_proven;
        prepared.center_image_count = seed_set.center_images.size();
        prepared.augmented_seed_count = seeds.size();
        prepared.result = epoch_fused_gradient(
            LensParams{xs, ys, source_radius_, mass_ratio_, separation_, barycentric},
            limb_darkening, seeds, {mu_atol, mu_rtol});
        return prepared;
    }

    PreparedEpoch evaluate_value(double xs, double ys, double limb_darkening,
                                 double mu_atol, double mu_rtol,
                                 bool barycentric) {
        const lcbinint::SourcePosition source{xs, ys};
        const auto seed_set = lcbinint::holonomic::make_caustic_aware_binary_seeds(
            finite_, point_, separation_, mass_ratio_, source, source_radius_);
        lcbinint::holonomic::AdaptiveWorkspace workspace;
        const auto value = lcbinint::holonomic::epoch_seeded_radial_frontier_value(
            LensParams{xs, ys, source_radius_, mass_ratio_, separation_, barycentric},
            limb_darkening, seed_set.radial_seeds, mu_atol, mu_rtol, workspace);

        PreparedEpoch prepared;
        prepared.result.mu = value.integral.mu;
        prepared.result.value_converged = value.integral.value_converged;
        prepared.result.stop = lcbinint::holonomic::adaptive_stop_name(
            value.integral.stop);
        prepared.result.support_certified = value.frontier.coverage_certified;
        prepared.result.node_evaluations = value.integral.stats.node_evaluations;
        prepared.seed_support_proven = seed_set.support_proven;
        prepared.center_image_count = seed_set.center_images.size();
        prepared.augmented_seed_count = seed_set.radial_seeds.size();
        return prepared;
    }

private:
    static lcbinint::magnification::FiniteSourceSettings make_settings(
        int caustic_bins) {
        if (caustic_bins < 64) {
            throw std::invalid_argument("caustic_bins must be at least 64");
        }
        lcbinint::magnification::FiniteSourceSettings settings;
        settings.caustic_bins = caustic_bins;
        return settings;
    }

    double separation_;
    double mass_ratio_;
    double source_radius_;
    lcbinint::magnification::PointSourceMagnifier point_;
    lcbinint::magnification::FiniteSourceMagnifier finite_;
};

PYBIND11_MODULE(_lcbinint_holonomic_fused,m) {
    m.doc()="Optional forward fused finite-source gradient for one binary-lens epoch.";
    m.def("epoch_fused_gradient",[](double xs,double ys,double rho,double q,double separation,
        double u,const std::vector<std::array<double,2>>& seeds,double mu_atol,double mu_rtol,
        bool barycentric) {
        const LensParams p{xs,ys,rho,q,separation,barycentric};
        const FusedGradientOptions options{mu_atol,mu_rtol};
        const auto r=epoch_fused_gradient(p,u,seeds,options);
        py::dict out;
        out["mu"]=r.mu;out["grad_mu"]=r.grad_mu;
        out["gradient_valid"]=r.gradient_valid;
        out["value_converged"]=r.value_converged;
        out["gradient_precision_checked"]=r.gradient_precision_checked;
        out["support_certified"]=r.support_certified;
        out["stop"]=r.stop;out["node_evaluations"]=r.node_evaluations;
        return out;
    },py::arg("xs"),py::arg("ys"),py::arg("rho"),py::arg("q"),py::arg("separation"),
      py::arg("limb_darkening"),py::arg("primary_image_seeds"),py::arg("mu_atol")=1e-8,
      py::arg("mu_rtol")=1e-4,py::arg("barycentric")=false);

    py::class_<BinarySeededFusedGradient>(m, "BinarySeededFusedGradient")
        .def(py::init<double, double, double, int>(),
             py::arg("separation"), py::arg("mass_ratio"),
             py::arg("source_radius"), py::arg("caustic_bins") = 1400)
        .def("evaluate", [](BinarySeededFusedGradient& evaluator,
                            double xs, double ys, double limb_darkening,
                            double mu_atol, double mu_rtol, bool barycentric) {
            PreparedEpoch prepared;
            {
                py::gil_scoped_release release;
                prepared = evaluator.evaluate(xs, ys, limb_darkening,
                                              mu_atol, mu_rtol, barycentric);
            }
            const auto& r = prepared.result;
            py::dict out;
            out["mu"] = r.mu;
            out["grad_mu"] = r.grad_mu;
            out["gradient_valid"] = r.gradient_valid;
            out["value_converged"] = r.value_converged;
            out["gradient_precision_checked"] = r.gradient_precision_checked;
            out["support_certified"] = r.support_certified;
            out["seed_support_proven"] = prepared.seed_support_proven;
            out["center_image_count"] = prepared.center_image_count;
            out["augmented_seed_count"] = prepared.augmented_seed_count;
            out["stop"] = r.stop;
            out["node_evaluations"] = r.node_evaluations;
            return out;
        }, py::arg("xs"), py::arg("ys"), py::arg("limb_darkening"),
           py::arg("mu_atol") = 1e-8, py::arg("mu_rtol") = 1e-4,
           py::arg("barycentric") = false)
        .def("evaluate_value", [](BinarySeededFusedGradient& evaluator,
                                  double xs, double ys, double limb_darkening,
                                  double mu_atol, double mu_rtol,
                                  bool barycentric) {
            PreparedEpoch prepared;
            {
                py::gil_scoped_release release;
                prepared = evaluator.evaluate_value(xs, ys, limb_darkening,
                                                    mu_atol, mu_rtol, barycentric);
            }
            py::dict out;
            out["mu"] = prepared.result.mu;
            out["value_converged"] = prepared.result.value_converged;
            out["support_certified"] = prepared.result.support_certified;
            out["seed_support_proven"] = prepared.seed_support_proven;
            out["center_image_count"] = prepared.center_image_count;
            out["augmented_seed_count"] = prepared.augmented_seed_count;
            out["stop"] = prepared.result.stop;
            out["node_evaluations"] = prepared.result.node_evaluations;
            return out;
        }, py::arg("xs"), py::arg("ys"), py::arg("limb_darkening"),
           py::arg("mu_atol") = 1e-8, py::arg("mu_rtol") = 1e-4,
           py::arg("barycentric") = false);
}
