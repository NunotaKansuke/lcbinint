#pragma once
// Research-only D14 replacement. The caller retains ownership of caustic
// preparation/method selection and supplies its existing representative seeds
// in the primary frame (lenses at 0 and a). No additional lens equation solve.
#include "lcbinint/magnification/holonomic/seeded_boundary_events.hpp"
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
namespace lcbinint::holonomic {
struct SeededEpoch {
 SeededBoundaryResult discovery;
 TopologyResult topology;
 AdaptiveResult integral;
};
inline SeededEpoch epoch_adaptive_seeded(
 const LensParams&p,const std::vector<Cplx<double>>&primary_seeds,double u,
 const AdaptiveConfig&cfg,AdaptiveWorkspace&workspace,
 const SeededBoundaryConfig&discovery_cfg={}) {
 SeededEpoch out;auto pf=PrimaryFrame::from(p);
 out.discovery=seeded_boundary_events(pf,primary_seeds,discovery_cfg);
 out.topology=seeded_boundary_cell_plan(pf,out.discovery);
 out.integral=flux_adaptive_integrate(p,u,out.topology,cfg,workspace);
 return out;
}
}
