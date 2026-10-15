from pathlib import Path
p=Path('build-algebraic-boundary/stress-cause/include/lcbinint/magnification/holonomic/adaptive_epoch.hpp');s=p.read_text();at='    bool deferred_children=false;';s=s.replace(at,'''    if(std::getenv("LCBININT_CAUSE_DEBUG")) {
        std::fprintf(stderr,"INTEGRAL %.17g %.17g %.17g %.17g %.17g\\n",p.xs,p.ys,p.rho,result.mu,result.grad_mu[2]);
        for(const auto& z:workspace.panels)if(z.active)std::fprintf(stderr,"PANEL %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g\\n",z.evaluator_cell,z.depth,z.level,z.map.a,z.map.b,z.xl,z.xr,z.q[0],z.q[3]);
    }
'''+at);p.write_text(s)
p=Path('build-algebraic-boundary/stress-cause/include/lcbinint/magnification/holonomic/seeded_radial_frontier.hpp');s=p.read_text().replace('if(!geometry_ok){','if(!geometry_ok){if(std::getenv("LCBININT_CAUSE_DEBUG"))std::fprintf(stderr,"NAN geometry_ok false\\n");').replace('if(!geometry_complete)for','if(!geometry_complete && std::getenv("LCBININT_CAUSE_DEBUG"))std::fprintf(stderr,"NAN geometry_complete false\\n");\n    if(!geometry_complete)for');p.write_text(s)
