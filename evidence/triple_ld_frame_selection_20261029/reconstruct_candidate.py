from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
s=(P/'finite_source_magnifier.before.cpp').read_text();a=s.index('    const auto value=kernel::integrate<Gradient>(frame.geometry,finite.limb_darkening_c,options,supplied);',s.index('template<bool Gradient> auto triple_algebraic_epoch('));s=s[:a]+s[a:].replace('    const auto value=kernel::integrate<Gradient>(frame.geometry,finite.limb_darkening_c,options,supplied);\n','',1);pos=s.index('    assign(value,frame);',a)
block='''    if constexpr(Gradient) {
        // Crossing LD supports can be much cheaper in another polar frame.
        // Select only from bounded VALUE-only pilots and deterministic work
        // counts. Never reuse their flux, mesh or tangents in the final solve.
        if(finite.limb_darkening_c>0 && accuracy.relative_tolerance>=1e-4 &&
           plan.completed && plan.events.size()>=10) {
            auto pilot_options=options;pilot_options.maximum_refinements=8;
            const auto primal=kernel::integrate<false>(frame.geometry,finite.limb_darkening_c,pilot_options,&plan);
            const bool primal_complete=primal.converged&&primal.support_complete;
            if(!primal_complete || primal.value_evaluations>=192) {
                const auto unlimited=std::numeric_limits<std::size_t>::max();
                const auto original_work=primal_complete?primal.value_evaluations:unlimited;
                auto best_work=original_work;int selected=-1;
                kernel::RadialEventPlan selected_plan;
                const std::array<std::array<int,3>,2> orders{{{1,0,2},{2,0,1}}};
                pilot_options.event_box_budget=std::min(options.event_box_budget,8192);
                for(std::size_t k=0;k<orders.size();++k) {
                    const auto order=orders[k];
                    const std::array<kernel::Complex,3> ordered{lenses[order[0]],lenses[order[1]],lenses[order[2]]};
                    const auto coordinates=kernel::primary_frame<true>(ordered,{source.x,source.y},rho,
                        geometry.masses[order[1]],geometry.masses[order[2]]);
                    auto support=triple_reframed_support_preparation(k).plan(coordinates.geometry,
                        accuracy.relative_tolerance,pilot_options.event_box_budget,false);
                    if(!support.completed)continue;
                    const auto probe=kernel::integrate<false>(coordinates.geometry,finite.limb_darkening_c,pilot_options,&support);
                    if(probe.converged&&probe.support_complete&&probe.value_evaluations>0 &&
                       probe.value_evaluations<best_work &&
                       (!primal_complete || probe.value_evaluations<=original_work/2)) {
                        best_work=probe.value_evaluations;selected=static_cast<int>(k);
                        selected_plan=std::move(support);
                    }
                }
                if(selected>=0) {
                    const auto order=orders[static_cast<std::size_t>(selected)];
                    const std::array<kernel::Complex,3> ordered{lenses[order[0]],lenses[order[1]],lenses[order[2]]};
                    const auto coordinates=kernel::primary_frame<true>(ordered,{source.x,source.y},rho,
                        geometry.masses[order[1]],geometry.masses[order[2]]);
                    auto request=options;request.event_box_budget=pilot_options.event_box_budget;
                    const auto trial=kernel::integrate<true>(coordinates.geometry,finite.limb_darkening_c,request,&selected_plan);
                    if(complete(trial)) {
                        assign(trial,coordinates);
                        out.gradient=kernel::unpermute_physical_gradient(out.gradient,order);
                        out.value.diagnostic_stop+=":primal_work_primary_"+std::to_string(selected+1);
                        return out;
                    }
                }
            }
        }
    }
    const auto value=kernel::integrate<Gradient>(frame.geometry,finite.limb_darkening_c,options,supplied);
''';s=s[:pos]+block+s[pos:];expected=json.loads((P/'candidate-provenance.json').read_text())['source_hashes']['src/lcbinint/magnification/finite_source_magnifier.cpp'];assert hashlib.sha256(s.encode()).hexdigest()==expected
(P/'finite_source_magnifier.candidate.cpp').write_text(s);print(expected)
