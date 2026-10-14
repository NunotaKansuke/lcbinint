from pathlib import Path
root=Path('/rogue1_8/nunota/lcbinint/src/lcbinint/magnification/finite_source_magnifier.cpp')
d=Path('evidence/holonomic/seed_count_v26')
s=(d/'seed_before.cpp').read_text() if (d/'seed_before.cpp').exists() else root.read_text(); (d/'seed_before.cpp').write_text(s)
s=s.replace('bool build_support = false)\n{\n    constexpr std::size_t max_probes', 'bool build_support = false,\n    bool support_only = false)\n{\n    constexpr std::size_t max_probes',1)
s=s.replace('    std::vector<double> segment_length2;', '''    const double xlo = std::nextafter(source.x-source_radius, -std::numeric_limits<double>::infinity());
    const double xhi = std::nextafter(source.x+source_radius, std::numeric_limits<double>::infinity());
    const double ylo = std::nextafter(source.y-source_radius, -std::numeric_limits<double>::infinity());
    const double yhi = std::nextafter(source.y+source_radius, std::numeric_limits<double>::infinity());
    std::vector<double> segment_length2;''', 1)
s=s.replace('if (!is_phantom_segment(i)) {\n                const SourcePosition candidate', '''// For support-only callers, a segment projection is needed only for
            // blind crossings: both endpoints outside, chord bounding box
            // intersects the disk bounding box. This rejects only impossible
            // crossings and retains the historical extremum/probe order.
            const bool possible_blind_crossing =
                build_support && branch.size() >= 3 &&
                distance_squared(p0, source) >= source_radius2 &&
                distance_squared(p1, source) >= source_radius2 &&
                std::min(p0.x,p1.x) <= xhi &&
                std::max(p0.x,p1.x) >= xlo &&
                std::min(p0.y,p1.y) <= yhi &&
                std::max(p0.y,p1.y) >= ylo;
            if ((!support_only || possible_blind_crossing) && !is_phantom_segment(i)) {
                const SourcePosition candidate''',1)
s=s.replace('if (candidate_distance2 < min_distance2) {','if (!support_only && candidate_distance2 < min_distance2) {',1)
s=s.replace('        if (inside) {\n                scan.any_vertex_inside', '        if (!support_only && inside) {\n                scan.any_vertex_inside',1)
s=s.replace('if (inside && !prev_inside && scan.crossing_probes.size()', 'if (!support_only && inside && !prev_inside && scan.crossing_probes.size()',1)
s=s.replace('} else if (inside && source_radius >= 2.0e-2 && i % 20', '} else if (!support_only && inside && source_radius >= 2.0e-2 && i % 20',1)
s=s.replace('*caustic_branches, source, source_radius, true);','*caustic_branches, source, source_radius, true, true);',1)
assert s!=(d/'seed_before.cpp').read_text(); (d/'seed_after.cpp').write_text(s)
