#include "lcbinint/magnification/component_certificate.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using lcbinint::SourcePosition;
using namespace lcbinint::magnification;

// An independently known closed caustic analogue: a circle separates the
// three-image exterior from the five-image interior. In particular, a tiny
// circle wholly inside the source must not be skipped by source-sized steps.
int main() {
    try {
        for (double radius : {0.1, 1e-3, 1e-5, 1e-7}) {
            const SourcePosition centre{0.25, 0.13}, source{0.0, 0.0};
            std::vector<std::vector<SourcePosition>> branches(1);
            for (int i = 0; i < 1400; ++i) {
                const double angle = 6.2831853071795864769 * i / 1400;
                branches[0].push_back({centre.x + radius * std::cos(angle),
                                       centre.y + radius * std::sin(angle)});
            }
            auto support = certify_disk_support(branches, source, 1.0);
            const auto images = [&](SourcePosition position) {
                const double dx = position.x - centre.x;
                const double dy = position.y - centre.y;
                return dx * dx + dy * dy < radius * radius ? 5 : 3;
            };
            int solves = 0;
            const bool proven = resolve_certified_probes(support, [&](SourcePosition position) {
                ++solves;
                return images(position);
            });
            if (!proven) throw std::runtime_error("missed analytic interior");
            if (solves > static_cast<int>(support.probes.size() + 24 * support.extrema.size()))
                throw std::runtime_error("unbounded retry");
            if (radius == 0.1 && solves > static_cast<int>(support.probes.size()))
                throw std::runtime_error("ordinary case exceeded offered work");
            if (resolve_certified_probes(support, [](SourcePosition) { return 3; }))
                throw std::runtime_error("accepted absent transition");
            if (resolve_certified_probes(support, [](SourcePosition) { return -1; }))
                throw std::runtime_error("accepted unusable probes");
            if (radius <= 1e-5) {
                auto shortened = support;
                shortened.probes.clear();
                int count = 0, previous_extremum = -1;
                ProbeDirection previous_direction = ProbeDirection::normal_plus;
                for (const auto& candidate : support.probes) {
                    if (candidate.extremum != previous_extremum ||
                        candidate.direction != previous_direction) count = 0;
                    previous_extremum = candidate.extremum;
                    previous_direction = candidate.direction;
                    if (count++ < 4) shortened.probes.push_back(candidate);
                }
                if (resolve_certified_probes(shortened, images))
                    throw std::runtime_error("ignored shortened diagnostic ladder");
            }
            support.unresolved = true;
            if (resolve_certified_probes(support, images))
                throw std::runtime_error("accepted unresolved descriptor");
            std::cout << "radius=" << radius << " extrema=" << support.extrema.size()
                      << " offered=" << support.probes.size() << " solved=" << solves
                      << " PASS\n";
        }
        std::cout << "analytic support checks PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
