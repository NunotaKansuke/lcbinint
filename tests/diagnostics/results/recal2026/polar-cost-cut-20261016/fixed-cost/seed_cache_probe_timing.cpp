#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace lcbinint;
using namespace lcbinint::magnification;
using Clock = std::chrono::steady_clock;

struct Input {
    int id = -1;
    SourcePosition source{};
    std::vector<SourcePosition> centers;
    std::vector<SourcePosition> expected_seeds;
    bool expected_support_proven = false;
};

bool same_seeds(const std::vector<SourcePosition>& a,
                const std::vector<SourcePosition>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].x != b[i].x || a[i].y != b[i].y) return false;
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "usage: seed_cache_probe INPUT.tsv CAPACITY OPERATIONS\n";
        return 2;
    }
    const std::size_t capacity = static_cast<std::size_t>(std::stoul(argv[2]));
    const std::size_t operations = static_cast<std::size_t>(std::stoull(argv[3]));
    std::ifstream input(argv[1]);
    std::vector<Input> rows;
    int id = 0;
    double x = 0.0, y = 0.0;
    while (input >> id >> x >> y) {
        rows.push_back({id, {x, y}, {}});
        if (rows.size() == capacity) break;
    }
    if (!input.eof() && input.fail()) input.clear();
    if (rows.size() != capacity || operations < 1) {
        std::cerr << "invalid input size or operation count\n";
        return 2;
    }

    constexpr double separation = 1.137;
    constexpr double mass_ratio = 0.073;
    constexpr double source_radius = 0.001;
    FiniteSourceSettings settings;
    settings.caustic_bins = 1400;
    settings.source_bins = settings.polar_source_bins = 2;
    settings.finite_mode = 2;
    settings.limb_darkening_c = 0.5;
    const FiniteSourceMagnifier magnifier(settings);
    const PointSourceMagnifier point;
    double checksum = 0.0;

    for (auto& row : rows) {
        const auto images = point.binary_images(separation, mass_ratio, row.source);
        row.centers.reserve(images.size());
        for (const auto& image : images) row.centers.push_back(image.position);
        bool proven = false;
        const auto seeds = magnifier.cached_binary_image_seeds(
            point, separation, mass_ratio, row.source, source_radius,
            std::numeric_limits<double>::infinity(), &row.centers, &proven);
        row.expected_seeds = seeds;
        row.expected_support_proven = proven;
        checksum += static_cast<double>(seeds.size()) + (proven ? 0.25 : 0.0);
    }

    const auto measure = [&](const char* name, auto&& body) {
        const auto begin = Clock::now();
        for (std::size_t operation = 0; operation < operations; ++operation) body(operation);
        const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        const double calls = static_cast<double>(operations);
        std::cout << name << '\t' << std::setprecision(12) << seconds * 1.0e9 / calls
                  << '\t' << checksum << '\n';
    };
    const auto hit = [&](std::size_t index) {
        bool proven = false;
        const auto seeds = magnifier.cached_binary_image_seeds(
            point, separation, mass_ratio, rows[index].source, source_radius,
            std::numeric_limits<double>::infinity(), &rows[index].centers, &proven);
        checksum += static_cast<double>(seeds.size()) + (proven ? 0.25 : 0.0);
    };

    const std::size_t retained_begin = capacity > 1024 ? capacity - 1024 : 0;
    const std::size_t retained_size = capacity - retained_begin;
    for (std::size_t i = retained_begin; i < capacity; ++i) {
        bool proven = false;
        const auto seeds = magnifier.cached_binary_image_seeds(
            point, separation, mass_ratio, rows[i].source, source_radius,
            std::numeric_limits<double>::infinity(), &rows[i].centers, &proven);
        if (!same_seeds(seeds, rows[i].expected_seeds) ||
            proven != rows[i].expected_support_proven) {
            std::cerr << "seed cache identity mismatch for id=" << rows[i].id << '\n';
            return 3;
        }
    }
    measure("oldest_hit", [&](std::size_t) { hit(retained_begin); });
    measure("newest_hit", [&](std::size_t) { hit(capacity - 1); });
    measure("cyclic_hits", [&](std::size_t operation) {
        hit(retained_begin + operation % retained_size);
    });
    return 0;
}
