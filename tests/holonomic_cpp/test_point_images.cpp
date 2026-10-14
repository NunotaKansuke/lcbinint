// ATPT holonomic solver -- binary point-source image solver correctness.
//
// Checks on binary_point_images (point_images.hpp), phase A:
//   1. image count is odd (3 or 5) for every non-degenerate config;
//   2. lens-equation residual < 1e-9 for every certified image;
//   3. signed-magnification sum theorem  sum_i sign(detJ_i) * mag_i == 1
//      -- ONLY for the maximal (5-image) configs.  For 3-image configs the
//      two "missing" images are complex and the real-image signed sum is not
//      an invariant (it ranges well away from 1); do not test it there.
//   4. phi_lens(|z|, arg z) == 1 at every image (g = f(z) - zeta = 0 there);
//   5. independent cross-check: a from-scratch Newton grid search over the
//      image plane finds exactly the same image set (count + positions).
//
// Pure C++, no reference TSV.  Exit non-zero on any failure.

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/point_images.hpp"

using namespace lcbinint::holonomic;
using cd = std::complex<double>;

static int failures = 0;
#define CHECK(cond, msg, ...)                                       \
    do {                                                           \
        if (!(cond)) {                                             \
            std::fprintf(stderr, "FAIL: " msg "\n", ##__VA_ARGS__); \
            ++failures;                                            \
        }                                                          \
    } while (0)

// Independent brute-force image finder: Newton on the non-holomorphic lens
// equation from a coarse grid of seeds, deduplicated.  Shares no code with
// binary_point_images (no degree-5 polynomial).
static std::vector<cd> brute_images(const PrimaryFrame& pf) {
    const double m0 = pf.m0, m1 = 1.0 - pf.m0, a = pf.a;
    const cd w(pf.X, pf.Y);
    auto residual = [&](cd z) {
        return std::abs(w - (z - m0 / std::conj(z) - m1 / (std::conj(z) - a)));
    };
    // Coarse plane grid plus dense rings around each lens: faint images hug the
    // lens positions and a plane grid alone can miss them for extreme mass
    // ratios.
    std::vector<cd> seeds;
    for (double x = -6.0; x <= 6.0001; x += 0.05)
        for (double y = -6.0; y <= 6.0001; y += 0.05)
            seeds.emplace_back(x, y);
    for (cd c : {cd(0.0, 0.0), cd(a, 0.0)})
        for (double rr : {1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1})
            for (int k = 0; k < 24; ++k)
                seeds.push_back(c + std::polar(rr, 2.0 * M_PI * k / 24));

    std::vector<cd> found;
    for (cd z0 : seeds) {
            cd z = z0;
            bool ok = true;
            for (int it = 0; it < 80; ++it) {
                cd zb = std::conj(z), zba = zb - a;
                if (std::abs(zb) < 1e-13 || std::abs(zba) < 1e-13) { ok = false; break; }
                cd g = w - z + m0 / zb + m1 / zba;
                cd dgdzb = -m0 / (zb * zb) - m1 / (zba * zba);
                cd gx = cd(-1, 0) + dgdzb;
                cd gy = cd(0, -1) + cd(0, -1) * dgdzb;
                double J00 = gx.real(), J01 = gy.real(), J10 = gx.imag(), J11 = gy.imag();
                double det = J00 * J11 - J01 * J10;
                if (std::abs(det) < 1e-15) { ok = false; break; }
                double dx = (-g.real() * J11 + g.imag() * J01) / det;
                double dy = (-g.imag() * J00 + g.real() * J10) / det;
                z += cd(dx, dy);
                if (std::hypot(dx, dy) < 1e-14) break;
            }
            if (!ok || residual(z) > 1e-10) continue;
            bool dup = false;
            for (auto& f : found)
                if (std::abs(f - z) < 1e-6) { dup = true; break; }
            if (!dup) found.push_back(z);
        }
    return found;
}

int main() {
    struct Cfg { double xs, ys, q, a; const char* name; };
    // wide, close, resonant; on-axis and off; extreme and moderate q
    std::vector<Cfg> cfgs = {
        {0.2, 0.10, 0.5, 1.2, "wide-mod"},
        {0.05, 0.03, 0.3, 0.7, "close-mod"},
        {0.0, 0.15, 1.0, 1.0, "resonant-eq"},
        {0.3, 0.0, 0.1, 1.5, "wide-onaxis-lowq"},
        {-0.1, 0.25, 0.8, 0.9, "resonant-offaxis"},
        {0.5, 0.4, 0.01, 2.0, "wide-extremeq"},
        {0.02, 0.01, 0.6, 0.4, "veryclose"},
        {0.8, 0.6, 0.5, 1.2, "outside-mod"},
        {0.15, 0.0, 1.0, 0.6, "close-eq-onaxis"},
        {1.5, 0.2, 0.2, 2.5, "far"},
    };

    int total_imgs = 0;
    for (const auto& cf : cfgs) {
        LensParams p{cf.xs, cf.ys, 1e-3, cf.q, cf.a, /*barycentric=*/false};
        PrimaryFrame pf = PrimaryFrame::from(p);
        PointImages im = binary_point_images(pf);

        int n = (int)im.images.size();
        total_imgs += n;
        CHECK(n == 3 || n == 5, "[%s] image count %d not in {3,5}", cf.name, n);
        CHECK(im.reliable, "[%s] not reliable (n_raw=%d)", cf.name, im.n_raw);

        double signed_sum = 0.0, worst_res = 0.0, worst_phi = 0.0;
        for (const auto& g : im.images) {
            signed_sum += (g.positive_parity ? 1.0 : -1.0) * g.mag;
            worst_res = std::max(worst_res, g.residual);
            double th = std::atan2(g.z.im, g.z.re);
            double phi = phi_lens(g.radius, th, pf);
            worst_phi = std::max(worst_phi, std::fabs(phi - 1.0));
        }
        CHECK(worst_res < 1e-9, "[%s] worst residual %.3e", cf.name, worst_res);
        CHECK(worst_phi < 1e-7, "[%s] worst |phi-1| at image %.3e", cf.name,
              worst_phi);
        if (n == 5)
            CHECK(std::fabs(signed_sum - 1.0) < 1e-6,
                  "[%s] 5-image signed-mag sum %.9f != 1", cf.name, signed_sum);

        // independent cross-check
        std::vector<cd> bf = brute_images(pf);
        CHECK((int)bf.size() == n,
              "[%s] brute-force finds %zu images, solver finds %d", cf.name,
              bf.size(), n);
        double worst_match = 0.0;
        for (const auto& g : im.images) {
            double best = 1e9;
            for (const auto& b : bf)
                best = std::min(best, std::hypot(b.real() - g.z.re,
                                                 b.imag() - g.z.im));
            worst_match = std::max(worst_match, best);
        }
        CHECK(worst_match < 1e-6,
              "[%s] solver image not matched by brute force (dist %.3e)",
              cf.name, worst_match);

        std::fprintf(stderr,
                     "  %-20s n=%d  mu_total=%9.4f  signed_sum=%+.6f  "
                     "res<%.1e  |phi-1|<%.1e  match<%.1e\n",
                     cf.name, n, im.mu_total, signed_sum, worst_res, worst_phi,
                     worst_match);
    }

    std::fprintf(stderr, "\ntotal images across %zu configs: %d\n", cfgs.size(),
                 total_imgs);
    if (failures) {
        std::fprintf(stderr, "\n%d CHECK failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "\nall point-image checks passed\n");
    return 0;
}
