#pragma once

// ATPT holonomic solver -- floating-point environment guard.
//
// The boundary-quartic Aberth iteration converges onto near-exact roots,
// so its intermediate residuals |p|, Newton steps |w|, and pair gaps
// |z_i - z_j| near a tangency are driven deep into the gradual-underflow
// range.  On x86 a denormal SSE operand costs a ~50-100x microcode
// penalty per hit, which dominates the warm-started root solve (its seeds
// start already near the roots) and inflates the cold solve too.
//
// Every quantity the solver reports is astronomically larger than the
// 2.2e-308 normal-double floor -- mu ~ 1e0..1e2, its gradient ~ 1e0..1e3,
// the smallest resolved arc gap in theta ~ 1e-11 -- so flushing sub-normal
// *intermediates* to zero is numerically inert here.  The __float128 D14
// radial-event path is software (libquadmath) and is unaffected by MXCSR.
//
// Scope it with an RAII guard at the public entry points only; it saves
// and restores MXCSR, so nesting (epoch_jacobian -> flux_jacobian) is safe
// and the caller's FP environment is untouched on return.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <pmmintrin.h>
#include <xmmintrin.h>
#define LCBININT_HOLONOMIC_HAVE_MXCSR 1
#endif

namespace lcbinint::holonomic {

class ScopedFlushDenormals {
#ifdef LCBININT_HOLONOMIC_HAVE_MXCSR
 public:
    ScopedFlushDenormals() : saved_(_mm_getcsr()) {
        // FTZ (bit 15) flushes denormal results; DAZ (bit 6) treats
        // denormal inputs as zero.
        _mm_setcsr(saved_ | 0x8040u);
    }
    ~ScopedFlushDenormals() { _mm_setcsr(saved_); }

 private:
    unsigned int saved_;
#else
 public:
    ScopedFlushDenormals() = default;
#endif

    ScopedFlushDenormals(const ScopedFlushDenormals&) = delete;
    ScopedFlushDenormals& operator=(const ScopedFlushDenormals&) = delete;
};

}  // namespace lcbinint::holonomic
