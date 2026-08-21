// Small C ABI shim for the installed VBMicrolensing library.
//
// The Python bindings expose minannuli but not the final adaptive nannuli
// chosen by BinaryMagDark.  This shim keeps the VBM algorithm and settings
// identical while making that diagnostic value available to the benchmark.

#include "VBMicrolensingLibrary.h"

extern "C" {

VBMicrolensing* vbm_nannuli_create(double reltol, double limb_c) {
    VBMicrolensing* vbm = new VBMicrolensing();
    vbm->Tol = 1.0e-12;
    vbm->RelTol = reltol;
    vbm->a1 = limb_c;
    vbm->a2 = 0.0;
    vbm->SetLDprofile(VBMicrolensing::LDlinear);
    return vbm;
}

void vbm_nannuli_destroy(VBMicrolensing* vbm) {
    delete vbm;
}

double vbm_nannuli_binary_mag_dark(
    VBMicrolensing* vbm,
    double separation,
    double mass_ratio,
    double x,
    double y,
    double source_radius,
    double absolute_tolerance,
    int* nannuli
) {
    const double value = vbm->BinaryMagDark(
        separation,
        mass_ratio,
        -x,
        y,
        source_radius,
        absolute_tolerance
    );
    if (nannuli != nullptr) {
        *nannuli = vbm->nannuli;
    }
    return value;
}

}
