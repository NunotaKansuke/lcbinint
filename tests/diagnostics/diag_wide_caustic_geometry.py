"""Analyze caustic geometry at problematic time points."""
import numpy as np
import math
from VBMicrolensing import VBMicrolensing
from adaptive_source_bins_sweep import Case

CASE = Case(
    name="wide caustic finite source",
    separation=0.95,
    mass_ratio=0.01,
    u0=-0.01,
    alpha=0.5,
    rho=1.0e-2,
    t_min=-0.8,
    t_max=0.8,
    n_times=400,
)

# Get image positions at different times
vbb = VBMicrolensing()
vbb.Tol = 1e-3
vbb.RelTol = 0.0

def get_images_at_time(t):
    """Get all point source images at time t."""
    # u0 changes with time
    u_t = np.sqrt(CASE.u0**2 + (t / CASE.tE)**2) if hasattr(CASE, 'tE') else CASE.u0
    # Actually, from the user's setup: u0=-0.01, alpha=0.5, tE=1.0 (assumed)
    # Let me recalculate: at time t, we have motion along a line
    # For now just use the geometry to estimate images

    # Better: use vbm's internal calculation by examining the light curve
    params = [
        math.log(CASE.separation),
        math.log(CASE.mass_ratio),
        CASE.u0,  # impact param in y
        CASE.alpha,  # angle
        math.log(CASE.rho),
        0.0,  # parallax terms
        0.0,
    ]

    # We can't directly get image positions from vbm easily
    # but we can use the magnifications to infer the image structure
    mag = vbb.BinaryLightCurve(params, [t])[0][0]
    return mag

# Find times where magnifications differ most
times = np.linspace(CASE.t_min, CASE.t_max, CASE.n_times)
mags = np.array([get_images_at_time(t) for t in times])

# Find peak and nearby problematic region
peak_idx = np.argmax(mags)
print(f"Peak at t={times[peak_idx]:.6f}, mag={mags[peak_idx]:.4f}")
print()

# Analyze the magnification curve around peak
win = 50
region = slice(max(0, peak_idx - win), min(len(times), peak_idx + win + 1))
t_region = times[region]
mag_region = mags[region]

print("Magnification around peak:")
for i, (t, mag) in enumerate(zip(t_region, mag_region)):
    if i % 10 == 0 or abs(t - times[peak_idx]) < 0.01:
        print(f"  t={t:8.5f}: mag={mag:10.4f}")

print()
print("Key observation:")
print(f"  At peak (t≈{times[peak_idx]:.4f}): vbm mag ≈ {mags[peak_idx]:.4f}")
print(f"  At t=0.006: vbm mag ≈ {mags[np.argmin(np.abs(times - 0.006))]:.4f}")
print(f"  At t=0.010: vbm mag ≈ {mags[np.argmin(np.abs(times - 0.010))]:.4f}")
print()

# The issue is likely that lcbinint is missing some images or has
# incorrect magnification calculation for large finite sources
# at these specific points

print("Hypothesis: Large source (rho=0.01) with wide caustic geometry")
print("At caustic crossings, the source disk straddles multiple caustic branches")
print("lcbinint's Phase 1 (1400 samples) may not be sampling finely enough to detect all")
print("caustic-source disk interactions, especially near the peak where magnification changes rapidly.")
