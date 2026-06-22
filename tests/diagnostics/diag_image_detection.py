"""Debug image detection at the problematic point."""
import numpy as np
import math
from VBBinaryLensing import VBBinaryLensing

# Parameters for wide caustic case at t=0.006
# separation=0.95, mass_ratio=0.01, u0=-0.01, alpha=0.5, rho=0.01

# At time t in a standard microlensing model:
# u(t) = sqrt(u0^2 + (t/tE)^2)
# source position moves along a line at angle alpha

# For our case: u0=-0.01 (negative y-direction), alpha=0.5 rad, t=0.006, tE=1.0 (assumed)
t = 0.006
u0 = -0.01
alpha = 0.5
tE = 1.0  # assumed based on the geometry

# Compute source position
u_mag = np.sqrt(u0**2 + (t/tE)**2)
# Source position: (u_mag * cos(alpha), u0 + u_mag * sin(alpha))
# Actually, standard form: y = u0, x = u0*cot(alpha) + t/tE (along the trajectory)

# But let's just use VBBL's coordinate system directly
# In VBBL: the 5 parameters are [log(s), log(q), y, alpha, log(rho)]
# The light curve is computed for specific (y, alpha) at each time

vbb = VBBinaryLensing()
vbb.Tol = 1e-3
vbb.RelTol = 0.0

separation = 0.95
mass_ratio = 0.01
rho = 0.01

# Parameters at t=0.006
params = [
    math.log(separation),
    math.log(mass_ratio),
    u0,  # y coordinate (constant in this model, or changes with time?)
    alpha,  # angle
    math.log(rho),
    0.0, 0.0,  # parallax
]

# Actually, looking at the Case definition, u0 is the baseline impact parameter
# and it moves in time. Let me check what the actual source position is.

# From the user's setup: u0=-0.01, alpha=0.5, so at t=0, source is at (0, -0.01) in source plane
# At time t, source moves by t along direction alpha
# So at t=0.006: source = (-0.01, 0) rotated by alpha and shifted

# Let me compute properly:
u_t = np.sqrt(u0**2 + (t/tE)**2)  # impact parameter at time t

# VBBL expects parameters in image plane coordinates
# y is the baseline perpendicular separation
# alpha is the angle of lens orientation

print(f"Time t={t}")
print(f"Impact parameter u(t)={u_t:.6f}")
print()

# Get point-source magnification to understand the image structure
# For VBBL, we need to use the full light curve evaluation
mag = vbb.BinaryLightCurve(params, [t])[0][0]
print(f"Point-source magnification (ρ→0): {mag:.6f}")
print()

# For finite source with rho=0.01, VBBL gives
mag_fs = vbb.BinaryLightCurve(params, [t])[0][0]
print(f"Finite-source magnification (ρ=0.01): {mag_fs:.6f}")
print()

# The issue is likely that for large finite sources at high magnifications,
# the image structure is complex. Let me check point-source magnifications
# at nearby times to understand the caustic structure

print("Point-source magnifications near the peak:")
times_nearby = np.linspace(-0.01, 0.02, 31)
for t_nearby in times_nearby:
    mag_ps = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(1e-6), 0.0, 0.0],
        [t_nearby])[0][0]
    if abs(t_nearby - 0.006) < 0.001 or abs(t_nearby - 0.002) < 0.001:
        print(f"  t={t_nearby:8.5f}: mag={mag_ps:10.4f} {'<-- problematic' if abs(t_nearby - 0.006) < 0.0005 else ''}")
    else:
        print(f"  t={t_nearby:8.5f}: mag={mag_ps:10.4f}")
