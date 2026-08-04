"""Recalibration campaign for the certified finite-source integrator (2026-08).

The 2026-07 calibration was fitted against an integrator that had no
completeness proof, so its resolution rule had to buy safety with resolution.
The component certificate now proves separately that every image component of
the disk was entered, which decouples correctness from grid density and lets
the resolution rule be fitted against measured error alone.

This package holds the harness for refitting that rule, the method-selection
boundaries around it, and the accuracy-versus-time comparison against
VBMicrolensing, the differentiable JAX path, and microLUX.
"""
