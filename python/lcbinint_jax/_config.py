"""Runtime requirements shared by the experimental JAX kernels."""

import jax
import jax.numpy as jnp
from jax.interpreters import ad as jax_ad


def require_x64() -> None:
    """Raise with an actionable message when JAX 64-bit mode is disabled."""

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "lcbinint_jax requires 64-bit JAX. Call "
            "jax.config.update('jax_enable_x64', True) before evaluation."
        )


def as_float64(value):
    """Convert a numerical public-kernel input to the supported precision.

    Enabling ``jax_enable_x64`` merely permits 64-bit arrays; it does not
    promote user-provided ``float32`` leaves.  The native FFI already coerces
    its scalar ABI to ``float64``, so pure-JAX entry points must do the same
    rather than silently selecting a different numerical kernel.
    """

    return jnp.asarray(value, dtype=jnp.float64)


def reject_higher_order_ad(primals) -> None:
    """Fail before a first-order-only custom rule returns a false derivative.

    JAX does not expose a public transform-level predicate.  ``JVPTracer`` is
    imported through JAX's compatibility-facing ``jax.interpreters`` module
    and kept behind this helper so a future JAX compatibility update is local.
    """

    if any(
        isinstance(value, jax_ad.JVPTracer)
        for value in jax.tree_util.tree_leaves(primals)
    ):
        raise NotImplementedError(
            "second and higher derivatives through lcbinint_jax custom "
            "derivative rules are unsupported; use first-order "
            "jax.jvp/jax.grad only"
        )
