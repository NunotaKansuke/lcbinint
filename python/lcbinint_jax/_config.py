"""Runtime requirements shared by the experimental JAX kernels."""

import jax


def require_x64() -> None:
    """Raise with an actionable message when JAX 64-bit mode is disabled."""

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "lcbinint_jax requires 64-bit JAX. Call "
            "jax.config.update('jax_enable_x64', True) before evaluation."
        )
