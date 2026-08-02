import gc

import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def jnp():
    return pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def release_compiled_executables_between_tests():
    """Keep the many static-shape stress kernels from exhausting XLA memory."""

    yield
    jax.clear_caches()
    gc.collect()
