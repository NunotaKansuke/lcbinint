import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def jnp():
    return pytest.importorskip("jax.numpy")
