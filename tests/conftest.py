import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import kossmc_issf  # noqa: F401  (imports oscbf first: jax_enable_x64)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOL_NUMPY = 1e-12
ATOL_JAX = 1e-10


@pytest.fixture(scope="session")
def robot():
    from kossmc_issf.core.kinematics import get_robot_model
    return get_robot_model()
