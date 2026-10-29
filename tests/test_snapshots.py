"""Snapshot tests: kossmc_issf outputs == tests/fixtures/*.npz (made by tests/fixtures/make_fixtures.py).

The outputs are recomputed from the stored inputs. NumPy outputs atol 1e-12, JAX float64 outputs
atol 1e-10, discrete outputs (int, bool, str) identical.
"""

import importlib.util
import os

import numpy as np
import pytest

from conftest import ATOL_JAX, ATOL_NUMPY

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

_spec = importlib.util.spec_from_file_location("make_fixtures", os.path.join(FIXTURES, "make_fixtures.py"))
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def _load(group):
    with np.load(os.path.join(FIXTURES, f"{group}.npz")) as z:
        return {k: z[k] for k in z.files}


def _compare(key, actual, expected):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, f"{key}: shape {actual.shape} != {expected.shape}"
    if key.startswith("exact__"):
        if expected.dtype.kind == "U":
            for i, (a, e) in enumerate(zip(actual.ravel(), expected.ravel())):
                assert a == e, f"{key}[{i}]: text differs"
        else:
            assert actual.dtype.kind == expected.dtype.kind, f"{key}: dtype {actual.dtype} != {expected.dtype}"
            assert np.array_equal(actual, expected), f"{key}: values differ"
    else:
        atol = ATOL_JAX if key.startswith("jax__") else ATOL_NUMPY
        assert key.startswith(("np__", "jax__")), key
        np.testing.assert_allclose(actual.astype(float), expected, rtol=0, atol=atol, err_msg=key)


@pytest.mark.parametrize("group", mf.GROUPS)
def test_snapshot(group):
    stored = _load(group)
    inputs = {k: v for k, v in stored.items() if k.startswith("in__")}
    expected = {k: v for k, v in stored.items() if not k.startswith("in__")}
    actual = getattr(mf, f"compute_{group}")(inputs)
    assert sorted(actual) == sorted(expected)
    for key in expected:
        _compare(key, actual[key], expected[key])


def test_inputs_reproducible():
    """The stored inputs are those the fixture script generates (seeded)."""
    for group in mf.GROUPS:
        stored = _load(group)
        for key, value in getattr(mf, f"inputs_{group}")().items():
            assert np.array_equal(np.asarray(value), stored[key]), f"{group}.{key}"


def test_fixture_size():
    total = sum(os.path.getsize(os.path.join(FIXTURES, f"{g}.npz")) for g in mf.GROUPS)
    assert total < 1024 * 1024
