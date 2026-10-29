"""Presets: values == the reference parameter table; loaded types."""

import numpy as np
import pytest

from kossmc_issf.config import defaults
from kossmc_issf.config.presets import PRESETS, load_preset, load_raw

# Reference parameters (README "Reference parameters"); trajectory and duration per preset.
REFERENCE_TABLE = {
    "dt": 1/125,
    "q_init": [0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0],
    "Kp": 1.5,
    "K_null": 0.9,
    "K_jl": 5,
    "q_min": [-2*np.pi + 10] * 6,     # q_min > q_max (see the preset files)
    "q_max": [2*np.pi - 10] * 6,
    "q_mid": [0.0] * 6,
    "EPS": 1e-3,
    "Lambda": [5, 5, 5],
    "k_smc": 0.15,
    "eps_qsmc": 0.9,
    "amplitude": 0.23,
    "frequency_traj": 0.05,
    "POS_MIN": [-0.25, -0.49, 0],
    "POS_MAX": [0.7, 0.49, 0.7],
    "CBF_ALPHA": 10,
    "ISSF_EPSILON": 0.5,
    "OBS_CENTER": [0.53, -0.1, 0.49],
    "OBS_RADIUS": 0.055,
    "PNP_SPEED": 0.055,
    "PNP_THRESH": 2.0,
    "pnp_waypoints": [
        [-0.4942, -0.1353, 0.4853],
        [-0.49, -0.6, 0.49],
        [-0.49, -0.6, 0.2],
        [-0.49, -0.6, 0.49],
        [-0.49, 0.55, 0.49],
        [-0.1, 0.36, 0.49],
        [-0.1, 0.36, 0.2],
        [-0.1, 0.36, 0.49],
    ],
    "DIST_STD": 0.03,                  # np.clip(np.random.normal(0.0, 0.03, 6), -0.05, 0.05)
    "DIST_CLIP": 0.05,
}
PRESET_SPECIFIC = {
    "pnp": {"trajectory": "pnp_interpolated", "duration": 90},
    "circular": {"trajectory": "sinusoidal_3d", "duration": 30},
}

def _same(a, b):
    """Equal value AND equal Python/NumPy scalar kind (the YAML values are kept exactly)."""
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    return a == b and isinstance(a, int) == isinstance(b, int)


@pytest.mark.parametrize("name", ["pnp", "circular"])
def test_presets_match_reference_table(name):
    raw = load_raw(name)
    expected = dict(REFERENCE_TABLE, **PRESET_SPECIFIC[name])
    assert set(raw) == set(expected)
    for k, v in expected.items():
        assert _same(raw[k], v), f"{name}.{k}: {raw[k]!r} != {v!r}"


def test_default_ip_is_ursim():
    assert defaults.URSIM_IP == "127.0.0.2"
    from kossmc_issf.config.cli import build_parser
    assert build_parser().parse_args(["--preset", "pnp"]).ip == "127.0.0.2"


@pytest.mark.parametrize("name", PRESETS)
def test_loaded_types(name):
    P = load_preset(name)
    np.testing.assert_array_equal(P.Lambda, np.diag([5, 5, 5]))
    assert P.Lambda.dtype == np.diag([5, 5, 5]).dtype
    assert P.POS_MIN.dtype == np.array([-0.25, -0.49, 0]).dtype
    assert P.q_mid.dtype == np.zeros(6).dtype
    assert isinstance(P.q_init, list) and all(isinstance(w, np.ndarray) for w in P.pnp_waypoints)
    assert P.dt == 1/125
