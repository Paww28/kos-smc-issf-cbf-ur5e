"""Experiment parameter presets (kossmc_issf/config/presets/*.yaml)."""

import os
from dataclasses import dataclass, replace
from typing import Any, List, Optional

import numpy as np
import yaml

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
PRESETS = ("pnp", "circular")
CLI_PRESETS = PRESETS

_KEYS = ("trajectory", "dt", "duration", "q_init", "Kp", "K_null", "K_jl", "q_min", "q_max", "q_mid",
         "EPS", "Lambda", "k_smc", "eps_qsmc", "amplitude", "frequency_traj", "POS_MIN", "POS_MAX",
         "CBF_ALPHA", "ISSF_EPSILON", "OBS_CENTER", "OBS_RADIUS", "PNP_SPEED", "PNP_THRESH",
         "pnp_waypoints", "DIST_STD", "DIST_CLIP")


@dataclass(frozen=True)
class Params:
    """Experiment parameters, as given in the preset YAML file."""
    name: str
    trajectory: Optional[str]
    dt: float
    duration: Any
    q_init: List[float]
    Kp: Any
    K_null: Any
    K_jl: Any
    q_min: np.ndarray
    q_max: np.ndarray
    q_mid: np.ndarray
    EPS: float
    Lambda: np.ndarray
    k_smc: Any
    eps_qsmc: Any
    amplitude: Any
    frequency_traj: Any
    POS_MIN: np.ndarray
    POS_MAX: np.ndarray
    CBF_ALPHA: Any
    ISSF_EPSILON: Any
    OBS_CENTER: np.ndarray
    OBS_RADIUS: Any
    PNP_SPEED: Any
    PNP_THRESH: Any
    pnp_waypoints: List[np.ndarray]
    DIST_STD: Any
    DIST_CLIP: Any

    def with_trajectory(self, trajectory):
        return replace(self, trajectory=trajectory)


def load_raw(name):
    """The preset file as plain YAML values."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name!r}")
    with open(os.path.join(PRESET_DIR, f"{name}.yaml")) as f:
        raw = yaml.safe_load(f)
    if set(raw) != set(_KEYS):
        raise ValueError(f"{name}: keys {sorted(set(raw) ^ set(_KEYS))} missing or unexpected")
    return raw


def load_preset(name):
    raw = load_raw(name)
    values = dict(raw)
    for k in ("q_min", "q_max", "q_mid", "POS_MIN", "POS_MAX", "OBS_CENTER"):
        values[k] = np.array(raw[k])
    values["Lambda"] = np.diag(raw["Lambda"])
    values["q_init"] = list(raw["q_init"])
    values["pnp_waypoints"] = [np.array(w) for w in raw["pnp_waypoints"]]
    return Params(name=name, **values)
