"""Additive joint-velocity disturbance."""

import numpy as np

DISTURBANCES = ("none", "gaussian", "sinusoidal")


def disturbance(dist, rng=None, std=0.03, clip=0.05, t=None):
    """Disturbance vector, shape (6,).

    rng: object with `.normal(loc, scale, size)`. None uses the global `np.random`
    (one call per control cycle).
    t: control time (s), used by "sinusoidal" (0.05 rad/s amplitude, 1 Hz, same on all joints).
    """
    if rng is None:
        rng = np.random
    d_input = np.zeros(6)
    if dist == "gaussian":
        d_input = np.clip(rng.normal(0.0, std, 6), -clip, clip)
    elif dist == "sinusoidal":
        d_input = 0.05 * np.sin(2*np.pi*1*t) * np.ones(6)
    elif dist != "none":
        raise ValueError(f"unknown disturbance: {dist!r}")
    return d_input
