"""Plots on synthetic logs: the expected PNG files per scenario / trajectory / disturbance, and every
saved figure contains plotted data."""

import importlib.util
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kossmc_issf.plotting.plots import make_plots

_spec = importlib.util.spec_from_file_location(
    "make_fixtures", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "make_fixtures.py"))
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

COMMON = ["02_tracking_error.png", "02c_tracking_xyz.png", "04b_nullspace_joint.png", "05_vel_compare.png",
          "06_joint_positions.png", "07_singularity.png", "08_sliding_surface.png", "09_phase_plane.png",
          "10_jacobian_rank.png", "10b_control_effort.png", "12_cbf_analysis.png", "13_real_vs_twin.png"]
TRAJ_PNG = {"sinusoidal_3d": ["01_traj_3d.png"], "pnp_interpolated": ["01a_pnp_traj_3d.png", "01b_pnp_traj_2d.png"]}


def expected_pngs(traj, dist):
    return sorted(TRAJ_PNG[traj] + COMMON + (["11_disturbance.png"] if dist != "none" else []))


def _arrays(scenario, traj, dist):
    """A, P, R, dt of a synthetic run (the fixture builder, with the requested options)."""
    run = (scenario, traj, "smc_quasi", "issf", dist)
    runs, mf.RUNS = mf.RUNS, [run]
    try:
        rng = np.random.default_rng(200)
        lists, target, home = mf._synthetic_lists(rng, 120, mf.N_H[scenario], traj == "pnp_interpolated")
        inp = {f"in__0__{k}": np.array(v) for k, v in lists.items()}
        inp["in__0__target"], inp["in__0__home"] = target, home
        return mf.run_arrays(inp, 0)
    finally:
        mf.RUNS = runs


def _has_data(ax):
    if any(np.size(l.get_xdata()) for l in ax.lines):
        return True
    if any(len(c.get_paths()) or np.size(c.get_offsets()) for c in ax.collections):
        return True
    return bool(ax.texts or ax.patches or ax.images)


def _capture(monkeypatch, fn):
    saved = {}

    def fake_savefig(fname, *args, **kwargs):
        name = os.path.basename(fname)
        assert name not in saved, f"{name} saved twice"
        fig = plt.gcf()
        saved[name] = [_has_data(ax) for ax in fig.axes if ax.get_visible()]

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    try:
        fn()
    finally:
        plt.close("all")
    return saved


@pytest.mark.parametrize("dist", ["gaussian", "sinusoidal", "none"])
@pytest.mark.parametrize("traj", ["sinusoidal_3d", "pnp_interpolated"])
@pytest.mark.parametrize("scenario", mf.SCENARIOS)
def test_png_list_and_figures_have_data(scenario, traj, dist, monkeypatch, tmp_path):
    A, P, R, dt = _arrays(scenario, traj, dist)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="No artists with labels found")   # no empty legends
        saved = _capture(monkeypatch, lambda: make_plots(str(tmp_path), A, P, R, dt))
    assert sorted(saved) == expected_pngs(traj, dist)
    for name, axes in saved.items():
        assert axes and all(axes), f"{name}: axes without data {axes}"


def test_png_files_written(tmp_path):
    A, P, R, dt = _arrays("wb_combine", "pnp_interpolated", "gaussian")
    make_plots(str(tmp_path), A, P, R, dt)
    plt.close("all")
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".png")) == expected_pngs("pnp_interpolated", "gaussian")
