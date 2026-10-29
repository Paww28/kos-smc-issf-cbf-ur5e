"""Safety scenarios: start-pose barriers, and a short offline run per scenario (static fake robot).

Barrier values, filter outputs, h_vals.csv headers, experiment_info.txt lines, Prometheus gauges and
--debug text per scenario are covered by the snapshot tests (tests/test_snapshots.py).
"""

import dataclasses
import os

import jax.numpy as jnp
import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from kossmc_issf.config.naming import SCENARIO_NAMES
from kossmc_issf.config.presets import load_preset
from kossmc_issf.core.safety import SCENARIOS, EEContainmentConfig
from kossmc_issf.datalog.csv_io import h_headers
from kossmc_issf.experiment import Selection, run
from kossmc_issf.monitoring.prometheus import Metrics
from kossmc_issf.plotting.plots import make_plots
from kossmc_issf.sim.pybullet_env import PyBulletEnv

from test_gate import StaticRobot

N_H = {"ee_containment": 6, "wb_obstacle": 13, "wb_containment": 78, "wb_combine": 91}


def test_ee_barrier_at_start_pose():
    """EE position converted to base_link: at the reference start pose all six barriers are >= 0."""
    P = load_preset("pnp")
    cfg = EEContainmentConfig(P.POS_MIN, P.POS_MAX, alpha_gain=P.CBF_ALPHA)
    h = np.array(cfg.h_1(jnp.array(np.array(P.q_init, dtype=float))))
    assert h.shape == (6,) and np.all(h >= 0), h


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_run_scenario_offline(robot, scenario, tmp_path):
    P = dataclasses.replace(load_preset("pnp"), duration=0.4, PNP_SPEED=5.0)
    q = np.array(P.q_init, dtype=float)
    pos, _ = robot.forward_kinematics(q)
    fake = StaticRobot(q, list(pos) + [0.0, 0.0, 0.0])
    env = PyBulletEnv(P.dt, gui=False)
    try:
        out = run(P, Selection("pnp_interpolated", "smc_quasi", "issf", "gaussian", "127.0.0.2", scenario),
                  fake, env, Metrics(registry=CollectorRegistry()), input_fn=lambda p: "y",
                  sleep=lambda s: None, out_root=str(tmp_path), plot_fn=make_plots)
    finally:
        env.disconnect()
    assert out is not None and f"_{SCENARIO_NAMES[scenario]}_ISSf-CBF_" in os.path.basename(out)
    with open(os.path.join(out, "h_vals.csv")) as f:
        header = f.readline().strip().split(",")
    assert header == h_headers(scenario, N_H[scenario])
    with open(os.path.join(out, "experiment_info.txt"), encoding="utf-8") as f:
        assert f"Scenario    : {SCENARIO_NAMES[scenario]}" in f.read()
    assert len(fake.commands) == 50
    assert os.path.exists(os.path.join(out, "12_cbf_analysis.png"))
