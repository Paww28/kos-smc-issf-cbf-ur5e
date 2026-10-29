"""Pre-check gate: when a barrier is already negative at the start pose, `Continue? (y/n)` is asked;
any answer other than y stops the run before the control loop (no speedJ, no outputs written).

The box top is lowered (POS_MAX z = 0.3) so the containment barrier is violated at q_init."""

import dataclasses
import os

import numpy as np
import jax.numpy as jnp
from prometheus_client import CollectorRegistry

from kossmc_issf.config.presets import load_preset
from kossmc_issf.core.safety import WholeBodyCombineConfig
from kossmc_issf.experiment import Selection, run
from kossmc_issf.monitoring.prometheus import Metrics
from kossmc_issf.sim.pybullet_env import PyBulletEnv


class StaticRobot:
    """Robot that stays at q (no network)."""

    def __init__(self, q, tcp):
        self.q, self.tcp, self.loop_reads, self.commands = np.array(q), list(tcp), 0, []

    def connect(self): return True
    def get_q(self): return self.q.copy()
    def get_qd(self): return np.zeros(6)
    def get_tcp_pose(self): return self.tcp
    def get_robot_mode(self): return 7
    def get_safety_mode(self): return 1
    def move_j(self, target, speed, accel): pass
    def stop(self): pass
    def disconnect(self): pass

    def read_state(self):
        self.loop_reads += 1
        return self.q.copy(), self.tcp, np.zeros(6)

    def speed_j(self, dq): self.commands.append(dq)


def test_gate_no_stops_before_loop(robot, tmp_path):
    P = load_preset("pnp").with_trajectory("pnp_interpolated")
    P = dataclasses.replace(P, POS_MAX=np.array([0.7, 0.49, 0.3]))
    q = np.array(P.q_init, dtype=float)
    cfg = WholeBodyCombineConfig(robot.manipulator, P.POS_MIN, P.POS_MAX, P.OBS_CENTER, P.OBS_RADIUS,
                                 alpha_gain=P.CBF_ALPHA)
    assert np.min(np.array(cfg.h_1(jnp.array(q)))) < 0, "precondition: gate must be triggered"

    pos, _ = robot.forward_kinematics(q)
    fake = StaticRobot(q, list(pos) + [0.0, 0.0, 0.0])
    prompts = []
    env = PyBulletEnv(P.dt, gui=False)
    try:
        out = run(P, Selection("pnp_interpolated", "p", "none", "none", "127.0.0.2"), fake, env,
                  Metrics(registry=CollectorRegistry()),
                  input_fn=lambda prompt: prompts.append(prompt) or "n",
                  sleep=lambda s: None, out_root=str(tmp_path))
    finally:
        env.disconnect()
    assert out is None
    assert prompts == ["  Continue? (y/n): "]
    assert fake.loop_reads == 0 and fake.commands == []
    written = [f for _, _, files in os.walk(tmp_path) for f in files]
    assert written == []
