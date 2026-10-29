"""Minimal default terminal output; the troubleshooting text only with --debug.

A short offline run (static fake robot, 2 s, fast PnP so that phase transitions happen) is captured
in default and in debug mode.
"""

import dataclasses
import re

import numpy as np
from prometheus_client import CollectorRegistry

from kossmc_issf import console
from kossmc_issf.config.presets import load_preset
from kossmc_issf.experiment import Selection, run
from kossmc_issf.monitoring.prometheus import Metrics
from kossmc_issf.robot.rtde_interface import report_status
from kossmc_issf.sim.pybullet_env import PyBulletEnv
from kossmc_issf.trajectory.pnp import PNP_NAMES

from test_gate import StaticRobot

NUM = r"-?\d+\.\d{4}"
PHASES = "|".join(re.escape(n) for n in PNP_NAMES)
# The only lines printed without --debug.
DEFAULT_LINES = {
    "rtde":      re.compile(r"^\[RTDE\] Connected \| robot [A-Z_]+ \(-?\d+\) \| safety [A-Z_]+ \(\d+\)$"),
    "init":      re.compile(rf"^\[INIT\] q error \d+\.\d{{4}} rad \| TCP X={NUM} Y={NUM} Z={NUM}$"),
    "precheck":  re.compile(rf"^\[PRE-CHECK\] h_min {NUM}$"),
    "pnp":       re.compile(rf"^\[PNP\] -> ({PHASES})$"),
    "singular":  re.compile(r"^\[SINGULAR\] -> (deactivated|reactivated) \(σ_min=-?\d\.\d{4}e[+-]\d+\)$"),
    "status":    re.compile(rf"^\[\d+\.\d{{2}}s\] \d+/\d+ \| err \d+\.\d{{2}} mm \| h_min {NUM}$"),
    "done":      re.compile(r"^\[DONE\] .+$"),
}
# Troubleshooting text that must appear with --debug and never without it.
DEBUG_ONLY = [
    "=== PARAMETERS ===", "[FOLDER]", "BEFORE moveJ", "AFTER moveJ", "[move_init] dist to target",
    "=== INIT CHECK ===", "[RTDE] Initial TCP:", "=== AUTO PNP TIMING ===", "[PNP] Waypoint verification:",
    "RTDE TCP  :", "FK Pinoc  :", "Error Pinoc vs RTDE:", "[PRE-CHECK] Verifying CBF setup...",
    "WB Containment min_h :", "WB Obstacle    min_h :", "Overall        min_h :", "[CBF] JIT warm-up...",
    "[CBF] Ready.", "[CONTROL] Traj=", "→ [", "| Task+NS", "σ_min=", "w=", "[WB-COMBINE]",
    "[INFO] Saving CSV...", "  [CSV] dq_cmd.csv", "[INFO] All CSV saved.", "Saved: ",
]


class ConnectingRobot(StaticRobot):
    """StaticRobot that reports a connection like RTDERobot.connect does."""

    def connect(self):
        console.debug("[RTDE] Connecting...")
        report_status(7, 1)
        return True


def _run(robot_model, tmp_path, debug, capsys):
    P = load_preset("pnp").with_trajectory("pnp_interpolated")
    P = dataclasses.replace(P, duration=2, PNP_SPEED=5.0)
    q = np.array(P.q_init, dtype=float)
    pos, _ = robot_model.forward_kinematics(q)
    fake = ConnectingRobot(q, list(pos) + [0.0, 0.0, 0.0])
    env = PyBulletEnv(P.dt, gui=False)
    try:
        out = run(P, Selection("pnp_interpolated", "smc_quasi", "cbf", "none", "127.0.0.2"), fake, env,
                  Metrics(registry=CollectorRegistry()), debug=debug, input_fn=lambda p: "y",
                  sleep=lambda s: None, out_root=str(tmp_path))
    finally:
        env.disconnect()
        console.set_debug(False)
    assert out is not None
    return capsys.readouterr().out


def test_default_output_is_minimal(robot, tmp_path, capsys):
    out = _run(robot, tmp_path, False, capsys)
    lines = [l for l in out.splitlines() if l.strip()]
    kinds = []
    for line in lines:
        matched = [k for k, rx in DEFAULT_LINES.items() if rx.match(line)]
        assert matched, f"unexpected default output line: {line!r}"
        kinds.append(matched[0])
    assert kinds[:3] == ["rtde", "init", "precheck"] and kinds[-1] == "done"
    assert kinds.count("status") == 2       # steps 0 and 200 of 250
    assert kinds.count("pnp") >= 2
    for text in DEBUG_ONLY:
        assert text not in out, text


def test_singularity_transition_printed_by_default(robot, tmp_path, capsys):
    """Start in a wrist singularity (q5 = 0): one default line when the task is deactivated."""
    P = load_preset("pnp").with_trajectory("pnp_interpolated")
    q = np.array(P.q_init, dtype=float)
    q[4] = 0.0
    P = dataclasses.replace(P, duration=0.1, q_init=q.tolist())
    pos, _ = robot.forward_kinematics(q)
    fake = ConnectingRobot(q, list(pos) + [0.0, 0.0, 0.0])
    env = PyBulletEnv(P.dt, gui=False)
    try:
        run(P, Selection("pnp_interpolated", "p", "none", "none", "127.0.0.2"), fake, env,
            Metrics(registry=CollectorRegistry()), input_fn=lambda p: "y",
            sleep=lambda s: None, out_root=str(tmp_path))
    finally:
        env.disconnect()
    lines = capsys.readouterr().out.splitlines()
    singular = [l for l in lines if l.startswith("[SINGULAR]")]
    assert len(singular) == 1 and DEFAULT_LINES["singular"].match(singular[0])
    assert "deactivated" in singular[0]
    assert not any("SINGULAR!" in l for l in lines)       # troubleshooting text only with --debug
    for line in lines:
        if line.strip():
            assert any(rx.match(line) for rx in DEFAULT_LINES.values()), line


def test_debug_output_contains_troubleshooting_text(robot, tmp_path, capsys):
    out = _run(robot, tmp_path, True, capsys)
    for text in DEBUG_ONLY:
        assert text in out, text
    for kind in ("rtde", "init", "precheck", "status", "done"):
        assert any(DEFAULT_LINES[kind].match(l) for l in out.splitlines()), kind


def test_filter_state_labels():
    from kossmc_issf.experiment import filter_state
    assert [filter_state("cbf", True), filter_state("issf", False), filter_state("none", False)] == \
           ["correcting", "pass", "none"]
