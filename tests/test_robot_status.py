"""Robot / safety mode names in the connection line, and the "robot not ready" warning.

RTDERobot.connect is run with fake rtde_control / rtde_receive modules (no network).
"""

import sys
import types

import pytest

from kossmc_issf.robot.rtde_interface import (ROBOT_MODES, SAFETY_MODES, RTDERobot, mode_name,
                                             report_status)

WARN = "[WARN] robot not ready: check PolyScope (power on, brakes released, no protective stop)"


def test_robot_mode_table():
    assert ROBOT_MODES == {
        -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
        4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE"}


def test_safety_mode_table():
    assert SAFETY_MODES[1] == "NORMAL" and SAFETY_MODES[3] == "PROTECTIVE_STOP"
    assert SAFETY_MODES[5] == "SAFEGUARD_STOP" and SAFETY_MODES[7] == "ROBOT_EMERGENCY_STOP"
    assert sorted(SAFETY_MODES) == list(range(1, 21)) + [22]


@pytest.mark.parametrize("table,code,expected", [
    (ROBOT_MODES, 7, "RUNNING (7)"), (ROBOT_MODES, -1, "NO_CONTROLLER (-1)"), (ROBOT_MODES, 3, "POWER_OFF (3)"),
    (ROBOT_MODES, 42, "UNKNOWN (42)"), (SAFETY_MODES, 1, "NORMAL (1)"), (SAFETY_MODES, 21, "UNKNOWN (21)"),
])
def test_mode_name(table, code, expected):
    assert mode_name(table, code) == expected


@pytest.mark.parametrize("robot_mode,safety_mode,lines", [
    (7, 1, ["[RTDE] Connected | robot RUNNING (7) | safety NORMAL (1)"]),
    (5, 1, ["[RTDE] Connected | robot IDLE (5) | safety NORMAL (1)", WARN]),
    (7, 3, ["[RTDE] Connected | robot RUNNING (7) | safety PROTECTIVE_STOP (3)", WARN]),
    (99, 0, ["[RTDE] Connected | robot UNKNOWN (99) | safety UNKNOWN (0)", WARN]),
])
def test_report_status(robot_mode, safety_mode, lines, capsys):
    report_status(robot_mode, safety_mode)
    assert capsys.readouterr().out.splitlines() == lines


def _fake_rtde(monkeypatch, robot_mode, safety_mode):
    class Control:
        def __init__(self, ip): pass
        def isConnected(self): return True
        def isProgramRunning(self): return True

    class Receive:
        def __init__(self, ip): pass
        def getRobotMode(self): return robot_mode
        def getSafetyMode(self): return safety_mode

    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=Control))
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=Receive))


@pytest.mark.parametrize("robot_mode,safety_mode,warn", [(7, 1, False), (3, 1, True), (7, 5, True)])
def test_connect_prints_status_and_continues(monkeypatch, capsys, robot_mode, safety_mode, warn):
    _fake_rtde(monkeypatch, robot_mode, safety_mode)
    assert RTDERobot("127.0.0.2").connect() is True     # the warning does not stop the run
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[RTDE] Connected | robot ")
    assert (WARN in out) is warn
