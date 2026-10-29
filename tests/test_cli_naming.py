"""CLI resolution (arguments and menu), names and folder format, lazy RTDE import."""

import os
import subprocess
import sys
from datetime import datetime

import pytest

from conftest import REPO
from kossmc_issf.config import naming
from kossmc_issf.config.cli import build_parser, interactive_menu, resolve
from kossmc_issf.datalog.folders import create_experiment_folder


def _resolve(argv, answers=()):
    """resolve() with scripted menu answers; returns (P, sel, prompts asked)."""
    it, prompts = iter(answers), []

    def input_fn(prompt):
        prompts.append(prompt)
        return next(it)
    P, sel = resolve(build_parser().parse_args(argv), input_fn=input_fn)
    return P, sel, prompts


def _no_input(prompt):
    raise AssertionError(f"menu asked {prompt!r}")


def test_full_arguments_run_without_menu():
    args = build_parser().parse_args(["--preset", "pnp", "--ctrl", "smc_quasi", "--method", "issf",
                                      "--dist", "gaussian"])
    P, sel = resolve(args, input_fn=_no_input)
    assert (P.trajectory, P.duration, P.ISSF_EPSILON) == ("pnp_interpolated", 90, 0.5)
    assert (sel.traj, sel.ctrl, sel.method, sel.dist, sel.ip, sel.scenario) == \
           ("pnp_interpolated", "smc_quasi", "issf", "gaussian", "127.0.0.2", "wb_combine")
    args = build_parser().parse_args(["--preset", "pnp", "--ctrl", "p", "--method", "cbf", "--dist", "sinusoidal",
                                      "--scenario", "ee_containment"])
    _, sel = resolve(args, input_fn=_no_input)
    assert (sel.scenario, sel.dist) == ("ee_containment", "sinusoidal")
    args = build_parser().parse_args(["--preset", "circular", "--ctrl", "p", "--method", "none", "--dist", "none",
                                      "--ip", "192.168.0.191", "--debug"])
    _, sel = resolve(args, input_fn=_no_input)
    assert (sel.traj, sel.ip) == ("sinusoidal_3d", "192.168.0.191")


def test_no_arguments_menu_asks_everything():
    P, sel, prompts = _resolve([], answers=["1", "3", "4", "2", "1"])
    assert len(prompts) == 5
    assert (P.trajectory, P.duration, P.ISSF_EPSILON) == ("sinusoidal_3d", 30, 0.5)
    assert (sel.traj, sel.ctrl, sel.scenario, sel.method, sel.dist) == \
           ("sinusoidal_3d", "smc_quasi", "wb_combine", "issf", "gaussian")


def test_ip_and_debug_with_menu():
    args = build_parser().parse_args(["--ip", "10.0.0.7", "--debug"])
    assert args.debug
    it = iter(["2", "1", "1", "3", "3"])
    _, sel = resolve(args, input_fn=lambda prompt: next(it))
    assert (sel.ip, sel.traj, sel.ctrl, sel.scenario, sel.method, sel.dist) == \
           ("10.0.0.7", "pnp_interpolated", "p", "ee_containment", "none", "none")


@pytest.mark.parametrize("argv,answers,expected_prompts,expected", [
    # only the controller missing (scenario given)
    (["--preset", "pnp", "--method", "cbf", "--dist", "none", "--scenario", "wb_obstacle"], ["2"],
     ["Enter choice (1/2/3): "], ("pnp_interpolated", "smc_sign", "wb_obstacle", "cbf", "none")),
    # preset and ctrl given: scenario, method, disturbance asked
    (["--preset", "circular", "--ctrl", "p"], ["3", "1", "2"],
     ["Enter choice (1/2/3/4): ", "Enter choice (1/2/3): ", "Enter choice (1/2/3): "],
     ("sinusoidal_3d", "p", "wb_containment", "cbf", "sinusoidal")),
    # only the scenario given: the four others asked
    (["--scenario", "ee_containment"], ["2", "3", "2", "3"],
     ["Enter choice (1/2): ", "Enter choice (1/2/3): ", "Enter choice (1/2/3): ", "Enter choice (1/2/3): "],
     ("pnp_interpolated", "smc_quasi", "ee_containment", "issf", "none")),
])
def test_partial_arguments_menu_asks_only_missing(argv, answers, expected_prompts, expected):
    P, sel, prompts = _resolve(argv, answers)
    assert prompts == expected_prompts
    assert (sel.traj, sel.ctrl, sel.scenario, sel.method, sel.dist) == expected
    assert P.trajectory == sel.traj


def test_partial_arguments_without_input_exit():
    def eof(prompt):
        raise EOFError
    with pytest.raises(SystemExit):
        resolve(build_parser().parse_args(["--preset", "pnp", "--ctrl", "p"]), input_fn=eof)


@pytest.mark.parametrize("argv", [["--preset", "x"], ["--ctrl", "smc_atan"], ["--dist", "step"],
                                  ["--method", "issf2"], ["--scenario", "ee_obstacle"]])
def test_invalid_arguments_rejected_by_parser(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_menu_reasks_invalid_answers(capsys):
    # traj: 0, 5 -> 2 | ctrl: 4, 9 -> 3 | scenario: x, 5 -> 4 | method: 2 | dist: 4 -> 1
    answers = iter(["0", "5", "2", "4", "9", "3", "x", "5", "4", "2", "4", "1"])
    assert interactive_menu(lambda prompt: next(answers)) == \
           {"traj": "pnp_interpolated", "ctrl": "smc_quasi", "scenario": "wb_combine", "method": "issf",
            "dist": "gaussian"}
    assert capsys.readouterr().out.count("invalid choice") == 7


def test_menu_numbers():
    assert naming.MENU_TRAJ == {"1": "sinusoidal_3d", "2": "pnp_interpolated"}
    assert naming.MENU_CTRL == {"1": "p", "2": "smc_sign", "3": "smc_quasi"}
    assert naming.MENU_SCENARIO == {"1": "ee_containment", "2": "wb_obstacle", "3": "wb_containment",
                                    "4": "wb_combine"}
    assert naming.MENU_METHOD == {"1": "cbf", "2": "issf", "3": "none"}
    assert naming.MENU_DIST == {"1": "gaussian", "2": "sinusoidal", "3": "none"}


def test_names():
    assert naming.TRAJ_NAMES == {"sinusoidal_3d": "sinusoidal_3d", "pnp_interpolated": "pnp_interpolated"}
    assert naming.CTRL_NAMES == {"p": "P", "smc_sign": "SMC_sign", "smc_quasi": "SMC_quasi"}
    assert naming.DIST_NAMES == {"gaussian": "dist_gaussian", "sinusoidal": "dist_sinusoidal", "none": "no_dist"}
    assert naming.METHOD_NAMES == {"cbf": "CBF", "issf": "ISSf-CBF", "none": "No-CBF"}
    assert naming.SCENARIO_NAMES == {"ee_containment": "ee_containment", "wb_obstacle": "wb_obs_avo",
                                     "wb_containment": "whole_body_containment",
                                     "wb_combine": "whole_body_combine"}
    assert [naming.get_robot_source(ip) for ip in ["127.0.0.2", "192.168.0.191", "10.0.0.7"]] == \
           ["URSIM", "ROBOT", "Custom_10_0_0_7"]


def test_folder_format(tmp_path):
    args = ("pnp_interpolated", "SMC_quasi", "dist_gaussian", "URSIM", "whole_body_combine", "ISSf-CBF")
    now = datetime(2026, 1, 2, 3, 4, 5)
    path, ts = create_experiment_folder(*args, root=str(tmp_path / "experiments"), now=now)
    assert ts == "2026-01-02_03-04-05"
    assert os.path.relpath(path, tmp_path) == os.path.join(
        "experiments", "2026-01-02",
        "pnp_interpolated_manip_SMC_quasi_dist_gaussian_whole_body_combine_ISSf-CBF_URSIM_2026-01-02_03-04-05")
    assert os.path.isdir(path)


def test_rtde_imported_lazily():
    code = ("import sys, kossmc_issf.config.cli, kossmc_issf.experiment, kossmc_issf.robot.rtde_interface;"
            "assert 'rtde_control' not in sys.modules and 'rtde_receive' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, PYTHONPATH=""), cwd=REPO)
