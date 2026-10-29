"""Entry point `kossmc-issf-run`.

  kossmc-issf-run [--preset pnp|circular] [--ctrl p|smc_sign|smc_quasi]
                  [--method none|cbf|issf] [--dist none|gaussian|sinusoidal]
                  [--scenario ee_containment|wb_obstacle|wb_containment|wb_combine] [--ip IP] [--debug]

With --preset, --ctrl, --method and --dist the run starts directly (--scenario defaults to
wb_combine). Otherwise a menu asks for the options that are not given: trajectory (= preset),
controller, scenario, method, disturbance.
"""

import argparse

from kossmc_issf.config.defaults import PROMETHEUS_PORT, URSIM_IP
from kossmc_issf.config.naming import MENU_CTRL, MENU_DIST, MENU_METHOD, MENU_SCENARIO, MENU_TRAJ
from kossmc_issf.config.presets import CLI_PRESETS, load_preset
from kossmc_issf.core.controllers import CONTROLLERS
from kossmc_issf.core.disturbance import DISTURBANCES
from kossmc_issf.core.safety import SCENARIOS

METHODS = ("none", "cbf", "issf")
TRAJ_PRESET = {"sinusoidal_3d": "circular", "pnp_interpolated": "pnp"}
PRESET_TRAJ = {v: k for k, v in TRAJ_PRESET.items()}
OPTIONS = ("traj", "ctrl", "scenario", "method", "dist")   # menu order

MENUS = {
    "traj": ("=== SELECT TRAJECTORY TYPE ===",
             ["[1] Sinusoidal 3D (XYZ)", "[2] Pick and Place - Interpolated Trajectory"],
             "Enter choice (1/2): ", MENU_TRAJ),
    "ctrl": ("=== SELECT CONTROL ALGORITHM ===",
             ["[1] Proportional (P)", "[2] SMC - Sign", "[3] SMC - Quasi (saturation)"],
             "Enter choice (1/2/3): ", MENU_CTRL),
    "scenario": ("=== SELECT SAFETY SCENARIO ===",
                 ["[1] EE Containment ", "[2] Whole-body Obstacle Avoidance ",
                  "[3] Whole-body Containment ", "[4] Whole-body Containment + Obstacle Avoidance "],
                 "Enter choice (1/2/3/4): ", MENU_SCENARIO),
    "method": ("=== SELECT SAFETY METHOD ===",
               ["[1] CBF ", "[2] ISSf-CBF ", "[3] No CBF"],
               "Enter choice (1/2/3): ", MENU_METHOD),
    "dist": ("=== SELECT DISTURBANCE ===",
             ["[1] Gaussian noise  ", "[2] Sinusoidal      ", "[3] No disturbance"],
             "Enter choice (1/2/3): ", MENU_DIST),
}


def build_parser():
    ap = argparse.ArgumentParser(
        prog="kossmc-issf-run",
        description="KOS-SMC + CBF / ISSf-CBF experiment on the UR5e. Options that are not given "
                    "are asked in a menu.")
    ap.add_argument("--preset", choices=CLI_PRESETS, help="trajectory and its parameters")
    ap.add_argument("--ctrl", choices=CONTROLLERS, help="task-space controller")
    ap.add_argument("--method", choices=METHODS, help="safety filter")
    ap.add_argument("--dist", choices=DISTURBANCES, help="joint-velocity disturbance")
    ap.add_argument("--scenario", choices=SCENARIOS,
                    help="safety scenario (default wb_combine when --preset/--ctrl/--method/--dist are all given)")
    ap.add_argument("--ip", default=URSIM_IP, help=f"robot IP (default {URSIM_IP}, URSim)")
    ap.add_argument("--debug", action="store_true", help="print troubleshooting output")
    return ap


def interactive_menu(input_fn=input, ask_for=OPTIONS):
    """Menu for the options in `ask_for` (in menu order). Returns {option: key}.

    An answer that is not a menu number is asked again. Raises SystemExit if there is no input.
    """
    chosen = {}
    first = True
    for option in OPTIONS:
        if option not in ask_for:
            continue
        title, lines, prompt, valid = MENUS[option]
        print(("" if first else "\n") + title)
        first = False
        for line in lines:
            print(line)
        while True:
            try:
                answer = input_fn(prompt).strip()
            except EOFError:
                raise SystemExit("no input for the menu: give --preset, --ctrl, --method and --dist")
            if answer in valid:
                chosen[option] = valid[answer]
                break
            print(f"  invalid choice {answer!r}")
    return chosen


def resolve(args, input_fn=input):
    """Parsed arguments -> (Params, Selection); the menu asks for the missing options."""
    from kossmc_issf.experiment import Selection

    given = {"traj": PRESET_TRAJ.get(args.preset), "ctrl": args.ctrl, "scenario": args.scenario,
             "method": args.method, "dist": args.dist}
    if all(given[k] is not None for k in ("traj", "ctrl", "method", "dist")):
        given["scenario"] = given["scenario"] or "wb_combine"
    else:
        given.update(interactive_menu(input_fn, [k for k in OPTIONS if given[k] is None]))
    P = load_preset(TRAJ_PRESET[given["traj"]])
    return P, Selection(traj=given["traj"], ctrl=given["ctrl"], method=given["method"], dist=given["dist"],
                        ip=args.ip, scenario=given["scenario"])


def main(argv=None):
    args = build_parser().parse_args(argv)

    from kossmc_issf.console import set_debug
    set_debug(args.debug)

    # Load the robot model first (as the original research script did at import time): loading the URDF
    # meshes consumes np.random, before any disturbance sample.
    from kossmc_issf.core.kinematics import get_robot_model
    get_robot_model()

    P, sel = resolve(args)

    from kossmc_issf.experiment import run
    from kossmc_issf.monitoring.prometheus import Metrics
    from kossmc_issf.plotting.plots import make_plots
    from kossmc_issf.robot.rtde_interface import RTDERobot
    from kossmc_issf.sim.pybullet_env import PyBulletEnv

    try:
        metrics = Metrics()
        metrics.start_server(PROMETHEUS_PORT)
        env = PyBulletEnv(P.dt, gui=True)
        run(P, sel, RTDERobot(sel.ip), env, metrics, debug=args.debug, plot_fn=make_plots)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopped.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
