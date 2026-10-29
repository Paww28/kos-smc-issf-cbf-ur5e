"""Snapshot fixtures: fixed inputs and the current kossmc_issf outputs, one .npz per group.

    PYTHONPATH= python tests/fixtures/make_fixtures.py

Each group has `inputs_<group>()` (fixed inputs, seeded) and `compute_<group>(inputs)` (kossmc_issf
calls only). The .npz stores both; tests/test_snapshots.py recomputes the outputs from the stored
inputs and compares them. Key prefixes:

    in__     input
    np__     NumPy float output        (atol 1e-12)
    jax__    JAX float64 output        (atol 1e-10)
    exact__  discrete output (int, bool, str), identical

Regenerate only after an intended behavior change, and say so in the commit.
"""

import contextlib
import dataclasses
import io
import os
import tempfile

import numpy as np

import kossmc_issf  # noqa: F401  (imports oscbf first: jax_enable_x64)
from kossmc_issf.config.naming import CTRL_NAMES, DIST_NAMES, METHOD_NAMES, SCENARIO_NAMES
from kossmc_issf.config.presets import load_preset
from kossmc_issf.core.controllers import CONTROLLERS, lyapunov, task_space_control
from kossmc_issf.core.disturbance import disturbance
from kossmc_issf.core.kinematics import (compute_manipulability, get_robot_model, rpy_from_R,
                                    singularity_analysis, update_task_active)
from kossmc_issf.core.nullspace import joint_limit_gradient, nullspace_velocity
from kossmc_issf.core.safety import (SCENARIOS, build_config, build_safety_filter, compute_h_vals_numpy,
                                safety_step)
from kossmc_issf.datalog.csv_io import save_all_csv
from kossmc_issf.datalog.logger import RunLog, to_arrays
from kossmc_issf.datalog.summary import RunInfo, write_experiment_info
from kossmc_issf.experiment import _debug_status_scenario, filter_state
from kossmc_issf.monitoring.prometheus import Metrics
from kossmc_issf.trajectory.pnp import PnPInterpolated, generate_interpolated
from kossmc_issf.trajectory.trajectories import sinusoidal_3d

HERE = os.path.dirname(os.path.abspath(__file__))
PRESET = "pnp"
FLIP_XY = np.array([-1.0, -1.0, 1.0])
EVENTS = {None: 0, "deactivated": 1, "reactivated": 2}
REASONS = {"reached": 0, "seg_complete": 1}


def params():
    """Reference parameters; the preset name is fixed so that it does not enter the outputs."""
    return dataclasses.replace(load_preset(PRESET), name="fixture")


def _random_q(rng, n):
    P = params()
    near = np.array(P.q_init) + rng.normal(0, 0.3, (n // 2, 6))
    far = rng.uniform(-np.pi, np.pi, (n - n // 2, 6))
    return np.vstack([near, far])


# ---------------------------------------------------------------- kinematics

def inputs_kinematics():
    rng = np.random.default_rng(101)
    q = _random_q(rng, 40)
    q_sing = np.array(params().q_init, dtype=float)
    q_sing[4] = 0.0                       # wrist singularity
    q_elbow = np.array(params().q_init, dtype=float)
    q_elbow[2] = 0.0                      # elbow stretched
    return {"in__q": np.vstack([q, q_sing, q_elbow])}


def compute_kinematics(inp):
    rm = get_robot_model()
    out = {k: [] for k in ("np__pos", "np__R", "np__rpy", "np__J", "np__J_full", "np__manip", "np__grad_manip",
                           "np__manip_return", "np__sigma_min", "np__full_manip", "exact__rank_Jp", "exact__rank_J")}
    for q in inp["in__q"]:
        pos, R = rm.forward_kinematics(q)
        J = rm.jacobian(q)
        J_full = rm.full_jacobian(q)
        grad, w = rm.manipulability_gradient(q, J)
        s_min, full_manip, rank_Jp, rank_J = singularity_analysis(J_full)
        for k, v in zip(out, (pos, R, np.array(rpy_from_R(R)), J, J_full, compute_manipulability(J), grad, w,
                              s_min, full_manip, rank_Jp, rank_J)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- singularity hysteresis

def inputs_hysteresis():
    P = params()
    q = np.tile(np.array(P.q_init, dtype=float), (120, 1))
    q[:, 4] = np.concatenate([np.linspace(-0.05, 0.05, 60), np.linspace(0.05, -0.05, 60)])  # through q5 = 0
    rng = np.random.default_rng(102)
    sigma = P.EPS + rng.normal(0, 2 * P.EPS, 200)        # synthetic sequence around EPS
    sigma[::17] = P.EPS                                  # exactly at the threshold
    return {"in__q": q, "in__sigma": sigma, "in__eps": np.array(P.EPS)}


def _hysteresis(sigmas, eps):
    task_active, active, events = True, [], []
    for s in sigmas:
        task_active, event = update_task_active(task_active, s, eps)
        active.append(task_active)
        events.append(EVENTS[event])
    return np.array(active), np.array(events)


def compute_hysteresis(inp):
    rm = get_robot_model()
    eps = float(inp["in__eps"])
    sigma_q = np.array([singularity_analysis(rm.full_jacobian(q))[0] for q in inp["in__q"]])
    a_q, e_q = _hysteresis(sigma_q, eps)
    a_s, e_s = _hysteresis(inp["in__sigma"], eps)
    return {"np__sigma_q": sigma_q, "exact__active_q": a_q, "exact__event_q": e_q,
            "exact__active_s": a_s, "exact__event_s": e_s}


# ---------------------------------------------------------------- controllers

def inputs_controllers():
    rng = np.random.default_rng(103)
    n = 30
    return {"in__q": _random_q(rng, n), "in__qd": rng.normal(0, 0.3, (n, 6)),
            "in__delta_x": rng.normal(0, 0.02, (n, 3)), "in__v_des": rng.normal(0, 0.05, (n, 3))}


def compute_controllers(inp):
    P = params()
    rm = get_robot_model()
    out = {}
    for ctrl in CONTROLLERS:
        dq, sig, edot, V = [], [], [], []
        for q, qd, dx, vd in zip(inp["in__q"], inp["in__qd"], inp["in__delta_x"], inp["in__v_des"]):
            J = rm.jacobian(q)
            d, s, e = task_space_control(ctrl, J, np.linalg.pinv(J), dx, qd, vd, P.Kp, P.Lambda, P.k_smc, P.eps_qsmc)
            dq.append(d); sig.append(s); edot.append(e); V.append(lyapunov(s))
        out.update({f"np__{ctrl}__dq_task": np.array(dq), f"np__{ctrl}__sigma": np.array(sig),
                    f"np__{ctrl}__e_dot": np.array(edot), f"np__{ctrl}__V": np.array(V)})
    return out


# ---------------------------------------------------------------- null space

def inputs_nullspace():
    rng = np.random.default_rng(104)
    n = 30
    return {"in__q": _random_q(rng, n), "in__task_active": np.arange(n) % 3 != 0}


def compute_nullspace(inp):
    P = params()
    rm = get_robot_model()
    gjl, dqns = [], []
    for q, active in zip(inp["in__q"], inp["in__task_active"]):
        J = rm.jacobian(q)
        grad_manip, _ = rm.manipulability_gradient(q, J)
        g = joint_limit_gradient(q, P.q_min, P.q_max, P.q_mid)
        gjl.append(g)
        dqns.append(nullspace_velocity(grad_manip, g, P.K_null, P.K_jl, J, np.linalg.pinv(J), bool(active)))
    return {"np__grad_jl": np.array(gjl), "np__dq_ns": np.array(dqns)}


# ---------------------------------------------------------------- safety filters

def inputs_safety():
    rng = np.random.default_rng(105)
    n = 12
    return {"in__q": _random_q(rng, n), "in__dq": rng.normal(0, 0.3, (n, 6))}


def compute_safety(inp):
    P = params()
    rm = get_robot_model()
    out = {}
    for scenario in SCENARIOS:
        cfg = build_config(scenario, rm.manipulator, P)
        for method in ("cbf", "issf", "none"):
            cbf = build_safety_filter(method, cfg, P.ISSF_EPSILON)
            dq_out, active, h = [], [], []
            for q, dq in zip(inp["in__q"], inp["in__dq"]):
                d, a, hv = safety_step(cfg, cbf, q, dq.copy())
                dq_out.append(d); active.append(a); h.append(hv)
            key = f"{scenario}__{method}"
            out[f"jax__{key}__dq"] = np.array(dq_out)
            out[f"exact__{key}__active"] = np.array(active)
            out[f"jax__{key}__h"] = np.array(h)
        if scenario == "ee_containment":       # logged h: measured TCP converted to base_link
            out["np__ee_containment__h_log"] = np.array(
                [compute_h_vals_numpy(rm.forward_kinematics(q)[0] * FLIP_XY, P.POS_MIN, P.POS_MAX)
                 for q in inp["in__q"]])
    return out


# ---------------------------------------------------------------- trajectories

def inputs_trajectory():
    P = params()
    return {"in__center": np.array([-0.4942, -0.1353, 0.4853]), "in__amplitude": np.array(P.amplitude),
            "in__frequency": np.array(P.frequency_traj), "in__duration": np.array(5.0), "in__dt": np.array(P.dt)}


def compute_trajectory(inp):
    target, vel = sinusoidal_3d(inp["in__center"], float(inp["in__amplitude"]), float(inp["in__frequency"]),
                                float(inp["in__duration"]), float(inp["in__dt"]))
    return {"np__target": target, "np__velocity": vel}


def inputs_pnp():
    P = params()
    wps = np.array(P.pnp_waypoints)
    rng = np.random.default_rng(106)
    return {"in__waypoints": wps, "in__speed_ref": np.array(P.PNP_SPEED), "in__speed": np.array(0.5),
            "in__dt": np.array(P.dt), "in__thresh": np.array(P.PNP_THRESH),
            "in__noise": rng.normal(0, 0.0007, (2400, 3))}


def compute_pnp(inp):
    wps = [w for w in inp["in__waypoints"]]
    dt, thresh = float(inp["in__dt"]), float(inp["in__thresh"])
    out = {}
    # generator at the reference speed: segment sizes and times only (the arrays are large)
    pos, vel, times, lengths = generate_interpolated(wps, float(inp["in__speed_ref"]), dt)
    out["exact__ref_seg_len"] = np.array([len(p) for p in pos])
    out["np__ref_seg_times"] = np.array(times)
    out["np__ref_seg_lengths"] = np.array(lengths)
    out["np__ref_seg_first_last"] = np.array([[p[0], p[-1], v[0]] for p, v in zip(pos, vel)])
    # generator + phase machine at a higher speed (smaller fixture), noisy tracking, 2 cycles
    pos, vel, times, lengths = generate_interpolated(wps, float(inp["in__speed"]), dt)
    out["np__seg_pos"] = np.concatenate(pos)
    out["np__seg_vel"] = np.concatenate(vel)
    out["exact__seg_len"] = np.array([len(p) for p in pos])
    out["np__seg_times"] = np.array(times)
    pnp = PnPInterpolated(wps, pos, vel, thresh)
    target_pos, v_des = wps[0], np.zeros(3)
    rec = {k: [] for k in ("target", "v_des", "phase", "cycle", "tr_step", "tr_from", "tr_to", "tr_reason",
                           "tr_cycle_done", "tr_err")}
    for i, noise in enumerate(inp["in__noise"]):
        # measured TCP slightly ahead of the target, so that both transition reasons occur
        target_pos, v_des, phase, tr = pnp.step(target_pos + 1.6 * v_des * dt + noise)
        rec["target"].append(target_pos); rec["v_des"].append(v_des)
        rec["phase"].append(phase); rec["cycle"].append(pnp.pnp_cycle)
        if tr is not None:
            rec["tr_step"].append(i); rec["tr_from"].append(tr["from_phase"]); rec["tr_to"].append(tr["to_phase"])
            rec["tr_reason"].append(REASONS[tr["reason"]]); rec["tr_cycle_done"].append(tr["cycle_completed"])
            rec["tr_err"].append(tr["err_mm"])
    for k, v in rec.items():
        prefix = "np__" if k in ("target", "v_des", "tr_err") else "exact__"
        out[f"{prefix}{k}"] = np.array(v)
    return out


# ---------------------------------------------------------------- disturbance

def inputs_disturbance():
    P = params()
    return {"in__seed": np.array(1234), "in__n": np.array(200), "in__t": np.arange(0, 3, P.dt),
            "in__std": np.array(P.DIST_STD), "in__clip": np.array(P.DIST_CLIP)}


def compute_disturbance(inp):
    std, clip, n = float(inp["in__std"]), float(inp["in__clip"]), int(inp["in__n"])
    state = np.random.get_state()
    try:
        np.random.seed(int(inp["in__seed"]))     # global stream, as in a run
        g_global = np.array([disturbance("gaussian", std=std, clip=clip) for _ in range(n)])
        after = np.random.random(3)               # stream position after the samples
    finally:
        np.random.set_state(state)
    rng = np.random.RandomState(int(inp["in__seed"]))
    g_rng = np.array([disturbance("gaussian", rng=rng, std=std, clip=clip) for _ in range(n)])
    sin = np.array([disturbance("sinusoidal", t=t) for t in inp["in__t"]])
    none = np.array([disturbance("none", t=t) for t in inp["in__t"][:10]])
    return {"np__gaussian_global": g_global, "np__global_after": after, "np__gaussian_rng": g_rng,
            "np__sinusoidal": sin, "np__none": none}


# ---------------------------------------------------------------- csv_io + summary

RUNS = [  # scenario, trajectory, controller, method, disturbance
    ("ee_containment", "pnp_interpolated", "p", "cbf", "gaussian"),
    ("wb_obstacle", "sinusoidal_3d", "smc_sign", "issf", "sinusoidal"),
    ("wb_containment", "pnp_interpolated", "smc_quasi", "none", "none"),
    ("wb_combine", "sinusoidal_3d", "smc_quasi", "issf", "gaussian"),
    ("wb_combine", "pnp_interpolated", "smc_sign", "cbf", "none"),
]
N_H = {"ee_containment": 6, "wb_obstacle": 13, "wb_containment": 78, "wb_combine": 91}


def _synthetic_lists(rng, n, nh, is_pnp):
    walk = lambda d, s: np.cumsum(rng.normal(0, s, (n, d)), axis=0)
    home = np.array([-0.49, -0.13, 0.49])
    act = home + walk(3, 0.002)
    target = home + walk(3, 0.002)
    lists = dict(
        pnp_phase_log=list(np.repeat(np.arange(8), n // 8 + 1)[:n]) if is_pnp else [0] * n,
        pnp_cycle_log=[0] * (n - 5) + [1] * 5 if is_pnp else [0] * n,
        traj_act_fk=list(act + 0.0005), traj_act_rtde=list(act), rpy_log=list(walk(3, 0.1)),
        rpy_rtde_log=list(walk(3, 0.1)), t_log=list(np.arange(n) * 0.008),
        dq_ns_log=list(rng.normal(0, 0.01, (n, 6))), v_des_log=list(rng.normal(0, 0.05, (n, 3))),
        dq_ns_norm_log=list(rng.random(n) * 0.01), dq_cmd_log=list(rng.normal(0, 0.2, (n, 6))),
        q_log=list(walk(6, 0.01)), dq_act_log=list(rng.normal(0, 0.2, (n, 6))),
        twin_q_log=list(walk(6, 0.01)), twin_qd_log=list(rng.normal(0, 0.2, (n, 6))),
        twin_pos_log=list(act + 0.001), twin_rpy_log=list(walk(3, 0.1)),
        twin_time_log=list(np.arange(n) * 0.008 + 1e9), full_manip_log=list(rng.random(n)),
        manip_log=list(rng.random(n)), singularity_status_log=[0] * (n - 4) + [1] * 4,
        sigma_min_log=list(rng.random(n) * 0.2 + 0.1), rank_Jp_log=[3] * n, rank_J_log=[6] * n,
        j_norm_log=list(rng.random(n) + 0.5), log_sigma_smc=list(rng.normal(0, 0.1, (n, 3))),
        V_lyapunov_log=list(rng.random(n) * 0.01), log_edot=list(rng.normal(0, 0.05, (n, 3))),
        log_disturbance=list(rng.normal(0, 0.03, (n, 6))), cbf_active_log=list(rng.integers(0, 2, n)),
        h_vals_log=list(rng.normal(0.05, 0.1, (n, nh))), dq_nom_log=list(rng.normal(0, 0.2, (n, 6))),
        dq_safe_log=list(rng.normal(0, 0.2, (n, 6))),
        trajectory_target_pnp=list(target) if is_pnp else [],
    )
    return lists, target, home


def inputs_datalog():
    rng = np.random.default_rng(107)
    inp = {}
    for r, (scenario, traj, *_rest) in enumerate(RUNS):
        lists, target, home = _synthetic_lists(rng, 24, N_H[scenario], traj == "pnp_interpolated")
        for k, v in lists.items():
            inp[f"in__{r}__{k}"] = np.array(v)
        inp[f"in__{r}__target"] = target
        inp[f"in__{r}__home"] = home
    return inp


def run_arrays(inp, r):
    """(A, P, R, dt) of run r, as a run builds them before saving; dt as returned by write_experiment_info."""
    scenario, traj, ctrl, method, dist = RUNS[r]
    is_pnp = traj == "pnp_interpolated"
    P = params().with_trajectory(traj)
    lists = {f.name: list(inp[f"in__{r}__{f.name}"]) for f in dataclasses.fields(RunLog)}
    home = inp[f"in__{r}__home"]
    A = to_arrays(RunLog.from_lists(lists), None if is_pnp else inp[f"in__{r}__target"], is_pnp, P.dt)
    wps = [w.copy() for w in P.pnp_waypoints]
    wps[0] = home.copy()
    R = RunInfo(timestamp="2026-01-01_00-00-00", ip="127.0.0.2", traj_name=traj, ctrl_name=CTRL_NAMES[ctrl],
                dist_name=DIST_NAMES[dist], scenario_name=SCENARIO_NAMES[scenario], method_name=METHOD_NAMES[method],
                ctrl=ctrl, method=method, dist=dist, is_pnp=is_pnp, total_time=0.192, center_x=home[0],
                center_y=home[1], center_z=home[2], pnp_waypoints=wps, PNP_SEGMENT_TIMES=[1.0] * 8,
                pnp_cycle=1 if is_pnp else 0, kossmc_issf_version="fixture", git_commit="fixture", scenario=scenario)
    return A, P, R, A["t_log"][1] - A["t_log"][0]


def run_files(inp, r):
    """Files written by save_all_csv + write_experiment_info for run r: ({name: text}, dt)."""
    A, P, R, _ = run_arrays(inp, r)
    is_pnp, scenario = R.is_pnp, R.scenario
    with tempfile.TemporaryDirectory() as d:
        save_all_csv(d, A, is_pnp, scenario)
        dt = write_experiment_info(d, A, P, R)
        files = {}
        for name in sorted(os.listdir(d)):
            with open(os.path.join(d, name)) as f:
                files[name] = f.read()
    return files, dt


def compute_datalog(inp):
    out = {}
    for r in range(len(RUNS)):
        files, dt = run_files(inp, r)
        out[f"exact__{r}__names"] = np.array(list(files))
        out[f"exact__{r}__texts"] = np.array(list(files.values()))
        out[f"np__{r}__dt"] = np.array(dt)
    return out


# ---------------------------------------------------------------- Prometheus gauges + --debug text

def inputs_monitoring():
    rng = np.random.default_rng(108)
    n = 10
    inp = {"in__delta_x": rng.normal(0, 0.05, (n, 3)), "in__target_pos": rng.normal(0, 0.3, (n, 3)),
           "in__x_actual": rng.normal(0, 0.3, (n, 3)), "in__dq_ns": rng.normal(0, 0.1, (n, 6)),
           "in__cbf_active": np.arange(n) % 2 == 0, "in__dq_cmd": rng.normal(0, 0.3, (n, 6)),
           "in__dq_nom": rng.normal(0, 0.3, (n, 6)), "in__dq_safe": rng.normal(0, 0.3, (n, 6)),
           "in__sigma": rng.normal(0, 0.1, (n, 3)), "in__qd": rng.normal(0, 0.3, (n, 6)),
           "in__q": rng.uniform(-2 * np.pi, 2 * np.pi, (n, 6)), "in__q_twin": rng.normal(0, 2, (n, 6)),
           "in__qd_twin": rng.normal(0, 0.3, (n, 6)), "in__q_diff": rng.normal(0, 0.1, (n, 6)),
           "in__sigma_min": rng.random(n), "in__manip": rng.random(n), "in__pnp_phase": np.arange(n) % 8}
    inp["in__q"][3, 5] = 2 * np.pi - 0.05      # joint 6 near its limit: joint-limit warning in --debug
    for scenario in SCENARIOS:
        inp[f"in__h__{scenario}"] = rng.normal(0.05, 0.1, (n, N_H[scenario]))
    return inp


def compute_monitoring(inp):
    from prometheus_client import CollectorRegistry
    P = params()
    out = {}
    for scenario in SCENARIOS:
        reg = CollectorRegistry()
        m = Metrics(registry=reg)
        m.set_static(P.OBS_CENTER, P.OBS_RADIUS, P.POS_MIN, P.POS_MAX)
        values, texts = [], []
        for k in range(len(inp["in__q"])):
            h = inp[f"in__h__{scenario}"][k]
            m.update(inp["in__delta_x"][k], inp["in__target_pos"][k], inp["in__x_actual"][k], inp["in__dq_ns"][k],
                     h, bool(inp["in__cbf_active"][k]), inp["in__dq_cmd"][k], inp["in__dq_nom"][k],
                     inp["in__dq_safe"][k], P.OBS_RADIUS, inp["in__sigma"][k], inp["in__qd"][k], inp["in__q"][k],
                     inp["in__q_twin"][k], inp["in__qd_twin"][k], inp["in__q_diff"][k], inp["in__sigma_min"][k],
                     inp["in__manip"][k], int(inp["in__pnp_phase"][k]), scenario=scenario)
            samples = sorted((s.name, s.value) for fam in reg.collect() for s in fam.samples)
            values.append([v for _, v in samples])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _debug_status_scenario(scenario, h, filter_state("cbf", bool(inp["in__cbf_active"][k])), inp["in__q"][k])
            texts.append(buf.getvalue())
        out[f"exact__{scenario}__gauge_names"] = np.array([name for name, _ in samples])
        out[f"np__{scenario}__gauge_values"] = np.array(values)
        out[f"exact__{scenario}__debug_text"] = np.array(texts)
    return out


# ----------------------------------------------------------------

GROUPS = ("kinematics", "hysteresis", "controllers", "nullspace", "safety", "trajectory", "pnp",
          "disturbance", "datalog", "monitoring")


def build(group):
    inp = globals()[f"inputs_{group}"]()
    return inp, globals()[f"compute_{group}"](inp)


def main():
    total = 0
    for group in GROUPS:
        inp, out = build(group)
        path = os.path.join(HERE, f"{group}.npz")
        np.savez_compressed(path, **inp, **out)
        size = os.path.getsize(path)
        total += size
        print(f"{group:12s} {size / 1024:8.1f} KiB")
    print(f"{'total':12s} {total / 1024:8.1f} KiB")


if __name__ == "__main__":
    main()
