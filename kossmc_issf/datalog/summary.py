"""experiment_info.txt."""

import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from kossmc_issf.trajectory.pnp import PNP_NAMES


def f3(x):
    """Parameter value for display: 3 decimals (the values used by the program are not rounded)."""
    return f"{float(x):.3f}"


def l3(values):
    """List of parameter values for display, e.g. [0.000, -1.571, 1.571]."""
    return "[" + ", ".join(f3(v) for v in values) + "]"


@dataclass
class RunInfo:
    """Run-specific values written to experiment_info.txt."""
    timestamp: str
    ip: str
    traj_name: str
    ctrl_name: str
    dist_name: str
    scenario_name: str
    method_name: str
    ctrl: str            # "p" / "smc_sign" / "smc_quasi"
    method: str          # "cbf" / "issf" / "none"
    dist: str            # "gaussian" / "none"
    is_pnp: bool
    total_time: float
    center_x: Any
    center_y: Any
    center_z: Any
    pnp_waypoints: List
    PNP_SEGMENT_TIMES: List
    pnp_cycle: int
    kossmc_issf_version: str = "unknown"
    git_commit: str = "unknown"
    scenario: str = "wb_combine"   # option key (ee_containment / wb_obstacle / wb_containment / wb_combine)


def write_scenario_cbf(f, scenario, P, h_vals_arr):
    """Scenario-specific lines of the CBF section."""
    POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS = P.POS_MIN, P.POS_MAX, P.OBS_CENTER, P.OBS_RADIUS
    if scenario == "ee_containment":
        f.write(f"Box MIN     : {l3(POS_MIN)}\n")
        f.write(f"Box MAX     : {l3(POS_MAX)}\n")
        h_labels_summary = ['x≤xmax','x≥xmin','y≤ymax','y≥ymin','z≤zmax','z≥zmin']
        worst = np.argmin(np.min(h_vals_arr, axis=0))
        f.write(f"Worst barrier: {h_labels_summary[worst]}\n")
        f.write(f"min(h) per barrier: {[f'{v:.4f}' for v in np.min(h_vals_arr, axis=0).tolist()]}\n")
    elif scenario == "wb_obstacle":
        f.write(f"Obs Center  : {l3(OBS_CENTER)}\n")
        f.write(f"Obs Radius  : {f3(OBS_RADIUS)}\n")
    elif scenario == "wb_containment":
        f.write(f"Box MIN     : {l3(POS_MIN)}\n")
        f.write(f"Box MAX     : {l3(POS_MAX)}\n")
    elif scenario == "wb_combine":
        f.write(f"Box MIN     : {l3(POS_MIN)}\n")
        f.write(f"Box MAX     : {l3(POS_MAX)}\n")
        f.write(f"Obs Center  : {l3(OBS_CENTER)}\n")
        f.write(f"Obs Radius  : {f3(OBS_RADIUS)}\n")
        n_s = h_vals_arr.shape[1] // 7
        h_cont_summary = h_vals_arr[:, :n_s*6]
        h_obs_summary  = h_vals_arr[:, n_s*6:]
        f.write(f"min(h) containment : {np.min(h_cont_summary):.4f}\n")
        f.write(f"min(h) obstacle    : {np.min(h_obs_summary):.4f}\n")
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")


def write_experiment_info(exp_folder, A, P, R):
    """A: logger.to_arrays output, P: config.presets.Params, R: RunInfo.

    Returns dt recomputed as t_log[1] - t_log[0]: dt is overwritten here and the
    plots use that value.
    """
    t_log = A["t_log"]
    errors_mm, error_xyz = A["errors_mm"], A["error_xyz"]
    j_norm_log = A["j_norm_log"]
    dq_act_log = A["dq_act_log"]
    h_vals_arr, cbf_active_arr = A["h_vals_arr"], A["cbf_active_arr"]
    log_disturbance, manip_log = A["log_disturbance"], A["manip_log"]
    sigma_min_log, singularity_status_log = A["sigma_min_log"], A["singularity_status_log"]
    rank_Jp_log, rank_J_log = A["rank_Jp_log"], A["rank_J_log"]
    sigma_norms = A["sigma_norms"]
    control_effort = A["control_effort"]
    dt, duration, total_time = P.dt, P.duration, R.total_time
    Lambda, k_smc = P.Lambda, P.k_smc

    sing_pct    = np.sum(singularity_status_log) / len(t_log) * 100
    cbf_act_pct = np.sum(cbf_active_arr) / len(t_log) * 100
    with open(os.path.join(exp_folder, 'experiment_info.txt'), 'w') as f:
        f.write("=" * 55 + "\n")
        f.write("=== EXPERIMENT SUMMARY ===\n")
        f.write("=" * 55 + "\n\n")

        # --- SETUP ---
        f.write("=== SETUP ===\n")
        f.write(f"Timestamp   : {R.timestamp}\n")
        f.write(f"Robot IP    : {R.ip}\n")
        f.write(f"Trajectory  : {R.traj_name}\n")
        f.write(f"Controller  : {R.ctrl_name}\n")
        f.write(f"Disturbance : {R.dist_name}\n")
        f.write(f"Scenario    : {R.scenario_name}\n")
        f.write(f"Safety      : {R.method_name}\n")
        f.write(f"Duration    : {f3(duration)}s\n")
        f.write(f"Run duration : {total_time:.2f} s\n")
        f.write(f"Steps       : {len(t_log)} @ {1/dt:.0f}Hz\n\n")

        # --- RUN INFO ---
        f.write("=== RUN INFO ===\n")
        f.write(f"Preset      : {P.name}\n")
        f.write(f"kossmc_issf : {R.kossmc_issf_version}\n")
        f.write(f"Git commit  : {R.git_commit}\n\n")

        # --- CONTROL PARAMETERS ---
        f.write("=== CONTROL PARAMETERS ===\n")
        f.write(f"Kp          : {f3(P.Kp)}\n")
        f.write(f"K_null      : {f3(P.K_null)}\n")
        f.write(f"Lambda      : diag{l3(np.diag(Lambda))}\n")
        f.write(f"k_smc       : {f3(k_smc)}\n")
        f.write(f"eps_qsmc    : {f3(P.eps_qsmc)}\n")
        f.write(f"Amplitude   : {f3(P.amplitude)}m\n")
        f.write(f"Frequency   : {f3(P.frequency_traj)}Hz\n")
        f.write(f"dt          : {f3(P.dt)}s\n")
        f.write(f"q_init      : {l3(P.q_init)}\n")
        f.write(f"K_jl        : {f3(P.K_jl)}\n")
        f.write(f"q_min       : {l3(P.q_min)}\n")
        f.write(f"q_max       : {l3(P.q_max)}\n")
        f.write(f"q_mid       : {l3(P.q_mid)}\n")
        f.write(f"Singularity σ_min threshold : {f3(P.EPS)}\n")
        f.write(f"ISSf ε      : {f3(P.ISSF_EPSILON)}\n")
        f.write(f"Dist std    : {f3(P.DIST_STD)}\n")
        f.write(f"Dist clip   : {f3(P.DIST_CLIP)}\n")
        f.write(f"PNP_SPEED   : {f3(P.PNP_SPEED)}\n")
        f.write(f"PNP_THRESH  : {f3(P.PNP_THRESH)}mm\n\n")

        # --- TRACKING PERFORMANCE ---
        f.write("=== TRACKING PERFORMANCE ===\n")
        f.write(f"Initial TCP : X={R.center_x:.4f}  Y={R.center_y:.4f}  Z={R.center_z:.4f}\n")
        f.write(f"Mean error  : {np.mean(errors_mm):.2f}mm\n")
        f.write(f"Max error   : {np.max(errors_mm):.2f}mm\n")
        f.write(f"Min error   : {np.min(errors_mm):.2f}mm\n")
        f.write(f"Std error   : {np.std(errors_mm):.2f}mm\n")
        f.write(f"RMS error   : {np.sqrt(np.mean(errors_mm**2)):.2f}mm\n")
        # per axis
        f.write(f"Mean |ex|   : {np.mean(np.abs(error_xyz[:,0])):.2f}mm\n")
        f.write(f"Mean |ey|   : {np.mean(np.abs(error_xyz[:,1])):.2f}mm\n")
        f.write(f"Mean |ez|   : {np.mean(np.abs(error_xyz[:,2])):.2f}mm\n\n")
        f.write(f"Max ||J||_2 : {np.max(j_norm_log):.4f}\n")
        f.write(f"Mean ||J||_2: {np.mean(j_norm_log):.4f}\n")


        f.write("=== CONTROL EFFORT ===\n")
        control_effort_act = np.sum(np.linalg.norm(dq_act_log, axis=1)**2) * dt
        f.write(f"J_cmd = sum(||u_cmd||^2 * dt) : {control_effort:.4f} rad^2/s\n")
        f.write(f"J_act = sum(||u_act||^2 * dt) : {control_effort_act:.4f} rad^2/s\n\n")

        if R.is_pnp:
            pnp_cycle = R.pnp_cycle
            f.write("=== PICK AND PLACE ===\n")
            f.write(f"Waypoints   :\n")
            for name, wp in zip(PNP_NAMES, R.pnp_waypoints):
                f.write(f"  {name:20s}: {l3(wp)}\n")
            f.write(f"Seg Times   : {R.PNP_SEGMENT_TIMES}\n")
            f.write(f"Total cycles: {pnp_cycle}\n")

            # Error per phase
            f.write(f"\nError per phase:\n")
            pnp_phase_arr = np.array(A["pnp_phase_log"])
            for ph_idx, ph_name in enumerate(PNP_NAMES):
                mask = pnp_phase_arr == ph_idx
                if np.any(mask):
                    e_phase = errors_mm[mask]
                    f.write(f"  {ph_name:20s}: mean={np.mean(e_phase):.2f}mm  "
                            f"max={np.max(e_phase):.2f}mm  "
                            f"std={np.std(e_phase):.2f}mm\n")

            f.write("\n")

        # --- CBF SUMMARY ---
        f.write("=== CBF ===\n")
        f.write(f"Alpha gain  : {f3(P.CBF_ALPHA)}\n")

        h_min_per_step = np.min(h_vals_arr, axis=1)
        n_violations   = int(np.sum(h_min_per_step < 0))
        cbf_active_cnt = int(np.sum(cbf_active_arr))
        cbf_act_pct    = cbf_active_cnt / len(t_log) * 100

        write_scenario_cbf(f, R.scenario, P, h_vals_arr)

        f.write(f"Filter correcting : {cbf_active_cnt}/{len(t_log)} ({cbf_act_pct:.2f}%)\n")
        # pure filter intervention (no disturbance)
        corr = np.linalg.norm(A["dq_safe_arr"] - A["dq_nom_arr"], axis=1)
        f.write(f"Filter corr.: ||dq_safe-dq_nom|| mean={np.mean(corr):.4f} max={np.max(corr):.4f} rad/s\n")
        f.write(f"Violations  : {n_violations} steps\n")
        f.write(f"min(h)      : {np.min(h_vals_arr):.4f}\n")

        # first filter correction
        if np.any(cbf_active_arr):
            first_active_idx  = np.argmax(cbf_active_arr)
            first_active_time = t_log[first_active_idx]
        else:
            first_active_time = -1

        f.write(f"First filter correction   : {first_active_time:.3f} s\n")

        # dt is overwritten here (used by the plots afterwards)
        dt = t_log[1] - t_log[0]
        cbf_active_duration = np.sum(cbf_active_arr) * dt
        f.write(f"Filter correcting duration: {cbf_active_duration:.3f} s\n")

        # longest violation
        violations = h_min_per_step < 0
        max_violation_len = 0
        current = 0

        for v in violations:
            if v:
                current += 1
                max_violation_len = max(max_violation_len, current)
            else:
                current = 0

        f.write(f"Longest violation duration: {max_violation_len * dt:.3f} s\n")

        f.write("\n")

        # --- DISTURBANCE ---
        if R.dist != "none":
            d_norm = np.linalg.norm(log_disturbance, axis=1)
            f.write("=== DISTURBANCE ===\n")
            f.write(f"Type        : {R.dist_name}\n")
            f.write(f"Max ||d||   : {np.max(d_norm):.4f} rad/s\n")
            f.write(f"Mean ||d||  : {np.mean(d_norm[d_norm>0]):.4f} rad/s (while active)\n\n")

        # --- MANIPULABILITY ---
        f.write("=== MANIPULABILITY ===\n")
        f.write(f"Mean        : {np.mean(manip_log):.6f}\n")
        f.write(f"Max         : {np.max(manip_log):.6f}\n")
        f.write(f"Min         : {np.min(manip_log):.6f}\n")
        f.write(f"Std         : {np.std(manip_log):.6f}\n\n")

        # --- SINGULARITY ---
        f.write("=== SINGULARITY ===\n")
        f.write(f"Min σ_min         : {np.min(sigma_min_log):.6e}\n")
        f.write(f"Mean σ_min        : {np.mean(sigma_min_log):.6e}\n")
        f.write(f"Singular steps    : {int(np.sum(singularity_status_log))}/{len(t_log)} ({sing_pct:.2f}%)\n")
        f.write(f"rank(Jp) min/max  : {int(np.min(rank_Jp_log))}/{int(np.max(rank_Jp_log))}\n")
        f.write(f"rank(J)  min/max  : {int(np.min(rank_J_log))}/{int(np.max(rank_J_log))}\n\n")

        # --- SLIDING SURFACE (SMC only) ---
        if R.ctrl != "p":
            f.write("=== SLIDING SURFACE ===\n")
            f.write(f"Init ||σ||  : {sigma_norms[0]:.4f}\n")
            f.write(f"Final ||σ|| : {sigma_norms[-1]:.4f}\n")
            f.write(f"Max ||σ||   : {np.max(sigma_norms):.4f}\n")
            f.write(f"Mean ||σ||  : {np.mean(sigma_norms):.4f}\n")
            f.write("\n")

        f.write("=" * 55 + "\n")

    return dt
