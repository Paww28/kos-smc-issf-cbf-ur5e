"""CSV output of a run."""

import csv
import os

import numpy as np

from kossmc_issf.console import debug


def save_csv(exp_folder, filename, arr, header=None):
    path = os.path.join(exp_folder, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if header: writer.writerow(header)
        writer.writerows(arr if arr.ndim > 1 else arr.reshape(-1, 1))
    debug(f"  [CSV] {filename}")


def h_headers(scenario, n_constraints):
    """h_vals.csv header of a scenario."""
    if scenario == "ee_containment":
        return ["h_xmax","h_xmin","h_ymax","h_ymin","h_zmax","h_zmin"]
    elif scenario == "wb_obstacle":
        return [f"h_sphere_{i}" for i in range(n_constraints)]
    elif scenario == "wb_containment":
        n_spheres = n_constraints // 6

        h_headers = []

        for i in range(n_spheres):
            h_headers.extend([
                f"sphere{i}_xmin",
                f"sphere{i}_ymin",
                f"sphere{i}_zmin",
            ])

        for i in range(n_spheres):
            h_headers.extend([
                f"sphere{i}_xmax",
                f"sphere{i}_ymax",
                f"sphere{i}_zmax",
            ])
        return h_headers
    return h_headers_scenario5(n_constraints)


def h_headers_scenario5(n_constraints):
    n_spheres = n_constraints // 7
    h_headers = []
    for i in range(n_spheres):
        h_headers.extend([f"s{i}_xmin", f"s{i}_ymin", f"s{i}_zmin"])
    for i in range(n_spheres):
        h_headers.extend([f"s{i}_xmax", f"s{i}_ymax", f"s{i}_zmax"])
    for i in range(n_spheres):
        h_headers.append(f"s{i}_obs")
    return h_headers


def save_all_csv(exp_folder, A, is_pnp, scenario="wb_combine"):
    """A: output of datalog.logger.to_arrays.

    Known issue: disturbance.csv has the header ["dx","dy","dz"] over 6 columns (joint disturbances);
    the data rows are correct.
    """
    def save(filename, arr, header=None):
        save_csv(exp_folder, filename, arr, header)

    twin_q_arr, twin_qd_arr = A["twin_q_arr"], A["twin_qd_arr"]

    debug("[INFO] Saving CSV...")
    save("trajectory_target.csv",   A["trajectory_target"],      ["x_target","y_target","z_target"])
    save("traj_act_fk.csv",         A["traj_act_fk"],            ["x_fk","y_fk","z_fk"])
    save("traj_act_rtde.csv",       A["traj_act_rtde"],          ["x_rtde","y_rtde","z_rtde"])
    save("rpy_log.csv",             A["rpy_log"],                ["roll_deg","pitch_deg","yaw_deg"])
    save("rpy_rtde_log.csv",        A["rpy_rtde_log"],           ["rx_deg","ry_deg","rz_deg"])
    save("dq_ns_norm.csv",          A["dq_ns_norm_log"],         ["dq_ns_norm"])
    save("dq_ns.csv",               A["dq_ns_log"],              [f"dq_ns_{i+1}" for i in range(6)])
    save("dq_cmd.csv",              A["dq_cmd_log"],             [f"dq_cmd_{i+1}" for i in range(6)])
    save("dq_nom.csv",              A["dq_nom_arr"],             [f"dq_nom_{i+1}" for i in range(6)])
    save("dq_safe.csv",             A["dq_safe_arr"],            [f"dq_safe_{i+1}" for i in range(6)])
    save("q_log.csv",               np.rad2deg(A["q_log"]),      [f"q_deg_{i+1}" for i in range(6)])
    save("dq_act_log.csv",          A["dq_act_log"],             [f"dq_act_{i+1}" for i in range(6)])
    save("time_log.csv",            A["t_log"],                  ["time_s"])
    save("twin_q_log.csv",          twin_q_arr,                  [f"q_twin_{i+1}" for i in range(twin_q_arr.shape[1])])
    save("twin_qd_log.csv",         twin_qd_arr,                 [f"dq_twin_{i+1}" for i in range(twin_qd_arr.shape[1])])
    save("twin_pos_log.csv",        A["twin_pos_arr"],           ["x_twin","y_twin","z_twin"])
    save("twin_rpy_log.csv",        A["twin_rpy_arr"],           ["roll_twin","pitch_twin","yaw_twin"])
    save("twin_time_log.csv",       A["twin_time_arr"],          ["twin_timestamp"])
    save("singularity_w.csv",       A["full_manip_log"],         ["manipulability_6x6"])
    save("manipulability_3x6.csv",  A["manip_log"],              ["manipulability_3x6"])
    save("singularity_status.csv",  A["singularity_status_log"], ["is_singular"])
    save("sigma_min.csv",           A["sigma_min_log"],          ["sigma_min"])
    save("rank_Jp.csv",             A["rank_Jp_log"],            ["rank_Jp_3x6"])
    save("rank_J_full.csv",         A["rank_J_log"],             ["rank_J_6x6"])
    save("j_norm_log.csv",          np.array(A["j_norm_log"]),   ["J_norm_2"])
    save("sliding_surface.csv",     A["log_sigma_smc"],          ["sigma_x","sigma_y","sigma_z"])
    save("lyapunov_V.csv",          A["V_lyapunov_log"],         ["V_lyapunov"])
    save("error_dot.csv",           A["log_edot"],               ["edot_x","edot_y","edot_z"])
    save("error_xyz_mm.csv",        A["error_xyz"],              ["ex_mm","ey_mm","ez_mm"])
    save("error_norm_mm.csv",       A["errors_mm"],              ["error_norm_mm"])
    save("disturbance.csv",         A["log_disturbance"],        ["dx","dy","dz"])
    save("cbf_active.csv",          A["cbf_active_arr"],         ["cbf_active"])
    save("h_vals.csv", A["h_vals_arr"], h_headers(scenario, A["h_vals_arr"].shape[1]))

    if is_pnp:
        save("pnp_phase.csv", np.array(A["pnp_phase_log"]), ["phase"])
        save("pnp_cycle.csv", np.array(A["pnp_cycle_log"]), ["cycle"])

    debug("[INFO] All CSV saved.\n")
