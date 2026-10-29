"""Per-cycle logs and their conversion to arrays."""

from dataclasses import dataclass, field, fields
from typing import List

import numpy as np


@dataclass
class RunLog:
    pnp_phase_log: List = field(default_factory=list)
    pnp_cycle_log: List = field(default_factory=list)
    traj_act_fk: List = field(default_factory=list)
    traj_act_rtde: List = field(default_factory=list)
    rpy_log: List = field(default_factory=list)
    rpy_rtde_log: List = field(default_factory=list)
    t_log: List = field(default_factory=list)
    dq_ns_log: List = field(default_factory=list)
    v_des_log: List = field(default_factory=list)
    dq_ns_norm_log: List = field(default_factory=list)
    dq_cmd_log: List = field(default_factory=list)
    q_log: List = field(default_factory=list)
    dq_act_log: List = field(default_factory=list)
    twin_q_log: List = field(default_factory=list)
    twin_qd_log: List = field(default_factory=list)
    twin_pos_log: List = field(default_factory=list)
    twin_rpy_log: List = field(default_factory=list)
    twin_time_log: List = field(default_factory=list)
    full_manip_log: List = field(default_factory=list)
    manip_log: List = field(default_factory=list)
    singularity_status_log: List = field(default_factory=list)
    sigma_min_log: List = field(default_factory=list)
    rank_Jp_log: List = field(default_factory=list)
    rank_J_log: List = field(default_factory=list)
    j_norm_log: List = field(default_factory=list)
    log_sigma_smc: List = field(default_factory=list)
    V_lyapunov_log: List = field(default_factory=list)
    log_edot: List = field(default_factory=list)
    log_disturbance: List = field(default_factory=list)
    cbf_active_log: List = field(default_factory=list)
    h_vals_log: List = field(default_factory=list)
    dq_nom_log: List = field(default_factory=list)
    dq_safe_log: List = field(default_factory=list)
    trajectory_target_pnp: List = field(default_factory=list)

    @classmethod
    def from_lists(cls, lists):
        """Build from a mapping of log name -> sequence (e.g. a recorded run)."""
        return cls(**{f.name: list(lists[f.name]) for f in fields(cls)})


def to_arrays(log, trajectory_target, is_pnp, dt):
    """Log lists -> arrays, plus the derived tracking quantities.

    trajectory_target: the precomputed target array (non-PnP); ignored for PnP, where the logged
    targets are used. j_norm_log, pnp_phase_log and pnp_cycle_log stay lists.
    Known issue: for sinusoidal_3d the full target array is compared with the logged positions, so
    if the loop stopped early (Ctrl-C, error) the shapes differ and the conversion fails.
    """
    A = {}
    if is_pnp :
        A["trajectory_target"] = np.array(log.trajectory_target_pnp)
    else:
        A["trajectory_target"]      = np.array(trajectory_target)
    A["traj_act_fk"]            = np.array(log.traj_act_fk)
    A["traj_act_rtde"]          = np.array(log.traj_act_rtde)
    A["rpy_log"]                = np.array(log.rpy_log)
    A["rpy_rtde_log"]           = np.array(log.rpy_rtde_log)
    A["t_log"]                  = np.array(log.t_log)
    A["dq_ns_log"]              = np.array(log.dq_ns_log)
    A["v_des_log"]              = np.array(log.v_des_log)
    A["dq_ns_norm_log"]         = np.array(log.dq_ns_norm_log)
    A["dq_cmd_log"]             = np.array(log.dq_cmd_log)
    A["q_log"]                  = np.array(log.q_log)
    A["dq_act_log"]             = np.array(log.dq_act_log)
    A["twin_q_arr"]             = np.array(log.twin_q_log)
    A["twin_qd_arr"]            = np.array(log.twin_qd_log)
    A["twin_pos_arr"]           = np.array(log.twin_pos_log)
    A["twin_rpy_arr"]           = np.array(log.twin_rpy_log)
    A["twin_time_arr"]          = np.array(log.twin_time_log)
    A["full_manip_log"]         = np.array(log.full_manip_log)
    A["manip_log"]              = np.array(log.manip_log)
    A["singularity_status_log"] = np.array(log.singularity_status_log)
    A["sigma_min_log"]          = np.array(log.sigma_min_log)
    A["rank_Jp_log"]            = np.array(log.rank_Jp_log)
    A["rank_J_log"]             = np.array(log.rank_J_log)
    A["log_sigma_smc"]          = np.array(log.log_sigma_smc)
    A["V_lyapunov_log"]         = np.array(log.V_lyapunov_log)
    A["log_edot"]               = np.array(log.log_edot)
    A["log_disturbance"]        = np.array(log.log_disturbance)
    A["cbf_active_arr"]         = np.array(log.cbf_active_log)
    A["h_vals_arr"]             = np.array(log.h_vals_log)
    A["dq_nom_arr"]             = np.array(log.dq_nom_log)
    A["dq_safe_arr"]            = np.array(log.dq_safe_log)
    A["sigma_norms"]            = np.linalg.norm(A["log_sigma_smc"], axis=1)

    A["errors_mm"]              = np.linalg.norm(A["traj_act_rtde"] - A["trajectory_target"], axis=1) * 1000
    A["error_xyz"]              = (A["traj_act_rtde"] - A["trajectory_target"]) * 1000
    A["control_effort"]         = np.sum(np.linalg.norm(A["dq_cmd_log"], axis=1)**2) * dt
    A["dq_cmd_norm"]            = np.linalg.norm(A["dq_cmd_log"], axis=1)

    A["j_norm_log"]    = log.j_norm_log
    A["pnp_phase_log"] = log.pnp_phase_log
    A["pnp_cycle_log"] = log.pnp_cycle_log
    return A
