"""Result plots of a run (all scenarios, sinusoidal_3d / pnp_interpolated)."""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kossmc_issf.console import debug
from kossmc_issf.trajectory.pnp import PNP_NAMES

COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']


def set_axes_equal_3d(ax):
    lims = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    max_range = max(l[1]-l[0] for l in lims)
    for setlim, lim in zip([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], lims):
        mid = np.mean(lim)
        setlim([mid - max_range/2, mid + max_range/2])


def _rtde_region(POS_MIN, POS_MAX, OBS_CENTER):
    """Safety box / obstacle center flipped back to the RTDE frame of traj_act_rtde."""
    pos_min_rtde    = np.array([-POS_MAX[0], -POS_MAX[1], POS_MIN[2]])
    pos_max_rtde    = np.array([-POS_MIN[0], -POS_MIN[1], POS_MAX[2]])
    obs_center_rtde = np.array([-OBS_CENTER[0], -OBS_CENTER[1], OBS_CENTER[2]])
    return pos_min_rtde, pos_max_rtde, obs_center_rtde


def draw_overlay_2d(ax, scenario, POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS):
    pos_min_rtde, pos_max_rtde, obs_center_rtde = _rtde_region(POS_MIN, POS_MAX, OBS_CENTER)

    if scenario in ("ee_containment", "wb_containment"):
        rect = mpatches.Rectangle(
            (pos_min_rtde[0], pos_min_rtde[1]),
            pos_max_rtde[0] - pos_min_rtde[0],
            pos_max_rtde[1] - pos_min_rtde[1],
            edgecolor='green', facecolor='green', alpha=0.15, label='Safe Box'
        )
        ax.add_patch(rect)

    elif scenario == "wb_obstacle":
        circle = mpatches.Circle(
            (obs_center_rtde[0], obs_center_rtde[1]),
            OBS_RADIUS,
            edgecolor='red', facecolor='red', alpha=0.2, label='Obstacle'
        )
        ax.add_patch(circle)
        ax.scatter(*obs_center_rtde[:2], color='red', marker='x', s=60)

    elif scenario == "wb_combine":
        rect = mpatches.Rectangle(
            (pos_min_rtde[0], pos_min_rtde[1]),
            pos_max_rtde[0] - pos_min_rtde[0],
            pos_max_rtde[1] - pos_min_rtde[1],
            edgecolor='green', facecolor='green', alpha=0.1, label='Safe Box'
        )
        ax.add_patch(rect)
        circle = mpatches.Circle(
            (obs_center_rtde[0], obs_center_rtde[1]),
            OBS_RADIUS,
            edgecolor='red', facecolor='red', alpha=0.3, label='Obstacle'
        )
        ax.add_patch(circle)
        ax.scatter(*obs_center_rtde[:2], color='red', marker='x', s=60)


def _box_verts(xn, yn, zn, xx, yx, zx):
    return [
        [(xn,yn,zn),(xx,yn,zn),(xx,yx,zn),(xn,yx,zn)],
        [(xn,yn,zx),(xx,yn,zx),(xx,yx,zx),(xn,yx,zx)],
        [(xn,yn,zn),(xx,yn,zn),(xx,yn,zx),(xn,yn,zx)],
        [(xn,yx,zn),(xx,yx,zn),(xx,yx,zx),(xn,yx,zx)],
        [(xn,yn,zn),(xn,yx,zn),(xn,yx,zx),(xn,yn,zx)],
        [(xx,yn,zn),(xx,yx,zn),(xx,yx,zx),(xx,yn,zx)]
    ]


def draw_overlay_3d(ax, scenario, POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS):
    pos_min_rtde, pos_max_rtde, obs_center_rtde = _rtde_region(POS_MIN, POS_MAX, OBS_CENTER)
    xn, yn, zn = pos_min_rtde
    xx, yx, zx = pos_max_rtde

    if scenario in ("ee_containment", "wb_containment"):
        ax.add_collection3d(
            Poly3DCollection(
                _box_verts(xn, yn, zn, xx, yx, zx),
                alpha=0.05,
                facecolor='green',
                edgecolor='none'
            )
        )
        ax.plot([], [], [], color='green', label='Safe Box (h=0)')

        padding = 0.1
        ax.set_xlim(xn - padding, xx + padding)
        ax.set_ylim(yn - padding, yx + padding)
        ax.set_zlim(zn - padding, zx + padding)

    elif scenario == "wb_obstacle":
        # drawn at obs_center_rtde (RTDE frame of the plotted trajectory), as in wb_combine
        theta, z_cyl = np.meshgrid(
            np.linspace(0, 2*np.pi, 30),
            np.linspace(obs_center_rtde[2] - 0.5, obs_center_rtde[2] + 0.5, 20)
        )
        ax.plot_surface(
            obs_center_rtde[0] + OBS_RADIUS * np.cos(theta),
            obs_center_rtde[1] + OBS_RADIUS * np.sin(theta),
            z_cyl, color='red', alpha=0.2
        )
        ax.scatter(*obs_center_rtde, color='red', s=50)
        ax.plot([], [], [], color='red', label='Obstacle (Cylinder)')

    elif scenario == "wb_combine":
        # containment box (green)
        ax.add_collection3d(Poly3DCollection(
            _box_verts(xn, yn, zn, xx, yx, zx), alpha=0.05, facecolor='green', edgecolor='none'
        ))
        ax.plot([], [], [], color='green', label='Safe Box')

        # obstacle cylinder (red)
        theta, z_cyl = np.meshgrid(
            np.linspace(0, 2*np.pi, 30),
            np.linspace(obs_center_rtde[2] - 0.5, obs_center_rtde[2] + 0.5, 20)
        )
        ax.plot_surface(
            obs_center_rtde[0] + OBS_RADIUS * np.cos(theta),
            obs_center_rtde[1] + OBS_RADIUS * np.sin(theta),
            z_cyl, color='red', alpha=0.3
        )
        ax.scatter(*obs_center_rtde, color='red', s=50)

        padding = 0.1
        ax.set_xlim(xn - padding, xx + padding)
        ax.set_ylim(yn - padding, yx + padding)
        ax.set_zlim(zn - padding, zx + padding)


def make_plots(exp_folder, A, P, R, dt):
    """Write all PNGs.

    A: datalog.logger.to_arrays output, P: Params, R: datalog.summary.RunInfo,
    dt: the value returned by write_experiment_info (t_log[1] - t_log[0]).
    """
    debug("[PLOTTING] Generating plots...")
    colors = COLORS
    is_pnp = R.is_pnp
    ctrl_name, scenario_name, method_name = R.ctrl_name, R.scenario_name, R.method_name
    POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS = P.POS_MIN, P.POS_MAX, P.OBS_CENTER, P.OBS_RADIUS
    EPS, Lambda, CBF_ALPHA = P.EPS, P.Lambda, P.CBF_ALPHA
    pnp_waypoints = R.pnp_waypoints
    n_wp = len(pnp_waypoints)

    trajectory_target = A["trajectory_target"]
    traj_act_fk, traj_act_rtde = A["traj_act_fk"], A["traj_act_rtde"]
    t_log, error_xyz = A["t_log"], A["error_xyz"]
    cbf_active_arr, h_vals_arr = A["cbf_active_arr"], A["h_vals_arr"]
    log_edot, log_sigma_smc, sigma_norms = A["log_edot"], A["log_sigma_smc"], A["sigma_norms"]
    dq_ns_log = A["dq_ns_log"]
    dq_nom_arr, dq_safe_arr = A["dq_nom_arr"], A["dq_safe_arr"]
    dq_cmd_log, dq_act_log, dq_cmd_norm = A["dq_cmd_log"], A["dq_act_log"], A["dq_cmd_norm"]
    q_log, sigma_min_log = A["q_log"], A["sigma_min_log"]
    singularity_status_log = A["singularity_status_log"]
    rank_Jp_log, rank_J_log = A["rank_Jp_log"], A["rank_J_log"]
    log_disturbance = A["log_disturbance"]
    twin_pos_arr, twin_q_arr = A["twin_pos_arr"], A["twin_q_arr"]
    pnp_phase_log, pnp_cycle_log = A["pnp_phase_log"], A["pnp_cycle_log"]
    h_min_per_step = np.min(h_vals_arr, axis=1)

    scenario = R.scenario
    overlay_2d = lambda ax: draw_overlay_2d(ax, scenario, POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS)
    overlay_3d = lambda ax: draw_overlay_3d(ax, scenario, POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS)

    # ==========================================
    # PLOT 1: TRAJECTORY
    # ==========================================
    if not is_pnp:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        overlay_3d(ax)

        ax.plot(trajectory_target[:,0], trajectory_target[:,1], trajectory_target[:,2],
                'b--', label='Target')

        ax.plot(traj_act_fk[:,0], traj_act_fk[:,1], traj_act_fk[:,2],
                'r-', label='Actual')

        ax.scatter(*traj_act_fk[0], color='green', s=80, label='Start')
        ax.scatter(*traj_act_fk[-1], color='black', s=80, label='End')

        ax.set_title("Trajectory 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.legend()

        set_axes_equal_3d(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, "01_traj_3d.png"))
        plt.close()

    else:
        # PLOT 1A: 3D TRAJECTORY
        fig = plt.figure(figsize=(12, 9))
        ax  = fig.add_subplot(111, projection='3d')

        overlay_3d(ax)

        ax.plot(
            traj_act_rtde[:, 0],
            traj_act_rtde[:, 1],
            traj_act_rtde[:, 2],
            color='red', linewidth=1.5, alpha=0.85, label='Actual'
        )

        wp_arr = np.array(pnp_waypoints)
        ax.scatter(
            wp_arr[:,0], wp_arr[:,1], wp_arr[:,2],
            c='black', s=80, zorder=5
        )

        # short arrow towards the next waypoint
        n_wp_plot = len(pnp_waypoints)
        for idx_wp in range(n_wp_plot):
            wp_cur  = pnp_waypoints[idx_wp]
            wp_next = pnp_waypoints[(idx_wp + 1) % n_wp_plot]
            direction = np.array(wp_next) - np.array(wp_cur)
            dist      = np.linalg.norm(direction)
            if dist > 0.01:
                unit       = direction / dist
                arrow_len  = min(0.06, dist * 0.25)
                arrow_start = np.array(wp_cur)
                ax.quiver(
                    arrow_start[0], arrow_start[1], arrow_start[2],
                    unit[0] * arrow_len,
                    unit[1] * arrow_len,
                    unit[2] * arrow_len,
                    color='blue', linewidth=1.5,
                    arrow_length_ratio=0.4
                )

        # label offsets: waypoints sharing an XY position are spread apart
        xy_clusters = defaultdict(list)
        for idx_wp, wp in enumerate(pnp_waypoints):
            key = (round(wp[0], 1), round(wp[1], 1))
            xy_clusters[key].append(idx_wp)

        label_offsets_3d = {}
        label_offsets_xy = {}

        for key, members in xy_clusters.items():
            if len(members) == 1:
                label_offsets_3d[members[0]] = (0.03,  0.03,  0.03)
                label_offsets_xy[members[0]] = (0.03,  0.03)
            else:
                members_sorted = sorted(members, key=lambda i: pnp_waypoints[i][2])
                for rank, idx_wp in enumerate(members_sorted):
                    label_offsets_3d[idx_wp] = (0.03, 0.03,  0.03 + rank * 0.05)
                    label_offsets_xy[idx_wp] = (0.03, 0.03 + rank * 0.03)
        for idx_wp, (name, wp) in enumerate(zip(PNP_NAMES, pnp_waypoints)):
            ox, oy, oz = label_offsets_3d[idx_wp]
            ax.text(
                wp[0] + ox, wp[1] + oy, wp[2] + oz,
                f'{idx_wp}:{name}',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )

        ax.scatter(*traj_act_rtde[0],  color='green',  s=120, zorder=6, label='Start')
        ax.scatter(*traj_act_rtde[-1], color='purple', s=120, zorder=6, label='End')

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(
            f'PnP Trajectory 3D | {ctrl_name} | {scenario_name} | {method_name}',
            fontsize=12, fontweight='bold'
        )

        legend_elements = [
            Line2D([0],[0], color='red', linewidth=2, label='Actual Trajectory'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
                markersize=8, label='Waypoints'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='green',
                markersize=8, label='Start'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='purple',
                markersize=8, label='End'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

        set_axes_equal_3d(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '01a_pnp_traj_3d.png'), dpi=150)
        plt.close()

        # PLOT 1B: TOP VIEW (XY)
        fig, ax_xy = plt.subplots(1, 1, figsize=(9, 8))
        fig.suptitle(
            f'PnP Trajectory 2D | {ctrl_name} | {scenario_name} | {method_name}',
            fontsize=13, fontweight='bold'
        )

        overlay_2d(ax_xy)

        ax_xy.plot(
            traj_act_rtde[:, 0],
            traj_act_rtde[:, 1],
            color='red', linewidth=1.5, alpha=0.85, label='Actual'
        )

        ax_xy.scatter(wp_arr[:,0], wp_arr[:,1], c='black', s=60, zorder=5)
        for idx_wp in range(n_wp_plot):
            wp_cur  = pnp_waypoints[idx_wp]
            wp_next = pnp_waypoints[(idx_wp + 1) % n_wp_plot]
            dx = wp_next[0] - wp_cur[0]
            dy = wp_next[1] - wp_cur[1]
            dist_xy = np.sqrt(dx**2 + dy**2)
            if dist_xy > 0.01:
                scale = min(0.06, dist_xy * 0.25) / dist_xy
                ax_xy.annotate(
                    '',
                    xy=(wp_cur[0] + dx*scale, wp_cur[1] + dy*scale),
                    xytext=(wp_cur[0], wp_cur[1]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
                )

        for idx_wp, (name, wp) in enumerate(zip(PNP_NAMES, pnp_waypoints)):
            ox, oy = label_offsets_xy[idx_wp]
            ax_xy.annotate(
                f'{idx_wp}:{name}',
                xy=(wp[0], wp[1]),
                xytext=(wp[0] + ox, wp[1] + oy),
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.6)
            )

        ax_xy.scatter(*traj_act_rtde[0,:2],  color='green',  s=100, zorder=6, label='Start')
        ax_xy.scatter(*traj_act_rtde[-1,:2], color='purple', s=100, zorder=6, label='End')
        ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('Top View (XY)', fontsize=11)
        ax_xy.axis('equal'); ax_xy.grid(True, alpha=0.3)
        legend_xy = [
            Line2D([0],[0], color='red', linewidth=2, label='Actual Trajectory'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
                markersize=8, label='Waypoints'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='green',
                markersize=8, label='Start'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='purple',
                markersize=8, label='End'),
        ]
        ax_xy.legend(handles=legend_xy, fontsize=8, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '01b_pnp_traj_2d.png'), dpi=150)
        plt.close()

    # ==========================================
    # PLOT 2 — TRACKING ERROR per axis (both trajectories)
    # ==========================================
    no_safety    = (R.method == "none")
    cbf_nom_mask = cbf_active_arr == 0
    axis_labels  = ['X', 'Y', 'Z']
    pos_min_vals = POS_MIN
    pos_max_vals = POS_MAX

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f'Tracking Error | {ctrl_name} | {scenario_name} | {method_name}',
                 fontsize=13, fontweight='bold')
    ax.plot(t_log, error_xyz[:,0], label='ex', color='blue',   linewidth=1.0)
    ax.plot(t_log, error_xyz[:,1], label='ey', color='orange', linewidth=1.0)
    ax.plot(t_log, error_xyz[:,2], label='ez', color='green',  linewidth=1.0)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    if not no_safety:
        e_range = max(np.max(np.abs(error_xyz)), 1.0)
        ax.fill_between(t_log, -e_range * 1.1, e_range * 1.1,
                        where=(cbf_active_arr > 0),
                        color='red', alpha=0.07, label='Filter correcting')
    ax.set_ylabel('Error (mm)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Error per Axis')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '02_tracking_error.png'), dpi=150)
    plt.close()

    if is_pnp:
        pnp_phase_arr = np.array(pnp_phase_log)
        t_arr = np.array(t_log)

        # PLOT 02c: tracking per axis (JAX frame: X and Y flipped)
        frame_sign    = np.array([-1.0, -1.0, 1.0])
        traj_act_plot = traj_act_rtde * frame_sign
        traj_des_plot = trajectory_target * frame_sign

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
        fig.suptitle(f'Tracking per Axis | {ctrl_name} | {scenario_name} | {method_name}',
                    fontsize=13, fontweight='bold')

        for idx in range(3):
            ax = axes[idx]

            ax.plot(t_log, traj_des_plot[:, idx],
                    color='blue', linestyle='--', linewidth=1.3,
                    label='Desired', alpha=0.8)
            ax.plot(t_log, traj_act_plot[:, idx],
                    color='red', linewidth=1.2,
                    label='Actual', alpha=0.9)

            if scenario in ("ee_containment", "wb_containment", "wb_combine"):
                ax.axhline(pos_min_vals[idx], color='green', linestyle='--',
                        linewidth=1.0, alpha=0.7,
                        label=f'{axis_labels[idx]}_min={pos_min_vals[idx]:.2f}m')
                ax.axhline(pos_max_vals[idx], color='green', linestyle='--',
                        linewidth=1.0, alpha=0.7,
                        label=f'{axis_labels[idx]}_max={pos_max_vals[idx]:.2f}m')

            for i in range(1, len(t_log)):
                if pnp_phase_arr[i] != pnp_phase_arr[i-1]:
                    ax.axvline(t_arr[i], color='gray', linestyle='--',
                            linewidth=0.8, alpha=0.5)

            e_ax        = error_xyz[:, idx]
            rmse_ax     = np.sqrt(np.mean(e_ax**2))
            rmse_ax_nom = np.sqrt(np.mean(e_ax[cbf_nom_mask]**2)) if cbf_nom_mask.sum() > 0 else np.nan
            stats_ax    = (f"RMSE total   : {rmse_ax:.2f} mm\n"
                        f"RMSE nominal : {rmse_ax_nom:.2f} mm")
            ax.text(0.98, 0.97, stats_ax, transform=ax.transAxes,
                fontsize=8, fontfamily='monospace', va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

            ax.set_ylabel(f'Pos {axis_labels[idx]} (m)')
            ax.legend(fontsize=7, ncol=3, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'({chr(65+idx)}) Axis {axis_labels[idx]}')

        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '02c_tracking_xyz.png'), dpi=150)
        plt.close()

    else:

        # Plot 02c: tracking per axis (sinusoidal_3d, RTDE frame)
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
        fig.suptitle(f'Tracking per Axis | {ctrl_name} | {scenario_name} | {method_name}',
                    fontsize=13, fontweight='bold')

        for idx in range(3):
            ax = axes[idx]

            ax.plot(t_log, trajectory_target[:, idx],
                    color='blue', linestyle='--', linewidth=1.3,
                    label='Desired', alpha=0.8)
            ax.plot(t_log, traj_act_rtde[:, idx],
                    color='red', linewidth=1.2,
                    label='Actual', alpha=0.9)

            if scenario in ("ee_containment", "wb_containment", "wb_combine"):
                ax.axhline(pos_min_vals[idx], color='green', linestyle='--',
                           linewidth=1.0, alpha=0.7,
                           label=f'{axis_labels[idx]}_min={pos_min_vals[idx]:.2f}m')
                ax.axhline(pos_max_vals[idx], color='green', linestyle='--',
                           linewidth=1.0, alpha=0.7,
                           label=f'{axis_labels[idx]}_max={pos_max_vals[idx]:.2f}m')

            if not no_safety:
                data_min = min(np.min(trajectory_target[:, idx]), np.min(traj_act_rtde[:, idx]))
                data_max = max(np.max(trajectory_target[:, idx]), np.max(traj_act_rtde[:, idx]))
                margin   = (data_max - data_min) * 0.15
                ax.fill_between(t_log, data_min - margin, data_max + margin,
                               where=(cbf_active_arr > 0),
                               color='red', alpha=0.07, label='Filter correcting')

            e_ax        = error_xyz[:, idx]
            rmse_ax     = np.sqrt(np.mean(e_ax**2))
            rmse_ax_nom = np.sqrt(np.mean(e_ax[cbf_nom_mask]**2)) if cbf_nom_mask.sum() > 0 else np.nan
            stats_ax    = (f"RMSE total   : {rmse_ax:.2f} mm\n"
                           f"RMSE nominal : {rmse_ax_nom:.2f} mm")
            ax.text(0.98, 0.97, stats_ax, transform=ax.transAxes,
                   fontsize=8, fontfamily='monospace', va='top', ha='right',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

            ax.set_ylabel(f'Pos {axis_labels[idx]} (m)')
            ax.legend(fontsize=7, ncol=3, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'({chr(65+idx)}) Axis {axis_labels[idx]}')

        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '02c_tracking_xyz.png'), dpi=150)
        plt.close()

    # Plot 4b: Nullspace component per joint
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Nullspace Component per Joint', fontsize=14, fontweight='bold')
    for j in range(6):
        ax = axes[j//2, j%2]
        ax.plot(t_log, dq_ns_log[:, j], color='purple', linewidth=1.2, label=f'dq_ns J{j+1}')
        ax.set_title(f'Joint {j+1}'); ax.set_xlabel('Time (s)'); ax.set_ylabel('rad/s')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '04b_nullspace_joint.png'), dpi=150); plt.close()

    # Plot 5: dq_nom vs dq_safe vs dq_cmd per joint
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    title_suffix = "Nom vs Cmd (No-CBF)" if no_safety else "Nom vs Safe vs Cmd"
    fig.suptitle(f'{title_suffix} Joint Velocities — {ctrl_name}', fontsize=13, fontweight='bold')

    for i in range(6):
        ax = axes[i//2, i%2]
        ax.plot(t_log, dq_nom_arr[:,i], color='tab:blue', label='Nom',
                 linewidth=1.4, alpha=0.6, zorder=1)
        if not no_safety:
            ax.plot(t_log, dq_safe_arr[:,i], color='tab:orange', label='Safe',
                     linewidth=1.2, alpha=0.8, zorder=2)
        ax.plot(t_log, dq_cmd_log[:,i], color='tab:red', label='Cmd',
                 linewidth=1.0, alpha=0.9, zorder=3)
        ax.plot(t_log, dq_act_log[:,i], color='tab:green', label='Actual',
                 linewidth=1.2, alpha=0.8, zorder=4)
        ax.set_title(f'Joint {i+1}'); ax.set_xlabel('Time (s)'); ax.set_ylabel('vel (rad/s)')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '05_vel_compare.png'), dpi=150); plt.close()

    # Plot 6: Joint positions
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.plot(t_log, np.rad2deg(q_log[:,i]), label=f'Joint {i+1}', linewidth=1.2, color=colors[i])
    plt.title("Joint Positions", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)"); plt.ylabel("Position (deg)"); plt.grid(True, alpha=0.3); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '06_joint_positions.png'), dpi=150); plt.close()

    # Plot 7: Singularity Detection
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Singularity Detection', fontsize=15, fontweight='bold')
    ax1.semilogy(t_log, sigma_min_log, color='blue', linewidth=1.5, label='σ_min')
    ax1.axhline(EPS, color='red', linestyle='--', linewidth=2, label=f'EPS={EPS:.1e}')
    ax1.set_ylabel('σ_min (log)'); ax1.legend(); ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title('(A) Minimum Singular Value')
    ax2.fill_between(t_log, 0, singularity_status_log,   where=singularity_status_log==1, color='red',   alpha=0.5, label='Singular')
    ax2.fill_between(t_log, 0, 1-singularity_status_log, where=singularity_status_log==0, color='green', alpha=0.3, label='Active')
    ax2.set_ylabel('Task Status'); ax2.set_yticks([0,1]); ax2.set_yticklabels(['STOPPED','ACTIVE'])
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_title('(B) Task Control State')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '07_singularity.png'), dpi=150); plt.close()

    # Plot 8: Sliding Surface
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f'Sliding Surface σ = ė + Λe — {ctrl_name}', fontsize=14, fontweight='bold')
    axes[0].plot(t_log, log_sigma_smc[:,0], label='σx', linewidth=1.2, color='red')
    axes[0].plot(t_log, log_sigma_smc[:,1], label='σy', linewidth=1.2, color='green')
    axes[0].plot(t_log, log_sigma_smc[:,2], label='σz', linewidth=1.2, color='blue')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('σ'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('(A) Sliding Surface Components')
    axes[1].plot(t_log, sigma_norms, color='purple', linewidth=1.3, label='||σ||')
    axes[1].set_ylabel('||σ||'); axes[1].set_xlabel('Time (s)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_title('(B) Sliding Surface Norm')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '08_sliding_surface.png'), dpi=150); plt.close()

    # Plot 9: Phase Plane
    lambda_val = np.diag(Lambda)[0]

    if is_pnp:
        pnp_phase_arr = np.array(pnp_phase_log)
        pnp_cycle_arr = np.array(pnp_cycle_log)
        n_phases = len(PNP_NAMES)
        fig, axes = plt.subplots(n_phases, 3, figsize=(15, 4 * n_phases))
        fig.suptitle(f'Phase Plane per Segment (e vs ė) — {ctrl_name}', fontsize=14, fontweight='bold')

        for ph in range(n_phases):
            if ph == 0:
                mask = (pnp_phase_arr == ph) & (pnp_cycle_arr == 1)
            else:
                mask = (pnp_phase_arr == ph) & (pnp_cycle_arr == 0)
            if not np.any(mask):
                for i in range(3):
                    axes[ph][i].set_visible(False)
                continue

            for i, label in enumerate(['X', 'Y', 'Z']):
                e_ax  = error_xyz[mask, i] / 1000.0
                ed_ax = log_edot[mask, i]

                axes[ph][i].plot(e_ax, ed_ax, color='blue', linewidth=1.2, zorder=2)
                axes[ph][i].scatter(e_ax[0],  ed_ax[0],  color='green', s=70, zorder=5,
                                    marker='o', edgecolors='black', label='Start')
                axes[ph][i].scatter(e_ax[-1], ed_ax[-1], color='red',   s=70, zorder=5,
                                    marker='X', edgecolors='black', label='End')
                e_range = np.linspace(min(e_ax), max(e_ax), 100)
                axes[ph][i].plot(e_range, -lambda_val * e_range, 'r--', label='Sliding surface', zorder=1)
                axes[ph][i].set_xlabel(f'e_{label} (m)')
                axes[ph][i].set_ylabel(f'ė_{label} (m/s)')
                axes[ph][i].set_title(f'{PNP_NAMES[(ph-1)%n_wp]} → {PNP_NAMES[ph]} — Axis {label}')
                axes[ph][i].legend(fontsize=8)
                axes[ph][i].grid(True)
                axes[ph][i].xaxis.set_major_locator(MaxNLocator(nbins=7))
                axes[ph][i].yaxis.set_major_locator(MaxNLocator(nbins=7))
                axes[ph][i].ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))

        plt.tight_layout(rect=[0, 0, 1, 0.97])

    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Phase Plane per Segment (e vs ė) — {ctrl_name}',
             fontsize=14, fontweight='bold', y=0.995)
        for i, label in enumerate(['X', 'Y', 'Z']):
            e_ax  = error_xyz[:, i] / 1000.0
            ed_ax = log_edot[:, i]
            axes[i].plot(e_ax, ed_ax, color='blue', linewidth=1.2, zorder=2)
            axes[i].scatter(e_ax[0],  ed_ax[0],  color='green', s=70, zorder=5,
                              marker='o', edgecolors='black', label='Start')
            axes[i].scatter(e_ax[-1], ed_ax[-1], color='red',   s=70, zorder=5,
                              marker='X', edgecolors='black', label='End')
            e_range = np.linspace(min(e_ax), max(e_ax), 100)
            axes[i].plot(e_range, -lambda_val * e_range, 'r--', label='Sliding surface', zorder=1)
            axes[i].set_xlabel(f'e_{label} (m)'); axes[i].set_ylabel(f'ė_{label} (m/s)')
            axes[i].set_title(f'Axis {label}'); axes[i].legend(fontsize=8); axes[i].grid(True)

        plt.tight_layout()

    plt.savefig(os.path.join(exp_folder, '09_phase_plane.png'), dpi=150); plt.close()

    # Plot 10: Jacobian Rank
    plt.figure(figsize=(10, 5))
    plt.step(t_log, rank_Jp_log, where='post', label='rank(Jp 3x6)', linewidth=1.5)
    plt.step(t_log, rank_J_log,  where='post', label='rank(J 6x6)',  linewidth=1.5, linestyle='--')
    plt.ylim(-0.5, 6.5); plt.yticks([0,1,2,3,4,5,6])
    plt.xlabel('Time (s)'); plt.ylabel('Rank')
    plt.title('Jacobian Rank over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '10_jacobian_rank.png'), dpi=150); plt.close()

    # Plot 10b: Control Effort
    dq_act_norm = np.linalg.norm(dq_act_log, axis=1)
    cumulative_J_cmd = np.cumsum(dq_cmd_norm**2) * dt
    cumulative_J_act = np.cumsum(dq_act_norm**2) * dt
    control_effort_act = np.sum(dq_act_norm**2) * dt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f'Control Effort — {ctrl_name}', fontsize=14, fontweight='bold')

    # (A) norm per step
    axes[0].plot(t_log, dq_cmd_norm, color='darkblue', linewidth=1.2, label='||u_cmd||', alpha=0.8)
    axes[0].plot(t_log, dq_act_norm, color='red',      linewidth=1.2, label='||u_act||', alpha=0.8)
    axes[0].axhline(np.mean(dq_cmd_norm), color='blue', linestyle='--',
                    linewidth=1.2, label=f'Mean cmd={np.mean(dq_cmd_norm):.4f}')
    axes[0].axhline(np.mean(dq_act_norm), color='orange', linestyle='--',
                    linewidth=1.2, label=f'Mean act={np.mean(dq_act_norm):.4f}')
    axes[0].set_ylabel('||dq|| (rad/s)')
    axes[0].legend(ncol=2); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('(A) Control Effort Norm per Step')

    # (B) cumulative
    axes[1].plot(t_log, cumulative_J_cmd, color='darkblue', linewidth=1.3,
                label=f'J_cmd total={np.sum(dq_cmd_norm**2)*dt:.4f}')
    axes[1].plot(t_log, cumulative_J_act, color='red', linewidth=1.3,
                label=f'J_act total={control_effort_act:.4f}')
    axes[1].set_ylabel('Cumulative J (rad²/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('(B) Cumulative Control Effort J')

    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '10b_control_effort.png'), dpi=150); plt.close()

    # Plot 11: Disturbance (only with a disturbance)
    if R.dist != "none":
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.suptitle(f'Injected Disturbance — {R.dist_name}', fontsize=14, fontweight='bold')
        for j in range(6):
            axes[0].plot(t_log, log_disturbance[:,j], label=f'd{j+1}', color=colors[j], linewidth=1.2)
        axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[0].set_ylabel('d (rad/s)'); axes[0].legend(ncol=3); axes[0].grid(True, alpha=0.3)
        axes[0].set_title('(A) Disturbance per Joint')
        axes[1].plot(t_log, np.linalg.norm(log_disturbance, axis=1), color='purple', linewidth=1.3, label='||d||')
        axes[1].set_ylabel('||d|| (rad/s)'); axes[1].set_xlabel('Time (s)')
        axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_title('(B) Disturbance Norm')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '11_disturbance.png'), dpi=150); plt.close()

    # Plot 12: CBF / Safety Analysis
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Safety Analysis — Scenario: {scenario_name} | Method: {method_name} | α={CBF_ALPHA}',
                 fontsize=13, fontweight='bold')

    # (A) barrier values
    if scenario == "ee_containment":
        h_labels = ['x≤xmax','x≥xmin','y≤ymax','y≥ymin','z≤zmax','z≥zmin']
        for idx in range(h_vals_arr.shape[1]):
            axes[0].plot(t_log, h_vals_arr[:, idx], label=h_labels[idx], linewidth=1.0, alpha=0.8)
    elif scenario == "wb_obstacle":
        for i in range(h_vals_arr.shape[1]):
            axes[0].plot(t_log, h_vals_arr[:, i], linewidth=0.7, alpha=0.5)
    elif scenario == "wb_containment":
        for i in range(h_vals_arr.shape[1]):
            axes[0].plot(t_log, h_vals_arr[:, i], linewidth=0.5, alpha=0.3)
    elif scenario == "wb_combine":
        n_spheres_plot = h_vals_arr.shape[1] // 7
        h_cont_arr = h_vals_arr[:, :n_spheres_plot * 6]
        h_obs_arr  = h_vals_arr[:, n_spheres_plot * 6:]

        for i in range(h_cont_arr.shape[1]):
            axes[0].plot(t_log, h_cont_arr[:, i],
                        color='blue', linewidth=0.4, alpha=0.2)

        for i in range(h_obs_arr.shape[1]):
            axes[0].plot(t_log, h_obs_arr[:, i],
                        color='red', linewidth=0.6, alpha=0.4)

        # legend proxies
        axes[0].plot([], [], color='blue',  alpha=0.6, label='Containment constraints')
        axes[0].plot([], [], color='red',   alpha=0.6, label='Obstacle constraints')

        h_cont_min = np.min(h_cont_arr, axis=1)
        h_obs_min  = np.min(h_obs_arr,  axis=1)
        axes[0].plot(t_log, h_cont_min, color='blue',  linewidth=1.5,
                    linestyle='--', label='min(h) containment')
        axes[0].plot(t_log, h_obs_min,  color='red',   linewidth=1.5,
                    linestyle='--', label='min(h) obstacle')

    axes[0].axhline(0, color='k', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('h(z)')
    if axes[0].get_legend_handles_labels()[0]:   # wb_obstacle / wb_containment draw no labelled lines
        axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('(A) Barrier Values h(z) — all constraints')

    if scenario == "wb_combine":
        n_s = h_vals_arr.shape[1] // 7
        h_cont_min_step = np.min(h_vals_arr[:, :n_s*6], axis=1)
        h_obs_min_step  = np.min(h_vals_arr[:, n_s*6:], axis=1)
        axes[1].plot(t_log, h_cont_min_step, color='blue', linewidth=1.0,
                    linestyle='--', alpha=0.7, label='min(h) containment')
        axes[1].plot(t_log, h_obs_min_step, color='orange', linewidth=1.0,
                    linestyle='--', alpha=0.7, label='min(h) obstacle')
        axes[1].plot(t_log, h_min_per_step, color='red', linewidth=1.5,
                    linestyle='--', label='min(h) overall')

    axes[1].axhline(0, color='k', linestyle='--', linewidth=1.5)
    axes[1].fill_between(t_log, h_min_per_step, 0,
        where=(h_min_per_step < 0), color='red', alpha=0.3, label='Violation')
    axes[1].set_ylabel('min(h)')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('(B) Minimum Barrier Value')

    # summary box
    cbf_active_pct  = np.mean(cbf_active_arr) * 100
    h_min_overall   = np.min(h_min_per_step)

    if scenario == "wb_combine":
        n_s = h_vals_arr.shape[1] // 7
        h_cont_min_val = np.min(h_vals_arr[:, :n_s*6])
        h_obs_min_val  = np.min(h_vals_arr[:, n_s*6:])
        summary_text = (
            f"min(h) overall   : {h_min_overall:.4f}\n"
            f"min(h) contain   : {h_cont_min_val:.4f}\n"
            f"min(h) obstacle  : {h_obs_min_val:.4f}\n"
            f"Filter correcting: {cbf_active_pct:.1f}%"
        )
    else:
        summary_text = (
            f"min(h)           : {h_min_overall:.4f}\n"
            f"Filter correcting: {cbf_active_pct:.1f}%"
        )

    axes[1].text(0.98, 0.05, summary_text,
                transform=axes[1].transAxes,
                fontsize=8, fontfamily='monospace',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '12_cbf_analysis.png'), dpi=150); plt.close()

    # Plot 13: Real vs Digital Twin
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    fig.suptitle(
        f'Real Robot vs Digital Twin | {ctrl_name} | {scenario_name} | {method_name}',
        fontsize=13,
        fontweight='bold'
    )

    ee_err_xyz   = np.abs(traj_act_rtde - twin_pos_arr) * 1000.0   # (N,3) mm
    mean_err_xyz = np.mean(ee_err_xyz, axis=0)
    rmse_err_xyz = np.sqrt(np.mean(ee_err_xyz**2, axis=0))

    joint_err_deg  = np.rad2deg(np.abs(q_log - twin_q_arr))         # (N,6) deg
    mean_joint_err = np.mean(joint_err_deg, axis=0)

    # (A) end-effector position
    axis_labels = ['X', 'Y', 'Z']

    colors_real = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c'   # green
    ]

    for i, lbl in enumerate(axis_labels):

        axes[0].plot(
            t_log,
            traj_act_rtde[:, i],
            color=colors_real[i],
            linewidth=1.5,
            label=f'{lbl} real'
        )

        axes[0].plot(
            t_log,
            twin_pos_arr[:, i],
            color=colors_real[i],
            linestyle='--',
            linewidth=1.5,
            alpha=0.9,
            label=f'{lbl} twin'
        )

    axes[0].set_ylabel('Position (m)')
    axes[0].grid(True, alpha=0.3)

    axes[0].set_title(
        '(A) End-Effector Position: Real vs Digital Twin'
    )

    axes[0].legend(
        fontsize=8,
        ncol=3,
        loc='upper right'
    )

    # (B) joint position
    joint_colors = [
        '#d62728',  # J1
        '#1f77b4',  # J2
        '#2ca02c',  # J3
        '#9467bd',  # J4
        '#ff7f0e',  # J5
        '#8c564b'   # J6
    ]

    for j in range(6):

        axes[1].plot(
            t_log,
            q_log[:, j],
            color=joint_colors[j],
            linewidth=1.2,
            label=f'J{j+1} real'
        )

        axes[1].plot(
            t_log,
            twin_q_arr[:, j],
            color=joint_colors[j],
            linestyle='--',
            linewidth=1.2,
            alpha=0.85,
            label=f'J{j+1} twin'
        )

    axes[1].set_ylabel('Joint Position (rad)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    axes[1].set_title(
        '(B) Joint Position: Real vs Digital Twin'
    )

    axes[1].legend(
        fontsize=7,
        ncol=4,
        loc='upper left'
    )

    # statistics boxes
    ee_stats = 'EE Error per Axis (mm)\n' + '\n'.join(
        f'e{"XYZ"[i]}: mean={mean_err_xyz[i]:.2f}  rmse={rmse_err_xyz[i]:.2f}'
        for i in range(3)
    )
    axes[0].text(
        0.985, 0.03, ee_stats,
        transform=axes[0].transAxes,
        fontsize=8, fontfamily='monospace',
        va='bottom', ha='right',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.75)
    )

    jnt_stats = 'Joint Mean Error (deg)\n' + '\n'.join(
        f'J{i+1} = {mean_joint_err[i]:.3f}°' for i in range(6)
    )
    axes[1].text(
        0.985, 0.03, jnt_stats,
        transform=axes[1].transAxes,
        fontsize=8, fontfamily='monospace',
        va='bottom', ha='right',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.75)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(exp_folder, '13_real_vs_twin.png'),
        dpi=150,
        bbox_inches='tight'
    )

    plt.close()

    debug("[PLOTTING] Done.\n")
