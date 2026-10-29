"""One experiment run: setup, 125 Hz control loop, outputs.

Step order inside the loop: numbered comments.
Scenarios: ee_containment, wb_obstacle, wb_containment, wb_combine.
"""

import time
import traceback
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from kossmc_issf.config.defaults import BASE_Z_OFFSET, STATUS_EVERY
from kossmc_issf.config.naming import (
    CTRL_NAMES, DIST_NAMES, METHOD_NAMES, SCENARIO_NAMES, TRAJ_NAMES, get_robot_source,
)
from kossmc_issf.buildinfo import git_commit, kossmc_issf_version
from kossmc_issf.console import debug as dbg, info, is_debug, set_debug
from kossmc_issf.core.controllers import lyapunov, task_space_control
from kossmc_issf.core.disturbance import disturbance
from kossmc_issf.core.kinematics import get_robot_model, rpy_from_R, singularity_analysis, update_task_active
from kossmc_issf.core.nullspace import joint_limit_gradient, nullspace_velocity
from kossmc_issf.core.safety import (
    build_config, build_safety_filter, compute_h_vals_numpy, safety_step, warmup,
)
from kossmc_issf.datalog.csv_io import save_all_csv
from kossmc_issf.datalog.folders import create_experiment_folder
from kossmc_issf.datalog.logger import RunLog, to_arrays
from kossmc_issf.datalog.summary import RunInfo, write_experiment_info
from kossmc_issf.robot.rtde_interface import move_init
from kossmc_issf.sim import visuals
from kossmc_issf.sim.digital_twin import twin_state, twin_step
from kossmc_issf.trajectory.pnp import PNP_NAMES, PnPInterpolated, generate_interpolated
from kossmc_issf.trajectory.trajectories import sinusoidal_3d


@dataclass(frozen=True)
class Selection:
    traj: str     # "sinusoidal_3d" / "pnp_interpolated"
    ctrl: str     # "p" / "smc_sign" / "smc_quasi"
    method: str   # "cbf" / "issf" / "none"
    dist: str     # "gaussian" / "sinusoidal" / "none"
    ip: str
    scenario: str = "wb_combine"   # "ee_containment" / "wb_obstacle" / "wb_containment" / "wb_combine"


WB_SCENARIOS = ("wb_obstacle", "wb_containment", "wb_combine")   # use the collision spheres
FLIP_XY = np.array([-1.0, -1.0, 1.0])   # RTDE base frame <-> base_link


# UR5e joint limits (datasheet), only for the --debug joint-limit warning
Q_MIN = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
Q_MAX = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])


def filter_state(method, cbf_active):
    """Filter label of the --debug status lines: "none" (no filter), "correcting" or "pass"."""
    if method == "none":
        return "none"
    return "correcting" if cbf_active else "pass"


def _debug_breakdown(h_vals, filt, q):
    """Scenario 5 barrier breakdown + joint-limit warning (troubleshooting output)."""
    q_deg = np.rad2deg(q)
    for ji in range(6):
        margin_min = np.rad2deg(q[ji] - Q_MIN[ji])
        margin_max = np.rad2deg(Q_MAX[ji] - q[ji])
        margin = min(margin_min, margin_max)

    idx = np.argmin(h_vals)
    n_spheres = h_vals.shape[0] // 7  # 6 containment + 1 obs per sphere

    h_cont = h_vals[:n_spheres * 6]
    h_obs  = h_vals[n_spheres * 6:]

    if idx < n_spheres * 6:
        if idx < 3 * n_spheres:
            sphere_id = idx // 3
            axis_id   = idx % 3
            side = "min"
        else:
            idx2 = idx - 3 * n_spheres
            sphere_id = idx2 // 3
            axis_id   = idx2 % 3
            side = "max"
        axis_name = ["x","y","z"][axis_id]
        source = f"CONTAINMENT sphere={sphere_id} boundary={axis_name}_{side}"
    else:
        obs_sphere_id = idx - n_spheres * 6
        source = f"OBSTACLE sphere={obs_sphere_id}"

    print(
        f"[WB-COMBINE] "
        f"min_h={h_vals[idx]:.4f} "
        f"| {source} "
        f"| h_cont_min={h_cont.min():.4f} "
        f"| h_obs_min={h_obs.min():.4f} "
        f"| filter={filt}"
    )
    # known issue: margin/ji are left over from the loop, so only joint 6 is checked
    if margin < 10.0:  # warn within 10° of the limit
        print(f"  ⚠ Joint {ji+1}: {q_deg[ji]:.1f}°  "
            f"limit=[{np.rad2deg(Q_MIN[ji]):.0f}°, {np.rad2deg(Q_MAX[ji]):.0f}°]  "
            f"margin={margin:.1f}°")


def _debug_status_scenario(scenario, h_vals, filt, q):
    """Troubleshooting line(s) printed with the status line (--debug). filt: see filter_state."""
    if scenario == "wb_obstacle":
        print(f"[WB] min_h={h_vals.min():.4f} | idx={h_vals.argmin()} | filter={filt}")
    elif scenario == "wb_containment":
        idx = np.argmin(h_vals)

        n_spheres = len(h_vals)//6

        if idx < 3*n_spheres:
            sphere_id = idx // 3
            axis_id   = idx % 3
            side = "min"
        else:
            idx2 = idx - 3*n_spheres
            sphere_id = idx2 // 3
            axis_id   = idx2 % 3
            side = "max"

        axis_name = ["x","y","z"][axis_id]

        print(
            f"[WB-CONT] "
            f"min_h={h_vals[idx]:.4f} "
            f"| sphere={sphere_id} "
            f"| boundary={axis_name}_{side} "
            f"| filter={filt}"
        )
    elif scenario == "wb_combine":
        _debug_breakdown(h_vals, filt, q)


def _precheck(scenario, cfg, q_home, rm, P):
    """Barrier values at the start pose (pre-check text, --debug). Returns h_init."""
    q_home_jax = jnp.array(q_home)
    POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS = P.POS_MIN, P.POS_MAX, P.OBS_CENTER, P.OBS_RADIUS
    if scenario == "ee_containment":
        # FK (RTDE frame) converted to base_link, as the logged h and the filter
        h_init = compute_h_vals_numpy(rm.forward_kinematics(q_home)[0] * FLIP_XY, POS_MIN, POS_MAX)
        labels = ['x≤xmax','x≥xmin','y≤ymax','y≥ymin','z≤zmax','z≥zmin']
        dbg("  Barrier values at home pose:")
        for lbl, hv in zip(labels, h_init):
            status = "OK" if hv > 0 else "VIOLATION"
            dbg(f"    {lbl}: h={hv:.4f}  [{status}]")
        if np.any(h_init < 0):
            dbg("  WARNING: EE already outside the box! Adjust POS_MIN/POS_MAX.")
    elif scenario == "wb_obstacle":
        h_init = np.array(cfg.h_1(q_home_jax))
        dbg(f"  WB min_h at home: {h_init.min():.4f}  idx={h_init.argmin()}")
        dbg(f"  {'OK — all spheres clear' if h_init.min() > 0 else 'WARNING: sphere overlap obstacle!'}")
        if np.any(h_init < 0):
            dbg("  WARNING: robot already in the collision zone at the home pose!")
            dbg(f"  Check OBS_CENTER={OBS_CENTER} OBS_RADIUS={OBS_RADIUS}")
    elif scenario == "wb_containment":
        h_init = np.array(cfg.h_1(q_home_jax))

        dbg(f"WB containment min_h at home: {h_init.min():.4f}")

        idx = np.argmin(h_init)
        n_spheres = len(h_init) // 6

        if idx < 3 * n_spheres:
            sphere_id, axis_id, side = idx // 3, idx % 3, "min"
        else:
            idx2 = idx - 3 * n_spheres
            sphere_id, axis_id, side = idx2 // 3, idx2 % 3, "max"

        axis_name = ["x", "y", "z"][axis_id]
        dbg(f"  → Worst: sphere={sphere_id} | boundary={axis_name}_{side} | h={h_init[idx]:.4f}")

        sphere_data = np.array(rm.manipulator.link_collision_data(q_home_jax))
        pos = sphere_data[sphere_id, :3] * np.array([-1.0, -1.0, 1.0])
        r   = sphere_data[sphere_id, 3]
        dbg(f"  → Sphere {sphere_id} pos(RTDE frame)={pos.round(4)}  radius={r:.4f}")
        dbg(f"  → POS_MIN={POS_MIN}  POS_MAX={POS_MAX}")

        if np.any(h_init < 0):
            dbg("WARNING: sphere already outside box")
    else:
        h_init = np.array(cfg.h_1(q_home_jax))

        n_spheres    = len(h_init) // 7  # 6 containment + 1 obs per sphere
        h_containment = h_init[:n_spheres * 6]
        h_obstacle    = h_init[n_spheres * 6:]

        dbg(f"WB Containment min_h : {h_containment.min():.4f}")
        dbg(f"WB Obstacle    min_h : {h_obstacle.min():.4f}")
        dbg(f"Overall        min_h : {h_init.min():.4f}")

        if np.any(h_init < 0):
            idx = np.argmin(h_init)
            if idx < n_spheres * 6:
                dbg(f"  → Violation in a CONTAINMENT constraint")
            else:
                dbg(f"  → Violation in an OBSTACLE constraint")
    return h_init


def run(P, sel, robot, env, metrics, debug=False, input_fn=input, clock=time.time, sleep=time.sleep,
        out_root="experiments", now=None, plot_fn=None):
    """Run one experiment.

    P: config.presets.Params (with P.trajectory == sel.traj); sel: Selection;
    robot: connected-on-demand robot (robot.rtde_interface.RTDERobot or a test double);
    env: sim.pybullet_env.PyBulletEnv; metrics: monitoring.prometheus.Metrics.
    Returns the output folder, or None if the run stopped before the control loop.
    """
    assert P.trajectory == sel.traj
    set_debug(debug)
    version, commit = kossmc_issf_version(), git_commit()
    rm = get_robot_model()
    robot_manipulator = rm.manipulator
    is_pnp = sel.traj == "pnp_interpolated"
    dt, duration = P.dt, P.duration
    POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS = P.POS_MIN, P.POS_MAX, P.OBS_CENTER, P.OBS_RADIUS
    CBF_ALPHA = P.CBF_ALPHA

    metrics.set_static(OBS_CENTER, OBS_RADIUS, POS_MIN, POS_MAX)

    dbg(f"=== PARAMETERS ===")
    dbg(f"dt={dt}s | Kp={P.Kp} | K_null={P.K_null} | Amp={P.amplitude}m | Freq={P.frequency_traj}Hz")
    dbg(f"CBF Box: min={POS_MIN.tolist()} max={POS_MAX.tolist()} alpha={CBF_ALPHA}\n")
    dbg(f"CBF Obstacle: center={OBS_CENTER.tolist()} radius={OBS_RADIUS} alpha={CBF_ALPHA}\n")

    # ==========================================
    # FOLDER SETUP
    # ==========================================
    traj_name     = TRAJ_NAMES[sel.traj]
    ctrl_name     = CTRL_NAMES[sel.ctrl]
    dist_name     = DIST_NAMES[sel.dist]
    robot_source  = get_robot_source(sel.ip)
    scenario_name = SCENARIO_NAMES[sel.scenario]
    method_name   = METHOD_NAMES[sel.method]
    exp_folder, timestamp = create_experiment_folder(
        traj_name, ctrl_name, dist_name, robot_source, scenario_name, method_name,
        root=out_root, now=now
    )
    dbg(f"\n[FOLDER] {exp_folder}\n")

    # ==========================================
    # RTDE CONNECTION
    # ==========================================
    if not robot.connect():
        return None

    # ==========================================
    # INITIAL CONFIGURATION
    # ==========================================
    q_init = P.q_init
    dbg("[RTDE] Moving to initial position...")
    move_init(robot, env, q_init, mode="joint", speed=0.5, accel=0.5, sleep=sleep)

    q_after = robot.get_q()

    dbg("\n=== INIT CHECK ===")
    dbg("q_target    :", np.round(q_init, 3))
    dbg("q_actual    :", np.round(q_after, 3))
    dbg("joint error :", np.linalg.norm(q_after - np.array(q_init)))

    try:
        dbg("robot mode  :", robot.get_robot_mode())
        dbg("safety mode :", robot.get_safety_mode())
    except:
        pass

    dbg("==================\n")

    tcp_init = robot.get_tcp_pose()
    center_x, center_y, center_z = tcp_init[:3]
    dbg(f"[RTDE] Initial TCP: X={center_x:.4f}, Y={center_y:.4f}, Z={center_z:.4f}\n")
    info(f"[INIT] q error {np.linalg.norm(q_after - np.array(q_init)):.4f} rad | "
         f"TCP X={center_x:.4f} Y={center_y:.4f} Z={center_z:.4f}")

    # HOME := actual TCP after moveJ
    pnp_waypoints = [w.copy() for w in P.pnp_waypoints]
    pnp_waypoints[0] = np.array([center_x, center_y, center_z])
    n_wp = len(pnp_waypoints)

    # ==========================================
    # GENERATE PNP TRAJECTORY
    # ==========================================
    dbg("\n=== AUTO PNP TIMING ===")
    pnp_seg_pos, pnp_seg_vel, PNP_SEGMENT_TIMES, seg_lengths = generate_interpolated(
        pnp_waypoints, P.PNP_SPEED, dt)
    for i in range(n_wp):
        dist, T_seg = seg_lengths[i], PNP_SEGMENT_TIMES[i]
        dbg(f"  Seg {i}: {PNP_NAMES[i]:20s} → {PNP_NAMES[(i+1)%n_wp]:20s} "
              f"| L={dist:.3f}m | T={T_seg:.2f}s | v={dist/T_seg:.3f}m/s")

    dbg("\n[PNP] Waypoint verification:")
    dbg(f"  Initial TCP : [{center_x:.4f}, {center_y:.4f}, {center_z:.4f}]")
    for name, wp in zip(PNP_NAMES, pnp_waypoints):
        dist_from_home = np.linalg.norm(wp - pnp_waypoints[0]) * 1000
        dbg(f"  {name:20s}: {wp}  (dist from home: {dist_from_home:.1f}mm)")

    if is_debug():
        # FK (Pinocchio) vs RTDE TCP at the start pose
        q_test = robot.get_q()
        tcp_test = robot.get_tcp_pose()[:3]
        fk_pin = rm.forward_kinematics(q_test)[0]
        dbg(f"RTDE TCP  : {tcp_test}")
        dbg(f"FK Pinoc  : {fk_pin}")
        dbg(f"Error Pinoc vs RTDE: {np.linalg.norm(fk_pin - tcp_test)*1000:.2f} mm")

    # ==========================================
    # BUILD SAFETY FILTER (scenario)
    # ==========================================
    scenario = sel.scenario
    cfg = build_config(scenario, robot_manipulator, P)
    cbf = build_safety_filter(sel.method, cfg, P.ISSF_EPSILON)
    visuals.visualize_safety_region(scenario, POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS)

    timesteps = np.arange(0, duration, dt)

    # ========================
    # PRE-CHECK
    # ========================
    dbg("\n[PRE-CHECK] Verifying CBF setup...")
    q_home = robot.get_q()

    sphere_body_ids = visuals.init_collision_spheres(max_spheres=30)
    if scenario in WB_SCENARIOS:
        visuals.update_collision_spheres(sphere_body_ids, robot_manipulator, q_home)

    h_init = _precheck(scenario, cfg, q_home, rm, P)
    info(f"[PRE-CHECK] h_min {h_init.min():.4f}")

    if np.any(h_init < 0):
        cont = input_fn("  Continue? (y/n): ")
        if cont.lower() != 'y':
            return None

    dbg("[CBF] JIT warm-up...")
    warmup(cbf)
    dbg("[CBF] Ready.\n")

    # ==========================================
    #   PRECOMPUTE TRAJECTORY
    # ==========================================
    if not is_pnp:
        trajectory_target, trajectory_velocity = sinusoidal_3d(
            (center_x, center_y, center_z), P.amplitude, P.frequency_traj, duration, dt)
        x_arr, y_arr, z_arr = trajectory_target.T
        # PyBullet preview
        traj_pb = np.column_stack([-x_arr, -y_arr, z_arr + BASE_Z_OFFSET])
        visuals.draw_trajectory_pybullet(traj_pb[::100].tolist())
        pnp = None
    else:
        trajectory_target = []
        visuals.draw_waypoint_markers(pnp_waypoints, PNP_NAMES)
        pnp = PnPInterpolated(pnp_waypoints, pnp_seg_pos, pnp_seg_vel, P.PNP_THRESH)

    L = RunLog()

    # ==========================================
    # CONTROL LOOP
    # ==========================================
    dbg(f"[CONTROL] Traj={traj_name} | Ctrl={ctrl_name} | Dist={dist_name} | Scenario={scenario_name} | Method={method_name}")
    start_time  = clock()
    task_active = True

    try:
        for i, t in enumerate(timesteps):
            loop_start = clock()

            # 1. READ STATE
            q_actual_rtde, tcp_pose, qd_actual = robot.read_state()
            q = q_actual_rtde.copy()
            x_actual_fk, R_actual = rm.forward_kinematics(q)
            L.traj_act_fk.append(x_actual_fk)
            x_actual_rtde = np.array(tcp_pose[:3])
            L.traj_act_rtde.append(x_actual_rtde)
            L.rpy_rtde_log.append(np.rad2deg(tcp_pose[3:]))

            # 2. SINGULARITY DETECTION
            sigma_min, full_manip, rank_Jp, rank_J = singularity_analysis(rm.full_jacobian(q))
            L.sigma_min_log.append(sigma_min)
            L.full_manip_log.append(full_manip)
            L.rank_Jp_log.append(rank_Jp)
            L.rank_J_log.append(rank_J)

            task_active, event = update_task_active(task_active, sigma_min, P.EPS)
            if event == "deactivated":
                dbg(f"[{t:.2f}s] SINGULAR! σ_min={sigma_min:.4e} → DEACTIVATED")
            elif event == "reactivated":
                dbg(f"[{t:.2f}s] Recovery: σ_min={sigma_min:.4e} → REACTIVATED")
            if event is not None:
                info(f"[SINGULAR] -> {event} (σ_min={sigma_min:.4e})")

            # 3. TRAJECTORY GENERATION
            if not is_pnp:
                target_pos = trajectory_target[i]
                v_des      = trajectory_velocity[i]
                phase_for_log = 0
            else:
                target_pos, v_des, phase_for_log, tr = pnp.step(x_actual_rtde)
                if tr is not None:
                    dbg(f"  → [{t:.2f}s] {PNP_NAMES[tr['from_phase']]} → {PNP_NAMES[tr['to_phase']]} "
                        f"(err={tr['err_mm']:.1f}mm, {tr['reason']})")
                    if tr["cycle_completed"]:
                        dbg(f"  ✓ Cycle {pnp.pnp_cycle} completed")
                    info(f"[PNP] -> {PNP_NAMES[tr['to_phase']]}")
                L.trajectory_target_pnp.append(target_pos.copy())

            L.v_des_log.append(v_des.copy())
            L.pnp_phase_log.append(phase_for_log if is_pnp else 0)
            L.pnp_cycle_log.append(pnp.pnp_cycle if is_pnp else 0)

            # 4. TASK-SPACE CONTROL
            J      = rm.jacobian(q)
            sigma_max = np.linalg.norm(J, ord=2)
            J_pinv = np.linalg.pinv(J)
            delta_x = x_actual_rtde - target_pos
            dq_task, sigma, e_dot = task_space_control(
                sel.ctrl, J, J_pinv, delta_x, qd_actual, v_des, P.Kp, P.Lambda, P.k_smc, P.eps_qsmc)

            L.j_norm_log.append(sigma_max)
            L.log_sigma_smc.append(sigma.copy())
            L.V_lyapunov_log.append(lyapunov(sigma))
            L.log_edot.append(e_dot.copy())

            # 5. NULLSPACE — MANIPULABILITY + JOINT LIMIT
            grad_manip, manip_val = rm.manipulability_gradient(q, J)
            grad_jl = joint_limit_gradient(q, P.q_min, P.q_max, P.q_mid)
            dq_ns = nullspace_velocity(grad_manip, grad_jl, P.K_null, P.K_jl, J, J_pinv, task_active)
            L.manip_log.append(manip_val)

            dq_cmd = dq_task + dq_ns

            # 6-7. BARRIER VALUES (logging) + SAFETY FILTER
            L.dq_nom_log.append(dq_cmd.copy())
            dq_nom = dq_cmd.copy()
            if scenario == "ee_containment":
                x_ee = x_actual_rtde.copy() * np.array([-1.0, -1.0, 1.0])  # RTDE frame -> base_link
                h_log = compute_h_vals_numpy(x_ee, POS_MIN, POS_MAX)
            else:
                h_log = None
            dq_cmd, cbf_active, h_vals = safety_step(cfg, cbf, q, dq_cmd, h_vals=h_log)
            dq_safe = dq_cmd.copy()
            L.dq_safe_log.append(dq_cmd.copy())
            L.cbf_active_log.append(int(cbf_active))
            L.h_vals_log.append(h_vals.copy())

            # 8. DISTURBANCE INJECTION
            d_input = disturbance(sel.dist, std=P.DIST_STD, clip=P.DIST_CLIP, t=t)
            dq_cmd = dq_cmd + d_input
            L.log_disturbance.append(d_input.copy())

            # 9. SEND COMMAND
            L.singularity_status_log.append(0 if task_active else 1)
            L.dq_cmd_log.append(dq_cmd.copy())
            robot.speed_j(dq_cmd)

            L.q_log.append(q.copy())
            L.dq_act_log.append(qd_actual.copy())

            # 10. DIGITAL TWIN (PyBullet) + COLLISION SPHERES
            twin_step(env, q_actual_rtde, qd_actual)
            if scenario in WB_SCENARIOS:
                visuals.update_collision_spheres(sphere_body_ids, robot_manipulator, q)
            q_twin, qd_twin = twin_state(env)

            pos_twin, R_twin = rm.forward_kinematics(q_twin)
            L.twin_q_log.append(q_twin.copy())
            L.twin_qd_log.append(qd_twin.copy())
            L.twin_pos_log.append(pos_twin.copy())
            L.twin_rpy_log.append(np.rad2deg(rpy_from_R(R_twin)))
            L.twin_time_log.append(clock())

            q_diff   = q_actual_rtde - q_twin          # joint angle difference (rad)

            # 11. LOGGING
            L.rpy_log.append(np.rad2deg(rpy_from_R(R_actual)))
            L.dq_ns_log.append(dq_ns.copy())
            L.dq_ns_norm_log.append(np.linalg.norm(dq_ns))

            L.t_log.append(t)

            # 12. PROMETHEUS
            metrics.update(delta_x, target_pos, x_actual_rtde, dq_ns, h_vals, cbf_active, dq_cmd,
                           dq_nom, dq_safe, OBS_RADIUS, sigma, qd_actual, q, q_twin, qd_twin, q_diff,
                           sigma_min, manip_val, pnp.pnp_phase if is_pnp else 0, scenario=scenario)

            # 13. STATUS (+ troubleshooting with --debug)
            if i % STATUS_EVERY == 0:
                elapsed  = clock() - start_time
                status   = "NS-only" if not task_active else "Task+NS"
                error_mm = np.linalg.norm(delta_x) * 1000
                h_min    = np.min(h_vals)

                dbg(f"[{elapsed:.1f}s] Step {i}/{len(timesteps)} | "
                      f"Err={error_mm:.2f}mm | σ_min={sigma_min:.4e} | "
                      f"w={manip_val:.4f} | h_min={h_min:.3f} | "
                      f"filter={filter_state(sel.method, cbf_active)} | {status}")
                info(f"[{t:.2f}s] {i}/{len(timesteps)} | err {error_mm:.2f} mm | h_min {h_min:.4f}")
                # early warning: twin and real robot diverging
                if np.linalg.norm(q_diff) > 0.15:          # threshold ~8.6° total
                    dbg(f"[TWIN] ⚠ divergence: |Δq|={np.linalg.norm(q_diff):.3f} rad ")
                if is_debug():
                    _debug_status_scenario(scenario, h_vals, filter_state(sel.method, cbf_active), q)

            # 14. TIMING
            sleep_t = dt - (clock() - loop_start)
            if sleep_t > 0:
                sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[CONTROL] Interrupted.")
    except Exception as e:
        print(f"\n[CONTROL ERROR] {e}")
        traceback.print_exc()

    robot.stop()
    total_time = clock() - start_time
    dbg(f"\n[DONE] {total_time:.2f}s | {len(L.t_log)} steps\n")

    # ==========================================
    # OUTPUTS
    # ==========================================
    A = to_arrays(L, trajectory_target, is_pnp, dt)
    save_all_csv(exp_folder, A, is_pnp, scenario)
    R = RunInfo(
        timestamp=timestamp, ip=sel.ip, traj_name=traj_name, ctrl_name=ctrl_name,
        dist_name=dist_name, scenario_name=scenario_name, method_name=method_name,
        ctrl=sel.ctrl, method=sel.method, dist=sel.dist, is_pnp=is_pnp, total_time=total_time,
        center_x=center_x, center_y=center_y, center_z=center_z,
        pnp_waypoints=pnp_waypoints, PNP_SEGMENT_TIMES=PNP_SEGMENT_TIMES,
        pnp_cycle=pnp.pnp_cycle if is_pnp else 0,
        kossmc_issf_version=version, git_commit=commit, scenario=scenario,
    )
    dt_plot = write_experiment_info(exp_folder, A, P, R)

    if plot_fn is not None:
        plot_fn(exp_folder, A, P, R, dt_plot)

    dbg(f"Saved: {exp_folder}")
    info(f"[DONE] {exp_folder}")

    robot.disconnect()
    env.disconnect()
    return exp_folder
