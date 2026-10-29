"""Prometheus gauges."""

import numpy as np
from prometheus_client import REGISTRY, Gauge, start_http_server

from kossmc_issf.console import debug


class Metrics:
    def __init__(self, registry=REGISTRY):
        G = lambda name, doc: Gauge(name, doc, registry=registry)
        # Safety
        self.g_h_min          = G('h_min',                 'Minimum barrier value h(q)')
        self.g_safety_status  = G('safety_status',          'Status: 1=SAFE, 0=VIOLATION')
        self.g_cbf_active     = G('cbf_active',             'Filter correcting: 1=yes, 0=no')
        self.g_cbf_correction = G('cbf_correction_norm',    '||dq_safe - dq_nom||')

        # Whole-body combine (scenario 5)
        self.g_h_cont_min      = G('h_cont_min',      'Min h containment whole-body')
        self.g_h_obs_min       = G('h_obs_min',       'Min h obstacle whole-body')
        self.g_dist_cont_m     = G('dist_cont_min_m', 'Distance of the closest sphere to a containment wall (m)')
        self.g_dist_obs_m      = G('dist_obs_min_m',  'Distance of the closest sphere to the obstacle (m)')
        self.g_cont_min_sphere = G('cont_min_sphere', 'Critical sphere ID, containment')
        self.g_obs_min_sphere  = G('obs_min_sphere',  'Critical sphere ID, obstacle')
        self.g_h_cont_ee    = G('h_cont_ee',     'h containment sphere EE (s12)')
        self.g_h_obs_ee     = G('h_obs_ee',      'h obstacle sphere EE (s12)')

        self.g_safety_cont     = G('safety_cont',     'Containment status 1=safe 0=violation')
        self.g_safety_obs      = G('safety_obs',      'Obstacle status 1=safe 0=violation')

        # Tracking
        self.g_error_mm       = G('tracking_error_mm', '||e|| norm (mm)')
        self.g_error_x        = G('error_x_mm',        'Error X (mm)')
        self.g_error_y        = G('error_y_mm',        'Error Y (mm)')
        self.g_error_z        = G('error_z_mm',        'Error Z (mm)')
        self.g_x_des          = G('x_desired_m',       'Target X (m)')
        self.g_y_des          = G('y_desired_m',       'Target Y (m)')
        self.g_z_des          = G('z_desired_m',       'Target Z (m)')
        self.g_x_act          = G('x_actual_m',        'Actual X (m)')
        self.g_y_act          = G('y_actual_m',        'Actual Y (m)')
        self.g_z_act          = G('z_actual_m',        'Actual Z (m)')

        # Controller
        self.g_sigma_norm     = G('sigma_norm',  '||sigma||')
        self.g_sigma_x        = G('sigma_x',    'Sigma X')
        self.g_sigma_y        = G('sigma_y',    'Sigma Y')
        self.g_sigma_z        = G('sigma_z',    'Sigma Z')
        self.g_dq_cmd_norm    = G('dq_cmd_norm','||dq_cmd|| (rad/s)')
        self.g_dq_act_norm    = G('dq_act_norm','||dq_act|| (rad/s)')

        # Joint
        self.g_q  = [G(f'q_joint_{i+1}_deg', f'Joint {i+1} pos (deg)') for i in range(6)]
        self.g_dq = [G(f'dq_joint_{i+1}',   f'Joint {i+1} vel (rad/s)') for i in range(6)]
        self.g_q_sim  = [G(f'q_sim_joint_{i+1}_deg', f'Joint {i+1} twin pos (deg)') for i in range(6)]
        self.g_dq_sim = [G(f'dq_sim_joint_{i+1}',    f'Joint {i+1} twin vel (rad/s)') for i in range(6)]
        self.g_q_diff = G('q_diff_norm_deg', 'Norm of the real-twin joint difference (deg)')

        # Singularity
        self.g_sigma_min_jac  = G('sigma_min_jacobian', 'Min singular value Jacobian')
        self.g_manipulability = G('manipulability',      'Manipulability w(q)')

        # Experiment
        self.g_pnp_phase      = G('pnp_phase', 'Active PnP phase (0-7)')

        self.g_obs_center_x   = G('obs_center_x', 'Obstacle center X (m)')
        self.g_obs_center_y   = G('obs_center_y', 'Obstacle center Y (m)')
        self.g_obs_center_z   = G('obs_center_z', 'Obstacle center Z (m)')
        self.g_obs_radius     = G('obs_radius',   'Obstacle radius (m)')

        self.g_pos_min_x = G('pos_min_x', 'CBF box min X (m)')
        self.g_pos_min_y = G('pos_min_y', 'CBF box min Y (m)')
        self.g_pos_min_z = G('pos_min_z', 'CBF box min Z (m)')
        self.g_pos_max_x = G('pos_max_x', 'CBF box max X (m)')
        self.g_pos_max_y = G('pos_max_y', 'CBF box max Y (m)')
        self.g_pos_max_z = G('pos_max_z', 'CBF box max Z (m)')

        self.g_h_cont_ee_xmin = G('h_cont_ee_xmin', 'h cont s12 x >= xmin')
        self.g_h_cont_ee_xmax = G('h_cont_ee_xmax', 'h cont s12 x <= xmax')
        self.g_h_cont_ee_ymin = G('h_cont_ee_ymin', 'h cont s12 y >= ymin')
        self.g_h_cont_ee_ymax = G('h_cont_ee_ymax', 'h cont s12 y <= ymax')
        self.g_h_cont_ee_zmin = G('h_cont_ee_zmin', 'h cont s12 z >= zmin')
        self.g_h_cont_ee_zmax = G('h_cont_ee_zmax', 'h cont s12 z <= zmax')

        self.g_dq_ns_norm     = G('dq_ns_norm', '||dq_ns|| null-space effort (manip + joint limit)')

    def start_server(self, port):
        start_http_server(port)
        debug(f"[PROMETHEUS] http://localhost:{port}/metrics")

    def set_static(self, OBS_CENTER, OBS_RADIUS, POS_MIN, POS_MAX):
        self.g_obs_center_x.set(float(OBS_CENTER[0]))
        self.g_obs_center_y.set(float(OBS_CENTER[1]))
        self.g_obs_center_z.set(float(OBS_CENTER[2]))
        self.g_obs_radius.set(float(OBS_RADIUS))

        self.g_pos_min_x.set(float(POS_MIN[0]))
        self.g_pos_min_y.set(float(POS_MIN[1]))
        self.g_pos_min_z.set(float(POS_MIN[2]))
        self.g_pos_max_x.set(float(POS_MAX[0]))
        self.g_pos_max_y.set(float(POS_MAX[1]))
        self.g_pos_max_z.set(float(POS_MAX[2]))

    def update(self, delta_x, target_pos, x_actual_rtde, dq_ns, h_vals, cbf_active, dq_cmd, dq_nom,
               dq_safe, OBS_RADIUS, sigma, qd_actual, q, q_twin, qd_twin, q_diff, sigma_min, manip_val, pnp_phase,
               scenario="wb_combine"):
        """One control cycle. dq_cmd includes the disturbance; cbf_correction_norm is the pure filter
        intervention ||dq_safe - dq_nom||. The whole-body breakdown gauges are
        set only for wb_combine."""
        # Tracking
        self.g_error_mm.set(float(np.linalg.norm(delta_x) * 1000))
        self.g_error_x.set(float(delta_x[0] * 1000))
        self.g_error_y.set(float(delta_x[1] * 1000))
        self.g_error_z.set(float(delta_x[2] * 1000))
        self.g_x_des.set(float(target_pos[0]))
        self.g_y_des.set(float(target_pos[1]))
        self.g_z_des.set(float(target_pos[2]))
        self.g_x_act.set(float(x_actual_rtde[0]))
        self.g_y_act.set(float(x_actual_rtde[1]))
        self.g_z_act.set(float(x_actual_rtde[2]))

        # Null space
        self.g_dq_ns_norm.set(float(np.linalg.norm(dq_ns)))

        # Safety
        _h_min = float(np.min(h_vals))
        self.g_h_min.set(_h_min)
        self.g_safety_status.set(1 if _h_min >= 0 else 0)
        self.g_cbf_active.set(int(cbf_active))
        self.g_cbf_correction.set(float(np.linalg.norm(dq_safe - dq_nom)))

        # Whole-body safety breakdown (wb_combine only)
        if scenario == "wb_combine":
            n_sph  = h_vals.shape[0] // 7
            h_cont = h_vals[:n_sph*6]
            h_obs  = h_vals[n_sph*6:]

            cont_min = float(h_cont.min())
            obs_min  = float(h_obs.min())

            self.g_h_cont_min.set(cont_min)
            self.g_h_obs_min.set(obs_min)

            self.g_dist_cont_m.set(cont_min)
            self.g_dist_obs_m.set(float(np.sqrt(max(obs_min + OBS_RADIUS**2, 0.0)) - OBS_RADIUS))

            cont_idx = int(np.argmin(h_cont))
            cont_sphere = (cont_idx % (3*n_sph)) // 3
            self.g_cont_min_sphere.set(cont_sphere)
            self.g_obs_min_sphere.set(int(np.argmin(h_obs)))

            EE_IDX = 12
            n_sph_ee = h_cont.shape[0] // 6
            h_lower_ee = h_cont[EE_IDX*3 : EE_IDX*3+3]
            h_upper_ee = h_cont[n_sph_ee*3 + EE_IDX*3 : n_sph_ee*3 + EE_IDX*3+3]

            self.g_h_cont_ee_xmin.set(float(h_lower_ee[0]))
            self.g_h_cont_ee_ymin.set(float(h_lower_ee[1]))
            self.g_h_cont_ee_zmin.set(float(h_lower_ee[2]))
            self.g_h_cont_ee_xmax.set(float(h_upper_ee[0]))
            self.g_h_cont_ee_ymax.set(float(h_upper_ee[1]))
            self.g_h_cont_ee_zmax.set(float(h_upper_ee[2]))
            self.g_h_cont_ee.set(float(min(h_lower_ee.min(), h_upper_ee.min())))
            self.g_h_obs_ee.set(float(h_obs[EE_IDX]))

            self.g_safety_cont.set(1 if cont_min >= 0 else 0)
            self.g_safety_obs.set(1 if obs_min  >= 0 else 0)

        # Controller
        self.g_sigma_norm.set(float(np.linalg.norm(sigma)))
        self.g_sigma_x.set(float(sigma[0]))
        self.g_sigma_y.set(float(sigma[1]))
        self.g_sigma_z.set(float(sigma[2]))
        self.g_dq_cmd_norm.set(float(np.linalg.norm(dq_cmd)))
        self.g_dq_act_norm.set(float(np.linalg.norm(qd_actual)))

        # Joint
        for _ji in range(6):
            self.g_q[_ji].set(float(np.rad2deg(q[_ji])))
            self.g_dq[_ji].set(float(qd_actual[_ji]))
            self.g_q_sim[_ji].set(float(np.rad2deg(q_twin[_ji])))
            self.g_dq_sim[_ji].set(float(qd_twin[_ji]))
        self.g_q_diff.set(float(np.rad2deg(np.linalg.norm(q_diff))))

        # Singularity
        self.g_sigma_min_jac.set(float(sigma_min))
        self.g_manipulability.set(float(manip_val))

        # Experiment
        self.g_pnp_phase.set(pnp_phase)
