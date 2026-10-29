"""Pick-and-place with linearly interpolated segments ("pnp_interpolated")."""

import numpy as np

PNP_NAMES = ["HOME", "APPROACH PICK", "PICK", "LIFT", "APPROACH PLACE 1", "APPROACH PLACE 2", "PLACE", "RETRACT"]


def generate_interpolated(pnp_waypoints, pnp_speed, dt):
    """One segment per waypoint i -> i+1 (the last one wraps back to HOME), at constant speed.

    Returns (pnp_seg_pos, pnp_seg_vel, segment_times, segment_lengths); pnp_seg_pos[i] and
    pnp_seg_vel[i] have shape (N_seg, 3) with N_seg = max(1, int(T_seg / dt)).
    """
    n_wp = len(pnp_waypoints)
    pnp_seg_pos       = []
    pnp_seg_vel       = []
    segment_times     = []
    segment_lengths   = []

    for i in range(n_wp):
        p_start = pnp_waypoints[i]
        p_end   = pnp_waypoints[(i + 1) % n_wp]

        dist  = np.linalg.norm(p_end - p_start)
        T_seg = dist / pnp_speed
        N_seg = max(1, int(T_seg / dt))

        # alpha_k = k / N_seg
        seg_pos = p_start + (p_end - p_start) * (np.arange(N_seg)[:, None] / N_seg)
        # constant velocity, norm = pnp_speed
        seg_vel = np.tile((p_end - p_start) / T_seg, (N_seg, 1))

        segment_times.append(T_seg)
        segment_lengths.append(dist)
        pnp_seg_pos.append(seg_pos)
        pnp_seg_vel.append(seg_vel)

    return pnp_seg_pos, pnp_seg_vel, segment_times, segment_lengths


class PnPInterpolated:
    """Phase machine stepping through the interpolated segments.

    State: pnp_phase (active segment), pnp_seg_idx (index inside the segment), pnp_cycle (completed cycles).
    """

    def __init__(self, pnp_waypoints, pnp_seg_pos, pnp_seg_vel, pnp_thresh):
        self.pnp_waypoints = pnp_waypoints
        self.pnp_seg_pos = pnp_seg_pos
        self.pnp_seg_vel = pnp_seg_vel
        self.pnp_thresh = pnp_thresh
        self.n_wp = len(pnp_waypoints)
        self.pnp_phase = 0
        self.pnp_seg_idx = 0
        self.pnp_cycle = 0

    def step(self, x_actual_rtde):
        """One control cycle.

        Returns (target_pos, v_des, phase_for_log, transition). phase_for_log is the phase before any
        transition. transition is None or a dict with keys from_phase, to_phase, err_mm, reason
        ("reached" / "seg_complete") and cycle_completed (bool).
        """
        phase_for_log = self.pnp_phase
        seg_len    = len(self.pnp_seg_pos[self.pnp_phase])
        idx_clamp  = min(self.pnp_seg_idx, seg_len - 1)
        target_pos = self.pnp_seg_pos[self.pnp_phase][idx_clamp]
        v_des      = self.pnp_seg_vel[self.pnp_phase][idx_clamp]
        self.pnp_seg_idx += 1

        wp_next   = self.pnp_waypoints[(self.pnp_phase + 1) % self.n_wp]
        err_to_wp = np.linalg.norm(x_actual_rtde - wp_next) * 1000  # mm

        transition = None
        if (self.pnp_seg_idx >= seg_len) or (err_to_wp < self.pnp_thresh):
            reason = "reached" if err_to_wp < self.pnp_thresh else "seg_complete"
            next_phase = (self.pnp_phase + 1) % self.n_wp
            transition = {
                "from_phase": self.pnp_phase,
                "to_phase": next_phase,
                "err_mm": err_to_wp,
                "reason": reason,
                "cycle_completed": False,
            }
            self.pnp_phase   = next_phase
            self.pnp_seg_idx = 0
            if self.pnp_phase == 0:
                self.pnp_cycle += 1
                transition["cycle_completed"] = True

        return target_pos, v_des, phase_for_log, transition
