"""Safety filters (CBFpy): CBF or ISSf-CBF over one of four scenarios.

- ee_containment: end-effector inside a box; plain cbfpy QP (identity objective, no joint
  velocity limits), FK from DH parameters.
- wb_obstacle, wb_containment, wb_combine: collision spheres of
  the whole arm; OSCBF velocity configuration (operational-space objective + URDF joint velocity limits).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from cbfpy import CBF, CBFConfig

from oscbf.core.oscbf_configs import OSCBFVelocityConfig


SCENARIOS = ("ee_containment", "wb_obstacle", "wb_containment", "wb_combine")

# ---- UR5e forward kinematics in JAX (DH parameters, UR e-Series) ----
_UR5E_A     = jnp.array([0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0])
_UR5E_D     = jnp.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
_UR5E_ALPHA = jnp.array([jnp.pi/2, 0.0, 0.0, jnp.pi/2, -jnp.pi/2, 0.0])

# DH base frame (UR "base", RTDE) -> base_link (URDF/Pinocchio, frame of the safety box)
_DH_TO_BASE_LINK = jnp.array([-1.0, -1.0, 1.0])


def _dh_matrix(a, d, alpha, theta):
    """One 4x4 DH transform (JAX-compatible)."""
    ct, st = jnp.cos(theta), jnp.sin(theta)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    return jnp.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0.0,     sa,      ca,     d],
        [0.0,    0.0,     0.0,   1.0],
    ])


def ee_position_jax(q):
    """EE position from the DH chain, in the DH base frame (UR base / RTDE frame)."""
    T = jnp.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(_UR5E_A[i], _UR5E_D[i], _UR5E_ALPHA[i], q[i])
    pos = T[:3, 3]

    return pos


@jax.tree_util.register_static
class EEContainmentConfig(CBFConfig):
    """End-effector containment box (6 barriers: pos_max - ee, ee - pos_min).

    The DH position is in the UR base (RTDE) frame; it is converted to base_link (x, y negated)
    before it is compared with the box, which is given in base_link.
    """

    def __init__(self, pos_min, pos_max, alpha_gain=10.0):
        self.pos_min    = jnp.array(pos_min)
        self.pos_max    = jnp.array(pos_max)
        self.alpha_gain = float(alpha_gain)

        super().__init__(n=6, m=6)

    def f(self, z):
        return jnp.zeros_like(z)

    def g(self, z):
        return jnp.eye(6)

    def h_1(self, z, **kwargs):
        ee_pos = ee_position_jax(z) * _DH_TO_BASE_LINK
        return jnp.concatenate([
            self.pos_max - ee_pos,
            ee_pos - self.pos_min,
        ])

    def alpha(self, h):
        return self.alpha_gain * h


@jax.tree_util.register_static
class WholeBodyObstacleAvoConfig(OSCBFVelocityConfig):
    """Whole-body obstacle avoidance: one barrier per collision sphere (vertical cylinder)."""

    def __init__(self, robot, obs_center, obs_radius, alpha_gain=10.0):
        self.robot      = robot
        self.obs_center = jnp.array(obs_center)
        self.obs_radius = float(obs_radius)
        self.alpha_gain = float(alpha_gain)

        super().__init__(robot)

    def h_1(self, z, **kwargs):
        collision_data = self.robot.link_collision_data(z)
        pos_pino = collision_data[:, :3]
        radii    = collision_data[:, 3]

        pos = pos_pino

        diff_xy = pos[:, :2] - self.obs_center[:2]
        dist_sq = jnp.sum(diff_xy**2, axis=1)
        safe_sq = (radii + self.obs_radius) ** 2

        return dist_sq - safe_sq

    def alpha(self, h):
        return self.alpha_gain * h


@jax.tree_util.register_static
class WholeBodyContainmentConfig(OSCBFVelocityConfig):
    """Whole-body containment: 6 barriers per collision sphere (3 lower, then 3 upper per sphere)."""

    def __init__(self, robot, pos_min, pos_max, alpha_gain=10.0):
        self.robot   = robot
        self.pos_min = jnp.array(pos_min)
        self.pos_max = jnp.array(pos_max)
        self.alpha_gain = float(alpha_gain)

        super().__init__(robot)

    def h_1(self, z, **kwargs):
        collision_data = self.robot.link_collision_data(z)
        pos_pino = collision_data[:, :3]
        radii    = collision_data[:, 3]

        pos = pos_pino

        # h_lower: pos - pos_min - r, h_upper: pos_max - pos - r
        h_lower = (pos - self.pos_min[None, :] - radii[:, None])
        h_upper = (self.pos_max[None, :] - pos - radii[:, None])

        return jnp.concatenate([h_lower.flatten(), h_upper.flatten()])

    def alpha(self, h):
        return self.alpha_gain * h


@jax.tree_util.register_static
class WholeBodyCombineConfig(OSCBFVelocityConfig):
    """
    Gabungan Whole-Body Containment (Eq.48) + Whole-Body Obstacle Avoidance (Eq.47)
    Total constraints = n_spheres*6 (containment) + n_spheres (obstacle)
    """

    def __init__(self, robot, pos_min, pos_max,
                 obs_center, obs_radius, alpha_gain=10.0):
        self.robot      = robot
        self.pos_min    = jnp.array(pos_min)
        self.pos_max    = jnp.array(pos_max)
        self.obs_center = jnp.array(obs_center)
        self.obs_radius = float(obs_radius)
        self.alpha_gain = float(alpha_gain)

        super().__init__(robot)

    def h_1(self, z, **kwargs):
        collision_data = self.robot.link_collision_data(z)
        pos_pino = collision_data[:, :3]
        radii    = collision_data[:, 3]

        # no X/Y flip: obs_center is also given in the RTDE frame
        pos = pos_pino

        # --- Whole-Body Containment (Eq.48) ---
        h_lower = pos - self.pos_min[None, :] - radii[:, None]
        h_upper = self.pos_max[None, :] - pos - radii[:, None]

        # --- Whole-Body Obstacle Avoidance (Eq.47) ---
        diff_xy  = pos[:, :2] - self.obs_center[:2]
        dist_sq  = jnp.sum(diff_xy**2, axis=1)
        safe_sq  = (radii + self.obs_radius) ** 2
        h_obs    = dist_sq - safe_sq

        return jnp.concatenate([
            h_lower.flatten(),   # n_spheres × 3
            h_upper.flatten(),   # n_spheres × 3
            h_obs                # n_spheres × 1
        ])

    def alpha(self, h):
        return self.alpha_gain * h


@jax.tree_util.register_static
class ISSfCBF(CBF):
    """Input-to-State Safety filtered CBF (ISSf-CBF)

    QP constraint modified from the standard CBF:
      CBF     : Lf_h + Lg_h·u          >= -alpha(h)
      ISSf-CBF: Lf_h + Lg_h·u - ε·||Lg_h||² >= -alpha(h)

    The extra term -ε·Lg_h·(Lg_h)ᵀ adds robustness to disturbance/uncertainty
    without increasing alpha_gain.
    """

    def __init__(self, epsilon: float, *args, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config: CBFConfig, epsilon: float = 0.1) -> "ISSfCBF":
        """Construct an ISSf-CBF from the same config as a standard CBF.

        Args:
            config   : CBFConfig object (exactly as for CBF.from_config)
            epsilon  : ISSf tuning parameter (ε > 0). Larger = more conservative/robust.

        Returns:
            ISSfCBF instance
        """
        instance = cls(
            epsilon,           # ISSf-CBF extra argument
            config.n,
            config.m,
            config.num_cbf,
            config.u_min,
            config.u_max,
            config.control_constrained,
            config.relax_cbf,
            config.cbf_relaxation_penalty,
            config.h_1,
            config.h_2,
            config.f,
            config.g,
            config.alpha,
            config.alpha_2,
            config.P,
            config.q,
            config.solver_tol,
        )
        instance._validate_instance(*config.init_args)
        return instance

    def h_qp(self, z: Array, u_des: Array, *h_args) -> Array:
        """Override h_qp: add the ISSf term -ε·||Lg_h||²

        Constraint QP: G·u <= h_qp
        where G = -Lg_h, so the original constraint is:
            -Lg_h·u <= alpha(h) + Lf_h - ε·||Lg_h||²
        equivalent to:
            Lf_h + Lg_h·u - ε·||Lg_h||² >= -alpha(h)  ← ISSf-CBF

        Args:
            z      : State, shape (n,)
            u_des  : Desired control, shape (m,)
            *h_args: Optional barrier args

        Returns:
            Array: h vector shape (num_constraints,)
        """
        hz, lfh = self.h_and_Lfh(z, *h_args)

        # Lgh shape: (num_cbf, m)
        lgh = self.Lgh(z, *h_args)

        # ε·||Lg_h(i)||² per barrier → shape (num_cbf,)
        # ||Lg_h(i)||² = sum of squares of row i
        issf_term = self.epsilon * jnp.sum(lgh ** 2, axis=1)

        # ISSf-CBF: alpha(h) + Lf_h - ε·||Lg_h||²
        h = self.alpha(hz) + lfh - issf_term

        if self.control_constrained:
            return jnp.concatenate(
                [h, jnp.asarray(self.u_max), -jnp.asarray(self.u_min)]
            )
        else:
            return h


def build_config(scenario, robot_manipulator, P):
    """Barrier configuration of a scenario (P: config.presets.Params)."""
    if scenario == "ee_containment":
        return EEContainmentConfig(P.POS_MIN, P.POS_MAX, alpha_gain=P.CBF_ALPHA)
    elif scenario == "wb_obstacle":
        return WholeBodyObstacleAvoConfig(robot_manipulator, P.OBS_CENTER, P.OBS_RADIUS, alpha_gain=P.CBF_ALPHA)
    elif scenario == "wb_containment":
        return WholeBodyContainmentConfig(robot_manipulator, P.POS_MIN, P.POS_MAX, alpha_gain=P.CBF_ALPHA)
    elif scenario == "wb_combine":
        return WholeBodyCombineConfig(robot_manipulator, P.POS_MIN, P.POS_MAX,
                                      P.OBS_CENTER, P.OBS_RADIUS, alpha_gain=P.CBF_ALPHA)
    raise ValueError(f"unknown scenario: {scenario!r}")


def compute_h_vals_numpy(x_ee, pos_min, pos_max):
    """EE box barriers for logging, order: xmax, xmin, ymax, ymin, zmax, zmin (x_ee in base_link)."""
    return np.array([
        pos_max[0] - x_ee[0],
        x_ee[0]    - pos_min[0],

        pos_max[1] - x_ee[1],
        x_ee[1]    - pos_min[1],

        pos_max[2] - x_ee[2],
        x_ee[2]    - pos_min[2],
    ])


def build_safety_filter(method, cfg, issf_epsilon):
    """method: "cbf", "issf" or "none". Returns the filter object, or None for "none"."""
    if method == "cbf":
        return CBF.from_config(cfg)
    elif method == "issf":
        return ISSfCBF.from_config(cfg, epsilon=issf_epsilon)
    elif method == "none":
        return None
    raise ValueError(f"unknown safety method: {method!r}")


def warmup(cbf_obj):
    """JIT warm-up at q = 0, u = 0 (only for an active filter)."""
    _q0 = jnp.zeros(6)
    _u0 = jnp.zeros(6)
    if cbf_obj is not None:
        cbf_obj.safety_filter(_q0, _u0)


def safety_step(cfg, cbf_obj, q, dq_cmd, h_vals=None):
    """Barrier values for logging, then the safety filter (if any).

    h_vals: precomputed logging values (ee_containment logs the measured TCP); None = cfg.h_1(q).
    Returns (dq_cmd, cbf_active, h_vals). With cbf_obj None, dq_cmd is returned unchanged.
    """
    q_jax      = jnp.array(q)
    dq_nom_jax = jnp.array(dq_cmd)

    if h_vals is None:
        h_vals = np.array(cfg.h_1(q_jax))

    cbf_active = False
    if cbf_obj is not None:
        dq_safe     = np.array(cbf_obj.safety_filter(q_jax, dq_nom_jax))
        cbf_active  = bool(np.linalg.norm(dq_safe - dq_cmd) > 1e-4)
        dq_cmd      = dq_safe

    return dq_cmd, cbf_active, h_vals
