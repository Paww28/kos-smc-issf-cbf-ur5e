"""Null-space secondary task: manipulability gradient + joint-limit term."""

import numpy as np


def joint_limit_gradient(q, q_min, q_max, q_mid):
    n_joints = 6
    return -(1.0 / n_joints) * (q - q_mid) / (q_max - q_min)**2


def nullspace_velocity(grad_manip, grad_jl, K_null, K_jl, J, J_pinv, task_active):
    """Projects K_null*grad_manip + K_jl*grad_jl onto the null space of J.

    When the task is deactivated (singular), the projector is the identity.
    """
    dq_null = K_null * grad_manip + K_jl * grad_jl
    N      = np.eye(6) - J_pinv @ J if task_active else np.eye(6)
    dq_ns  = N @ dq_null
    return dq_ns
