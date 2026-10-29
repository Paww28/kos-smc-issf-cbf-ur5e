"""Task-space control laws: P, SMC with sign, SMC with quasi-sliding (boundary layer)."""

import numpy as np

CONTROLLERS = ("p", "smc_sign", "smc_quasi")


def task_space_control(ctrl, J, J_pinv, delta_x, qd_actual, v_des,
                       Kp, Lambda, k_smc, eps_qsmc):
    """Nominal joint velocity from the task-space error.

    delta_x = x_actual - x_target (RTDE frame). Returns (dq_task, sigma, e_dot);
    for "p", sigma and e_dot are zeros(3).
    """
    if ctrl == "p":
        nu      = -Kp * delta_x
        dq_task = J_pinv @ nu
        sigma   = np.zeros(3)
        e_dot   = np.zeros(3)
    elif ctrl in ("smc_sign", "smc_quasi"):
        x_dot_actual = J @ qd_actual
        e_dot  = x_dot_actual - v_des
        sigma  = e_dot + Lambda @ delta_x

        sigma_norm = np.linalg.norm(sigma)

        if ctrl == "smc_sign":
            sigma_smooth = np.sign(sigma)
        else:
            sigma_smooth = sigma / (sigma_norm + eps_qsmc)
        dq_task = J_pinv @ (v_des - k_smc * sigma_smooth)
    else:
        raise ValueError(f"unknown controller: {ctrl!r}")
    return dq_task, sigma, e_dot


def lyapunov(sigma):
    """V = 0.5 * sigma^T sigma."""
    return 0.5 * float(np.dot(sigma, sigma))
