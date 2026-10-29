"""UR5e kinematics with Pinocchio (EE frame `tool0`, X/Y flipped to the RTDE base frame)."""

import os
from functools import lru_cache

import numpy as np
import pinocchio as pin

from oscbf.core.manipulator import Manipulator
from oscbf.core.ur5e_collision_model import ur5e_collision_data

URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "ur_e_description", "urdf", "ur5e.urdf",
)


class RobotModel:
    """Pinocchio model/data shared by all kinematics calls, plus the oscbf Manipulator.

    `data` is mutated by every call; each call recomputes forward kinematics first.
    """

    def __init__(self, urdf_path=URDF_PATH):
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("tool0")
        self.manipulator = Manipulator.from_urdf(
            urdf_path,
            collision_data=ur5e_collision_data
        )

    def forward_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        pos = oMf.translation.copy()
        pos[0] *= -1
        pos[1] *= -1
        return pos, oMf.rotation.copy()

    def jacobian(self, q):
        """Linear part of the EE Jacobian, shape (3, 6)."""
        pin.forwardKinematics(self.model, self.data, q)
        J_world = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_world[:3, :].copy()
        J[0, :] *= -1.0
        J[1, :] *= -1.0
        return J

    def full_jacobian(self, q):
        """Full EE Jacobian, shape (6, 6)."""
        pin.forwardKinematics(self.model, self.data, q)
        J_world = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id,
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_full = J_world.copy()
        J_full[0, :] *= -1.0
        J_full[1, :] *= -1.0
        return J_full

    def manipulability_gradient(self, q, J, epsilon=1e-6):
        """Forward-difference gradient of the manipulability of the linear Jacobian."""
        w_current = compute_manipulability(J)
        grad_w = np.zeros(6)
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += epsilon
            J_plus = self.jacobian(q_plus)
            w_plus = compute_manipulability(J_plus)
            grad_w[i] = (w_plus - w_current) / epsilon
        return grad_w, w_current


@lru_cache(maxsize=None)
def get_robot_model():
    """The single RobotModel instance of the process."""
    return RobotModel()


def compute_manipulability(J):
    JJT = J @ J.T
    return np.sqrt(max(0, np.linalg.det(JJT)))


def rpy_from_R(R):
    sy = np.clip(-R[2, 0], -1.0, 1.0)
    return np.arctan2(R[2,1], R[2,2]), np.arcsin(sy), np.arctan2(R[1,0], R[0,0])


def singularity_analysis(J_full):
    """SVD of the full Jacobian.

    Returns (sigma_min, |det J_full|, rank of the linear part, rank of J_full), ranks with tol=1e-3.
    """
    _, S, _ = np.linalg.svd(J_full)
    sigma_min = S[-1]
    full_manip = abs(np.linalg.det(J_full))
    Jp = J_full[:3, :]
    rank_Jp = np.linalg.matrix_rank(Jp, tol=1e-3)
    rank_J = np.linalg.matrix_rank(J_full, tol=1e-3)
    return sigma_min, full_manip, rank_Jp, rank_J


def update_task_active(task_active, sigma_min, eps):
    """Singularity hysteresis on sigma_min.

    Returns (task_active, event), event is None, "deactivated" or "reactivated".
    """
    if task_active:
        if sigma_min <= eps:
            return False, "deactivated"
    else:
        if sigma_min > eps:
            return True, "reactivated"
    return task_active, None
