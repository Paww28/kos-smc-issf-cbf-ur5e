"""Digital twin step: the twin replays the robot's ACTUAL joint velocity (not the command)."""

import numpy as np
import pybullet as p


def twin_step(env, q_actual_rtde, qd_actual):
    """Reset the twin to the measured state, drive it with qd_actual for one physics step."""
    for j, idx in enumerate(env.sim_joint_indices):
        p.resetJointState(env.robot_sim, idx, q_actual_rtde[j], qd_actual[j])

    p.setJointMotorControlArray(
        env.robot_sim,
        env.sim_joint_indices,
        p.VELOCITY_CONTROL,
        targetVelocities=qd_actual.tolist(),
    )

    p.stepSimulation()                          # twin: one physics step of dt


def twin_state(env):
    joint_states = p.getJointStates(env.robot_sim, env.sim_joint_indices)
    q_twin  = np.array([js[0] for js in joint_states])
    qd_twin = np.array([js[1] for js in joint_states])
    return q_twin, qd_twin
