"""Precomputed task-space trajectories."""

import numpy as np


def sinusoidal_3d(center, amplitude, frequency_traj, duration, dt):
    """Closed 3D curve around `center` (initial TCP), tilted by 30 deg (preset "circular").

    Returns (trajectory_target, trajectory_velocity), each shape (len(arange(0, duration, dt)), 3).
    """
    center_x, center_y, center_z = center
    t_arr = np.arange(0, duration, dt)
    omega = 2 * np.pi * frequency_traj
    theta  = np.radians(30)

    x_arr  = center_x - amplitude * np.cos(theta) * np.cos(omega * t_arr)
    y_arr  = center_y - amplitude                  * np.sin(omega * t_arr)
    z_arr  = center_z + amplitude * np.sin(theta) * np.cos(omega * t_arr)

    vx_arr =  amplitude * np.cos(theta) * omega * np.sin(omega * t_arr)
    vy_arr = -amplitude                  * omega * np.cos(omega * t_arr)
    vz_arr = -amplitude * np.sin(theta) * omega * np.sin(omega * t_arr)

    trajectory_target = np.column_stack([x_arr, y_arr, z_arr])
    trajectory_velocity = np.column_stack([vx_arr, vy_arr, vz_arr])
    return trajectory_target, trajectory_velocity
