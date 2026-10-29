"""PyBullet scene: plane, table and the UR5e digital twin."""

import pybullet as p
import pybullet_data

from kossmc_issf.config.defaults import TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH
from kossmc_issf.core.kinematics import URDF_PATH


class PyBulletEnv:
    """Connects to PyBullet (GUI by default; DIRECT for offline tests) and builds the scene."""

    def __init__(self, dt, gui=True):
        p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide all GUI panels
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[TABLE_LENGTH/2, TABLE_WIDTH/2, TABLE_HEIGHT/2]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[TABLE_LENGTH/2, TABLE_WIDTH/2, TABLE_HEIGHT/2],
            rgbaColor=[0.8, 0.6, 0.4, 1.0]  # wood colour
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.3, 0, TABLE_HEIGHT/2 ]
        )
        self.robot_sim = p.loadURDF(
            URDF_PATH,
            basePosition=[0, 0, TABLE_HEIGHT],
            useFixedBase=True
        )
        self.sim_joint_indices = [i for i in range(p.getNumJoints(self.robot_sim))
                                  if p.getJointInfo(self.robot_sim, i)[2] != p.JOINT_FIXED]

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=135,       # view from the front right
                cameraPitch=-25,    # slightly from above
                cameraTargetPosition=[0, 0, TABLE_HEIGHT + 0.3]
            )

        # twin physics: same time step as the 125 Hz loop, no damping / friction
        p.setTimeStep(dt)
        for idx in self.sim_joint_indices:
            p.changeDynamics(
                self.robot_sim, idx,
                linearDamping=0.0,
                angularDamping=0.0,
                jointDamping=0.0,
                lateralFriction=0.0,
                spinningFriction=0.0,
                rollingFriction=0.0
            )
        # replace PyBullet's default zero-velocity motors
        p.setJointMotorControlArray(
            self.robot_sim,
            self.sim_joint_indices,
            p.VELOCITY_CONTROL,
            forces=[150.0] * len(self.sim_joint_indices),
            )

    def reset_state(self, q, dq):
        for i, joint_idx in enumerate(self.sim_joint_indices):
            p.resetJointState(int(self.robot_sim), int(joint_idx),
                              float(q[i]), float(dq[i]))

    def disconnect(self):
        if p.isConnected():
            p.disconnect()
