"""PyBullet visual markers (no physics): safety box, obstacle, trajectory, waypoints, collision spheres."""

import numpy as np
import jax.numpy as jnp
import pybullet as p

from kossmc_issf.config.defaults import BASE_Z_OFFSET


def visualize_3D_box(pos_min, pos_max, offset=np.zeros(3), rgba=(0, 0.8, 0.2, 0.5)):
    pos_min = np.array(pos_min)+ np.array(offset)
    pos_max = np.array(pos_max) + np.array(offset)

    half_extents = (pos_max - pos_min) / 2.0
    center = (pos_max + pos_min) / 2.0

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgba
    )

    box_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=center.tolist()
    )

    return box_id


def visualize_obstacle_cylinder(center, radius, height=1, rgba=(1, 0.3, 0.3, 0.5)):
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,          # cylinder height
        rgbaColor=rgba
    )

    cylinder_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=center.tolist()
    )

    return cylinder_id


def visualize_safety_region(scenario, POS_MIN, POS_MAX, OBS_CENTER, OBS_RADIUS):
    """Box and/or obstacle of the scenario (PyBullet frame: X/Y flipped, Z offset by the table)."""
    if scenario in ("ee_containment", "wb_containment", "wb_combine"):
        pybullet_min = np.array([POS_MAX[0], POS_MAX[1], POS_MIN[2] + BASE_Z_OFFSET])
        pybullet_max = np.array([POS_MIN[0], POS_MIN[1], POS_MAX[2] + BASE_Z_OFFSET])
        visualize_3D_box(pybullet_min, pybullet_max)

    if scenario in ("wb_obstacle", "wb_combine"):
        pybullet_center = np.array([OBS_CENTER[0], OBS_CENTER[1], OBS_CENTER[2] + BASE_Z_OFFSET])
        visualize_obstacle_cylinder(pybullet_center, OBS_RADIUS, height=1)


def draw_trajectory_pybullet(trajectory):
    for i in range(len(trajectory)-1):
        p.addUserDebugLine(
            trajectory[i],
            trajectory[i+1],
            lineColorRGB=[0, 0, 1],
            lineWidth=2,
            lifeTime=0
        )


def draw_waypoint_markers(waypoints, names):
    """Waypoints as small spheres + label + arrow to the next waypoint."""
    DOT_COLOR  = [0.0, 0.0, 0.0]   # black
    LINE_COLOR = [1.0, 0.0, 0.0]   # red
    TEXT_COLOR = [1.0, 1.0, 1.0]   # white

    n = len(waypoints)
    for i, (wp, name) in enumerate(zip(waypoints, names)):
        # convert to the PyBullet frame
        pos_pb = [-wp[0], -wp[1], wp[2] + BASE_Z_OFFSET]

        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=DOT_COLOR + [1.0]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=pos_pb
        )

        p.addUserDebugText(
            text=f"{i}:{name}",
            textPosition=[pos_pb[0], pos_pb[1], pos_pb[2] + 0.05],
            textColorRGB=TEXT_COLOR,
            textSize=1.2,
            lifeTime=0
        )

        next_wp  = waypoints[(i + 1) % n]
        next_pb  = [-next_wp[0], -next_wp[1], next_wp[2] + BASE_Z_OFFSET]

        p.addUserDebugLine(
            lineFromXYZ=pos_pb,
            lineToXYZ=next_pb,
            lineColorRGB=LINE_COLOR,
            lineWidth=2,
            lifeTime=0
        )

        # arrow head
        direction = np.array(next_pb) - np.array(pos_pb)
        dist      = np.linalg.norm(direction)
        if dist > 0.01:
            unit    = direction / dist
            arrow_start = np.array(next_pb) - unit * 0.05
            perp = np.array([-unit[1], unit[0], 0]) * 0.03
            p.addUserDebugLine(
                arrow_start.tolist(),
                (np.array(next_pb) + perp).tolist(),
                lineColorRGB=LINE_COLOR, lineWidth=2, lifeTime=0
            )
            p.addUserDebugLine(
                (np.array(next_pb) + perp).tolist(),
                (np.array(next_pb) - perp).tolist(),
                lineColorRGB=LINE_COLOR, lineWidth=2, lifeTime=0
            )


def init_collision_spheres(max_spheres=30):
    return [None] * max_spheres


def update_collision_spheres(sphere_body_ids, robot_manipulator, q):
    """Create (first call) or move the oscbf collision spheres."""
    sphere_data = np.array(robot_manipulator.link_collision_data(jnp.array(q)))
    num_spheres = sphere_data.shape[0]

    for i in range(num_spheres):
        pos = sphere_data[i, :3]
        radius = sphere_data[i, 3]

        # PyBullet frame: no X/Y flip, Z offset by the table
        pos_pb = pos + np.array([0, 0, BASE_Z_OFFSET])

        if sphere_body_ids[i] is None:
            visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=float(radius),
                rgbaColor=[0, 0.8, 0, 0.5]
            )

            body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual,
                basePosition=pos_pb.tolist()
            )

            sphere_body_ids[i] = body

        else:
            p.resetBasePositionAndOrientation(
                sphere_body_ids[i],
                pos_pb.tolist(),
                [0, 0, 0, 1]
            )
