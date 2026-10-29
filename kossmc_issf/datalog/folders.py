"""Output folder: experiments/<date>/<traj>_manip_<ctrl>_<dist>_<scenario>_<method>_<source>_<timestamp>."""

import os
from datetime import datetime


def create_experiment_folder(traj_name, ctrl_name, dist_name, robot_source, scenario_name, method_name,
                             root="experiments", now=None):
    if now is None:
        now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(
        root, now.strftime("%Y-%m-%d"),
        f"{traj_name}_manip_{ctrl_name}_{dist_name}_{scenario_name}_{method_name}_{robot_source}_{timestamp}"
    )
    os.makedirs(full_path, exist_ok=True)
    return full_path, timestamp
