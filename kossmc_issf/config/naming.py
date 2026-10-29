"""Option keys -> the name strings used in folder names, logs and experiment_info.txt."""

TRAJ_NAMES     = {"sinusoidal_3d": "sinusoidal_3d", "pnp_interpolated": "pnp_interpolated"}
CTRL_NAMES     = {"p": "P", "smc_sign": "SMC_sign", "smc_quasi": "SMC_quasi"}
DIST_NAMES     = {"gaussian": "dist_gaussian", "sinusoidal": "dist_sinusoidal", "none": "no_dist"}
METHOD_NAMES   = {"cbf": "CBF", "issf": "ISSf-CBF", "none": "No-CBF"}
SCENARIO_NAMES = {"ee_containment": "ee_containment", "wb_obstacle": "wb_obs_avo",
                  "wb_containment": "whole_body_containment", "wb_combine": "whole_body_combine"}
SCENARIO_NAME  = SCENARIO_NAMES["wb_combine"]

# menu numbers -> option keys
MENU_TRAJ     = {"1": "sinusoidal_3d", "2": "pnp_interpolated"}
MENU_CTRL     = {"1": "p", "2": "smc_sign", "3": "smc_quasi"}
MENU_SCENARIO = {"1": "ee_containment", "2": "wb_obstacle", "3": "wb_containment", "4": "wb_combine"}
MENU_METHOD   = {"1": "cbf", "2": "issf", "3": "none"}
MENU_DIST     = {"1": "gaussian", "2": "sinusoidal", "3": "none"}

def get_robot_source(ip):
    if ip == '127.0.0.2': return "URSIM"
    elif ip == '192.168.0.191': return "ROBOT"
    else: return f"Custom_{ip.replace('.','_')}"
