# kos-smc-issf-cbf-ur5e

Task-space sliding mode control (KOS-SMC) with a CBF / ISSf-CBF safety filter for the Universal
Robots UR5e, controlled over RTDE at 125 Hz. A PyBullet digital twin follows the robot, Prometheus
gauges are served during the run, and each run writes CSV logs, plots and a summary.

## Built on OSCBF

The safety filter builds on [OSCBF](https://github.com/StanfordASL/oscbf) (Operational Space Control
Barrier Functions) by Daniel Morton and Marco Pavone (Stanford ASL, MIT License), upstream commit
`a5251f7`. Changes in `oscbf/`:

- `core/ur5e_collision_model.py` (UR5e collision spheres) added, developed by the author together with a collaborator;
- `core/manipulator.py`, `core/oscbf_configs.py` and `utils/urdf_parser.py` adapted for the UR5e with a collaborator;

The UR5e URDF and meshes in `kossmc_issf/assets/ur_e_description/` come from the `ur_e_description`
package of ROS-Industrial `universal_robot` (meshes by Universal Robots).

## Architecture

| Module (`kossmc_issf/`) | Content |
|---|---|
| `config/` | presets (YAML), option names, CLI `kossmc-issf-run` and its menu |
| `core/` | kinematics (Pinocchio), controllers, null space, safety filters, disturbance |
| `trajectory/` | circular trajectory; pick-and-place segments and phase machine |
| `robot/` | RTDE interface (ur_rtde) |
| `sim/` | PyBullet digital twin and visuals |
| `monitoring/` | Prometheus gauges |
| `datalog/` | logs, CSV files, `experiment_info.txt` |
| `plotting/` | result plots (PNG) |
| `experiment.py` | one run: moveJ, pre-check, control loop, saving |

```
RTDE state -> reference -> task control (P / KOS-SMC) + null space = dq_nom
           -> CBF / ISSf-CBF -> dq_safe -> + disturbance = dq_cmd -> speedJ
```

## Method in brief

Each control cycle builds the joint-velocity command in three layers.

**1. Task-space controller** — tracks the end-effector reference.
- Tracking error: e = x − x_d
- Sliding surface: σ = ė + Λe
- KOS-SMC: q̇_task = J⁺ (ẋ_d − k_smc · s(σ))
  - `smc_sign`: s(σ) = sign(σ)
  - `smc_quasi`: s(σ) = σ / (‖σ‖ + ε_q)
- P (baseline): q̇_task = J⁺ (−K_p e)

**2. Null-space term** — the task uses only the 3D position, so the 6-joint arm has 3 spare degrees of freedom; they are used without moving the end effector.
- Projector: N = I − J⁺J
- Null-space command: q̇_ns = N (K_null ∇w + K_jl ∇ℓ)
  - w = √det(J Jᵀ): manipulability; increasing w moves the arm away from singularities
  - ℓ: joint-limit term; ∇ℓ = −(1/6)(q − q_mid)/(q_max − q_min)² pulls each joint toward the middle of its range q_mid
- Near a singularity the task is switched off and only the null-space term acts.

**3. Safety filter** — changes the nominal command q̇_nom = q̇_task + q̇_ns as little as possible so that every barrier h stays ≥ 0 (one QP per cycle).
- Barriers: box containment and/or cylindrical obstacle, on the end effector or the arm's collision spheres
- CBF: L_g h · u ≥ −α h
- ISSf-CBF: L_g h · u − ε ‖L_g h‖² ≥ −α h (extra margin against disturbances)

## Requirements

| Python | numpy | jax / jaxlib | cbfpy | pin | pybullet | ur_rtde |
|---|---|---|---|---|---|---|
| 3.10.8 | 1.26.4 | 0.4.30 / 0.4.30 | 0.0.1 | 3.8.0 | 3.2.7 | 1.6.2 |

Install with `--no-deps`: the metadata of `cmeel-boost` (pulled in by `pin`) requires numpy ≥ 2.2,
cbfpy requires numpy < 2; `pip check` reports this conflict, which is expected.

## Installation

```bash
git clone https://github.com/Paww28/kos-smc-issf-cbf-ur5e.git && cd kos-smc-issf-cbf-ur5e
python3.10 -m venv .venv && source .venv/bin/activate
pip install --no-deps -r requirements.txt
pip install --no-deps -e .
```

Python 3.10.8 via [pyenv](https://github.com/pyenv/pyenv); Docker via [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) or [Docker Desktop with WSL 2](https://docs.docker.com/desktop/features/wsl/).

## Running

Start URSim, open <http://localhost:6080/vnc.html>, power on the robot and release the brakes:

```bash
docker run --rm -it -p 5900:5900 -p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004 universalrobots/ursim_e-series:5.24.0
```

Run with the menu (trajectory, controller, scenario, method, disturbance):

```bash
PYTHONPATH= kossmc-issf-run
```

Or directly (missing options are asked in the menu):

```bash
PYTHONPATH= kossmc-issf-run --preset pnp --ctrl smc_quasi --method issf --dist gaussian
PYTHONPATH= kossmc-issf-run --preset circular --ctrl smc_sign --method cbf --dist none --scenario wb_obstacle
```

`PYTHONPATH=` clears a ROS `PYTHONPATH`. 127.0.0.2 is the default URSim address used in this
project; change with `--ip` if your setup differs. Tested with URSim 5.24.0 (official image).

**Physical UR5e:** use `--ip <robot-ip>`, set the safety settings on the teach pendant (e.g. speed
limits, safety planes) as an independent safety layer, and keep the emergency stop within reach.
The safety filter can still allow small barrier violations, so it does not replace the robot's own
safety limits.

## Arguments

| Argument | Choices | If omitted |
|---|---|---|
| `--preset` | `pnp`, `circular` | asked in the menu |
| `--ctrl` | `p`, `smc_sign`, `smc_quasi` | asked in the menu |
| `--method` | `cbf`, `issf`, `none` | asked in the menu |
| `--dist` | `gaussian`, `sinusoidal`, `none` | asked in the menu |
| `--scenario` | `ee_containment`, `wb_obstacle`, `wb_containment`, `wb_combine` | `wb_combine` if the four above are given, otherwise asked in the menu |
| `--ip` | robot address | `127.0.0.2` (URSim) |
| `--debug` | print troubleshooting output | off |

## Outputs

Each run writes to `experiments/<date>/<traj>_manip_<ctrl>_<dist>_<scenario>_<method>_<source>_<timestamp>/`:
CSV logs (tracking, joint velocities `dq_nom` / `dq_safe` / `dq_cmd`, barrier values `h_vals`,
`cbf_active`, digital twin), PNG plots, and `experiment_info.txt` (setup, parameters, summary).
`cbf_active` / "Filter correcting" means the filter changed the command, not that a barrier was violated.

## Testing

`PYTHONPATH= pytest -q` runs the offline tests (no robot, no network). Snapshot tests recompute the
outputs for stored inputs (`tests/fixtures/`) and fail if the numerical behavior changes (tolerance 1e-12 NumPy, 1e-10 JAX).



## License and Citation

This project is distributed under the MIT License (see [LICENSE](LICENSE)); the vendored `oscbf/` folder keeps its own MIT license. Since the safety filter builds on OSCBF, any work that uses this code should credit it by citing:

```
@inproceedings{morton2025oscbf,
  author={Morton, Daniel and Pavone, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions}, 
  year={2025},
  pages={187-194},
  doi={10.1109/IROS60139.2025.11246389}
}
```

