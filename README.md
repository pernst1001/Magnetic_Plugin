# Magnetic Plugin for Isaac Lab

This repository provides tools and environments for simulating magnetic fields in NVIDIA Isaac Lab, including a custom plugin, runnable scripts, and a reinforcement learning (RL) catheter environment.

## Repository Contents

1. **Magnetic Plugin** (`magnetic/`) – Custom extension for simulating magnetic fields, forces, and torques.
2. **Utility Scripts** (`magnetic_scripts/`) – Preconfigured scripts for automation, testing, and simulation–experiment comparison.
3. **RL Environment** (`rigid_catheter/`) – Direct workflow RL environments for catheter control.
4. **Rigid Catheter Asset** (`USD/`) – 3D model and configuration files for catheter simulation.
5. **Experiment** (`Experiment/`) – The code used to play a learned policy in reality using ROS.

---

## Installation

### 1) Install Isaac Sim & Isaac Lab

Follow the official Isaac Lab installation guide:  
<https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>

**Tested System Configuration**
- Isaac Sim: 4.5
- Isaac Lab: 2.0
- Commit: `6b794ac`
- OS: Ubuntu 20.04
- GPU: NVIDIA RTX A4000
- CUDA: 12.2
- NVIDIA Driver: 535.183.01

### 2) Install Additional Packages

Install `mag-manip` in your Isaac Lab Python environment:
```bash
./isaaclab.sh -p -m pip install mag-manip
```
Install `Weights & Biases` in your Isaac Lab Python environment for logging:
```bash
./isaaclab.sh -p -m pip install wandb
```

Install `Optuna` in your Isaac Lab Python environment for the optimization:
```bash
./isaaclab.sh -p -m pip install optuna
```


---

## Where to Place Files in Isaac Lab

Consult the repo structure guide for context:  
<https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/repo_structure.html>

- **Magnetic Plugin**
  - **Path:** `/.../IsaacLab/source/isaaclab/isaaclab/magnetic`
  - Registers as an extension via the Extension Manager (see: <https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/development.html>).
  - Provides magnetic field computation and derived force/torque on embedded magnets.

- **Scripts**
  - **Path:** `/.../IsaacLab/scripts/magnetic_scripts/`
  - Ready-to-run utilities; import the magnetic plugin and can be used to compare simulation to real experiments.

- **Rigid Catheter RL Environment**
  - **Path:** `/.../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/rigid_catheter`
  - Implements three direct-workflow RL environments (tutorial: <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html>).

- **Rigid Catheter Asset**
  - **Path:** `/.../IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/rigidcatheter.py`
  - Installed here so assets are globally available within the Isaac Lab repository.

---

## How to Run

### 1) Scripts

Run scripts as described in the tutorial:  
<https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/create_empty.html>

Example:
```bash
./isaaclab.sh -p scripts/... .py
```

### 2) RL Environment

Launch RL training using SKRL as in the tutorial:  
<https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html>

Example:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-rigidcatheter-v1 --num_envs=64 --headless
```