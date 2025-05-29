# Magnetic Plugin for IsaacLab

This repository contains various components for working with magnetic simulations in IsaacLab.

## Repository Structure

1. **Magnetic Plugin** - Custom plugin for magnetic field simulations (folder magnetic)
2. **Scripts** - Utility scripts for automation and testing
3. **Cube RL Environment** - Reinforcement learning environment for magnetic control
4. **Optimization Framework** - Tools for optimizing magnetic control parameters
5. **RigidCatheter Asset** - 3D model and configuration for catheter simulation

## Installation Guide

### Where to Place Files in IsaacLab

- **Magnetic Plugin**: 
  - Place in `/.../IsaacLab/source/isaaclab/isaaclab/magnetic`
  - Register the extension in the extension manager

- **Scripts**:
  - Place in `/.../IsaacLab/scripts/magnetic_scripts/`
  - Everything prepared to run

- **Cube RL Environment**:
  - Place in `/.../Isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/cube`

- **Optimization Framework**:
  - Place in `/.../IsaacLab/scripts/magnetic_scripts/catheter_optimization`

- **RigidCatheter Asset**:
  - Place in `/.../IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/rigidcatheter.py`

## Component Details

### 1. Magnetic Plugin

A custom extension for IsaacLab that simulates magnetic fields and their interaction with objects.

**Usage**:
- Enable through the extension
- Apply force & torque to magnets through currents
- Get Currents through backward model

### 2. Scripts

Collection of Python scripts.

**Usage**:
- All the used scripts from magnetic cube to deformable catheter to rigidcatheter

### 3. Cube RL Environment

Reinforcement learning environment for training magnetic control policies.

**Usage**:
- Direct RL environement to train a cube

### 4. Optimization Framework

Tools for optimizing simulation parameter

**Usage**:
- Define objective function
- Set parameters
- Run optimization using optuna

### 5. RigidCatheter Asset

Loading USD asset as Articulation

