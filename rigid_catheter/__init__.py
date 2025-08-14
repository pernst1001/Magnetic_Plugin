# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Magnetic continuum robot environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-rigidcatheter-v0",
    entry_point=f"{__name__}.rigid_catheter_domainrandomization:RigidCatheterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rigid_catheter_domainrandomization:RigidCatheterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-rigidcatheter-v1",
    entry_point=f"{__name__}.rigid_catheter_MPEM:RigidCatheterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rigid_catheter_MPEM:RigidCatheterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-rigidcatheter-v2",
    entry_point=f"{__name__}.rigid_catheter_3D:RigidCatheterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rigid_catheter_3D:RigidCatheterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_big.yaml",
    },
)

gym.register(
    id="Isaac-rigidcatheter-kamerafahrt",
    entry_point=f"{__name__}.rigid_catheter_kamerafahrt:RigidCatheterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rigid_catheter_kamerafahrt:RigidCatheterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_big.yaml",
    },
)