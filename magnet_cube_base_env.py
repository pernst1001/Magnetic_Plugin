from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a floating cube environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


class RandomForceActionTerm(ActionTerm):
    """Simple action term that applies random forces to the cube."""

    _asset: RigidObject
    """The cube asset on which forces are applied."""

    def __init__(self, cfg: RandomForceActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # create buffer for forces
        self._forces = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        # force scaling factor
        self.force_scale = cfg.force_scale

    @property
    def action_dim(self) -> int:
        return self._forces.shape[1]  # Forces in x, y, z directions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._forces

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._forces

    def process_actions(self, actions: torch.Tensor):
        """Store the random forces."""
        self._forces[:] = actions * self.force_scale
        self._forces.reshape(self._forces.shape[0], 1, 3)

    def apply_actions(self):
        """Apply the forces to the cube."""
        self._vel_command[:, :3] = self._forces
        for i, force in enumerate(self._forces):
            # Apply force to the cube
            self._asset.set_external_force_and_torque(forces=force, torques=torch.zeros_like(force), env_ids=[i])
        self._asset.write_data_to_sim()

@configclass
class RandomForceActionTermCfg(ActionTermCfg):
    """Configuration for the random force action term."""

    class_type: type = RandomForceActionTerm
    force_scale: float = 20.0  # Scaling factor for the forces

##
# Observation term - just the cube position
##

def cube_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cube position relative to environment origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def out_of_bounds(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, bounds: tuple) -> torch.Tensor:
    """Terminate when cube leaves workspace.
    
    Args:
        bounds: Tuple of (x_max, y_max, z_max) defining workspace boundaries
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get position relative to environment origin
    pos = asset.data.root_pos_w - env.scene.env_origins
    # Check if any coordinate exceeds bounds (absolute value)
    return (pos.abs() > torch.tensor(bounds, device=env.device)).any(dim=1)
##
# Scene definition
##

@configclass
class SimpleCubeSceneCfg(InteractiveSceneCfg):
    """Simple scene with just a floating cube."""

    # Ground plane
    terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5, restitution=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# Environment configuration
##

@configclass
class ActionsCfg:
    """Action specifications - just random forces."""
    random_force = RandomForceActionTermCfg(asset_name="cube")

@configclass
class ObservationsCfg:
    """Observation specifications - just cube position."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        cube_pos = ObsTerm(func=cube_position, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("cube"), "bounds": ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))},
    )

@configclass
class EventCfg:
    """Simple reset event - just randomize cube position."""

    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.5, 1.0)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

@configclass
class SimpleCubeEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the simple cube environment."""

    scene: SimpleCubeSceneCfg = SimpleCubeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Basic environment settings."""
        self.decimation = 2  # control frequency = sim_frequency / decimation
        self.sim.dt = 0.01  # simulation timestep

##
# Main function to test the environment
##

def main():
    """Test the simple cube environment with random forces."""
    
    # setup environment
    env = ManagerBasedEnv(cfg=SimpleCubeEnvCfg())
    
    # simulate physics
    count = 0
    obs, _ = env.reset()
    
    while True:
        with torch.inference_mode():
            # reset periodically
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # generate random forces between -1 and 1
            actions = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
            actions[:, 2] = 0.0
            
            obs, _ = env.step(actions)
            # print cube position
            if count % 50 == 0:
                avg_pos = obs["policy"].mean(dim=0)
                print(f"[Step: {count:04d}]: Average cube position: {avg_pos}")
            
            count += 1

if __name__ == "__main__":
    main()