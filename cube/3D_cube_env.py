from __future__ import annotations
import torch
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from .create_prims import CUBE_CFG, GOAL_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils

@configclass
class CubeEnvCfg(DirectRLEnvCfg):
    # Environment parameters
    decimation = 2
    episode_length_s = 100.0
    action_space = 3
    observation_space = 6
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    cube_cfg = CUBE_CFG.replace(prim_path="/World/envs/env_.*/Cube")
    goal_cfg = GOAL_CFG
    #reward
    rew_scale_terminated = -4.0
    rew_scale_goal_reached = 5.0
    rew_scale_progress = 1.0
    rew_not_advance = -0.1

    action_scale_xy: float = 10.0
    action_scale_z: float = 40.0
    goal_radius: float = 0.1


class CubeEnv(DirectRLEnv):
    """Direct workflow environment for moving a cube to a goal position."""
    
    def __init__(self, cfg: CubeEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
        
        # Initialize buffers
        self._goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.workspace_bounds = torch.tensor([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]], device=self.device)
        self.action_scale_xy = cfg.action_scale_xy
        self.action_scale_z = cfg.action_scale_z
        
    def _setup_scene(self):
        """Create the scene elements."""
        self.cube = RigidObject(self.cfg.cube_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Add goal to the scene
        self.goal = VisualizationMarkers(self.cfg.goal_cfg)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # Add cube to the scene

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
            
    def _pre_physics_step(self, actions):
        """Apply actions before physics step."""
        self.actions = actions.clone()
        self.actions[:, :2] *= self.cfg.action_scale_xy
        self.actions[:, 2] *= self.cfg.action_scale_z

    def _apply_action(self):
        """Apply actions to the cube."""
        # Apply forces to cube
        for i, actions in enumerate(self.actions):
            self.cube.set_external_force_and_torque(forces=actions, torques=torch.zeros_like(actions), env_ids=[i])
        self.cube.write_data_to_sim()
    
    def _get_observations(self):
        self.cube.update(self.sim.get_physics_dt())
        self._position_error_vector = self._goal_positions - self.cube.data.root_pos_w[:, :3]
        self._previous_position_error = self._position_error
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        cube_xy = self.cube.data.root_pos_w[:, :3]  # Shape: [num_envs, 2]
        goal_xy = self._goal_positions[:, :3]       # Shape: [num_envs, 2]
    
        # Concatenate along dim 1 to get [num_envs, 4]
        obs = torch.cat([cube_xy, goal_xy], dim=1)
        if torch.any(obs.isnan()):
            print("observations are NAN\n", obs)
            raise ValueError("Observations cannot be NAN")
        return {"policy": obs}
    
    def _get_rewards(self):
        position_reward = self._previous_position_error - self._position_error
        goal_reached = self._position_error < self.cfg.goal_radius
        relative_pos = self.cube.data.root_pos_w - self.scene.env_origins
        out_of_bounds = ((relative_pos < self.workspace_bounds[0]) | (relative_pos > self.workspace_bounds[1])).any(dim=1)
        not_advancing =  position_reward < 1e-20
        time_out = self.episode_length_buf > self.cfg.episode_length_s
        composite_reward = (
            self.cfg.rew_scale_terminated * out_of_bounds.float() +
            self.cfg.rew_scale_goal_reached * goal_reached.float() +
            self.cfg.rew_scale_progress * position_reward +
            self.cfg.rew_not_advance * not_advancing.float() +
            self.cfg.rew_scale_terminated * time_out.float()
        )
        if torch.any(composite_reward.isnan()):
            print("rewards are NAN\n", composite_reward)
            raise ValueError("Rewards cannot be NAN")
        return composite_reward
    
    def _get_dones(self):
        """Check if the episode is done."""
        relative_pos = self.cube.data.root_pos_w - self.scene.env_origins
        out_of_bounds = ((relative_pos < self.workspace_bounds[0]) | (relative_pos > self.workspace_bounds[1])).any(dim=1)
        goal_reached = self._position_error < self.cfg.goal_radius
        time_out = self.episode_length_buf > self.cfg.episode_length_s
        done_mask = out_of_bounds | goal_reached
        return done_mask, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Reset the environment indices
        if env_ids is None:
            env_ids = self.cube._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        default_root_state = self.cube.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 2] = 0.1
        self.cube.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        self._goal_positions[env_ids] = torch.rand(num_reset, 3, device=self.device) * 2 - 1 + self.scene.env_origins[env_ids]
        self._goal_positions[env_ids, 2] /= 2
        self._goal_positions[env_ids, 2] += 0.5
        

        self.goal.visualize(translations=self._goal_positions)

        self._position_error_vector = self._goal_positions - self.cube.data.root_pos_w[:, :3]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()
        