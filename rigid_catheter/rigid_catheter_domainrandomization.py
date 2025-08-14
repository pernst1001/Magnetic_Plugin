from __future__ import annotations
import torch
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.rigidcatheter import RIGID_CATHETER_COLORED_CFG, GOAL_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
import omni.physics.tensors as tensors # Used for RigidBodyView
import wandb

@configclass
class EventCfg:
    randomize_frictions = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", body_names=".*"),
            "static_friction_range": (0.875, 0.878),
            "dynamic_friction_range": (0.0175, 0.018),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_weight = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),   
            "operation": "scale",
        },
    )
    randomize_collider_offset = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", body_names=".*"),
            "contact_offset_distribution_params": (8.73e-05, 8.74e-05),
            "rest_offset_distribution_params": (2.0e-05, 2.1e-05),
        },
    )
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", joint_names=".*"),
            "friction_distribution_params": (17.0e3, 19.0e3),
            "armature_distribution_params": (5.0e-8, 5.05e-8),
            "operation": "abs",
        },
    )
    
@configclass
class RigidCatheterEnvCfg(DirectRLEnvCfg):
    # Environment parameters
    episode_length_s = 40.0
    action_space = 1
    observation_space = 5
    state_space = 0
    sim_dt: float = 0.00501
    decimation = int(0.02 / sim_dt)  # Decimation factor for rendering and simulation steps
    action_scale: float = 4e-5
    goal_radius: float=0.003
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=sim_dt,
                                        render_interval=decimation,
                                        physx = sim_utils.PhysxCfg(gpu_max_rigid_patch_count=200000))


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.2, replicate_physics=False)
    rigid_chateter_cfg = RIGID_CATHETER_COLORED_CFG.replace(prim_path="/World/envs/env_.*/Catheter")
    goal_cfg = GOAL_CFG.replace(prim_path="World/envs/Goals")
    
    #reward
    rew_scale_positional_reward = 100.0
    rew_scale_bending = -1.0
    rew_scale_goal_reached = 0.0


    body_ids: list[int] = [2, 5, 8, 11, 14, 17, 20, 23, 27, 30, 33, 34, 35]

    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=5.0e-7, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=5.0e-7, operation="add"),
    )

class RigidCatheterEnv(DirectRLEnv):
    """Direct workflow environment for moving a cube to a goal position."""
    
    def __init__(self, cfg: RigidCatheterEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.cfg = cfg
        
        # Initialize buffers
        self._length_catheter = 0.0445
        self._goal_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.torque = torch.zeros(self.num_envs, len(self.cfg.body_ids), 3, device=self.device)
        self.action_scale = cfg.action_scale
        wandb.log({
            "config/episode_length_s": cfg.episode_length_s,
            "config/action_scale": cfg.action_scale,
            "config/goal_radius": cfg.goal_radius,
            "config/rew_scale_positional_reward": cfg.rew_scale_positional_reward,
            "config/rew_scale_bending": cfg.rew_scale_bending,
            "config/rew_scale_goal_reached": cfg.rew_scale_goal_reached,
        })

        
    def _setup_scene(self):
        """Create the scene elements."""
        self.rigid_catheter = Articulation(self.cfg.rigid_chateter_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8772987175254492,
            dynamic_friction=0.017636336305183548,
            compliant_contact_damping=18.891483360903592,
            compliant_contact_stiffness=8315.361151528119,
        )
        ))
        # Add goal to the scene
        self.goal = VisualizationMarkers(self.cfg.goal_cfg)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # Add cube to the scene
        self.scene.articulations["rigid_catheter"] = self.rigid_catheter

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
            
    def _pre_physics_step(self, actions):
        """Apply actions before physics step."""
        actions = actions.clone()
        self.actions = actions
        self.torque, self.force = compute_torque_force(actions, self.action_scale, len(self.cfg.body_ids))
        
    def _apply_action(self):
        """Apply actions to the cube."""
        self.rigid_catheter.set_external_force_and_torque(self.force, self.torque, body_ids=self.cfg.body_ids, env_ids=self.env_ids)
        self.rigid_catheter.write_data_to_sim()
    
    def _get_observations(self):
        rigid_catheter_xy = self.rigid_catheter.data.body_link_pos_w[:, -1, :2] * 1e2
        rigid_catheter_xy = add_position_noise(rigid_catheter_xy, noise_std=0.1)  # Add noise to the catheter position
        goal_xy = self._goal_position[:, :2] * 1e2      # Shape: [num_envs, 2]

        obs = torch.cat([rigid_catheter_xy, goal_xy, self.actions], dim=-1)
        if torch.any(obs.isnan()):
            print("observations are NAN\n", obs)
            raise ValueError("Observations cannot be NAN")
        return {"policy": obs}
    
    def _get_rewards(self):
        # self.goal_distance = torch.abs(self.rigid_catheter.data.body_link_pos_w[:, -1, 0]-self._goal_position[:, 0])
        self.goal_distance = torch.norm(self.rigid_catheter.data.body_link_pos_w[:, -1, :2]-self._goal_position[:, :2], p=2, dim=-1)
        bending = ((self.rigid_catheter.data.body_link_pos_w[:, -1, 1] - self.scene.env_origins[:, 1]) < 0.0)
        goal_reached = self.goal_distance < self.cfg.goal_radius

        reward, r_pos = compute_rewards(
            goal_distance=self.goal_distance,
            bending=bending,
            goal_reached=goal_reached,
            rew_scale_positional_reward=self.cfg.rew_scale_positional_reward,
            rew_scale_bending=self.cfg.rew_scale_bending,
            rew_scale_goal_reached=self.cfg.rew_scale_goal_reached
        )

        wandb.log({
            "reward/total": reward.mean().item(),
            "reward/goal_distance": self.goal_distance.mean().item(),
            "reward/std_goal_distance": self.goal_distance.std().item(),
            "reward/bending_penalty": bending.sum().item(),
            "reward/goals_reached": goal_reached.sum().item(),
            "reward/positional_reward": r_pos.mean().item(),
            "reward/positional_reward_std": r_pos.std().item(),
        })
        if torch.any(reward.isnan()):
            print("rewards are NAN\n", reward)
            raise ValueError("Rewards cannot be NAN")
        return reward
    
    def _get_dones(self):
        """Check if the episode is done."""
        goal_reached = self.goal_distance < self.cfg.goal_radius
        time_out = self.episode_length_buf > self.cfg.episode_length_s
        bending = ((self.rigid_catheter.data.body_link_pos_w[:, -1, 1] - self.scene.env_origins[:, 1]) < 0.0)
        done_mask = goal_reached | bending
        return bending, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Reset the environment indices
        if env_ids is None:
            env_ids = self.rigid_catheter._ALL_INDICES
        super()._reset_idx(env_ids)

        reset_env_ids = torch.argwhere(((self.rigid_catheter.data.body_link_pos_w[:, -1, 1] - self.scene.env_origins[:, 1]) < 0.0) == 1.0).squeeze(-1)
        if reset_env_ids.numel() != 0:
            self._reset_articulation_state(reset_env_ids)

        self._reset_goal_positions(env_ids)


    def _reset_articulation_state(self, env_ids):
        """Reset the articulation state."""
        # Reset the root state
        default_root_states = self.rigid_catheter.data.default_root_state.detach().clone()[env_ids, :]
        default_root_states[:, :3] += self.scene.env_origins[env_ids, :3]  # Set the position to the environment origin
        default_root_states[:, 2] += 0.0013  # Set the z position to 0.0013
        self.rigid_catheter.write_root_pose_to_sim(default_root_states[:, :7], env_ids)
        self.rigid_catheter.write_root_velocity_to_sim(default_root_states[:, 7:], env_ids)

        # Reset the joint state
        default_joint_pos = self.rigid_catheter.data.default_joint_pos.detach().clone()[env_ids, :]
        default_joint_vel = self.rigid_catheter.data.default_joint_vel.detach().clone()[env_ids, :]
        self.rigid_catheter.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    def _reset_goal_positions(self, env_ids):
        """Reset the goal positions."""
        num_reset = len(env_ids)
        self._goal_position[env_ids] = compute_goal_position(num_reset, self._length_catheter) + self.scene.env_origins[env_ids]
        self.goal.visualize(translations=self._goal_position)
        # self.goal_distance = torch.abs(self.rigid_catheter.data.body_link_pos_w[:, -1, 0]-self._goal_position[:, 0])
        self.goal_distance = torch.norm(self.rigid_catheter.data.body_link_pos_w[:, -1, :2]-self._goal_position[:, :2], p=2, dim=-1)
        self.torque[env_ids, : , :] = torch.zeros(num_reset, len(self.cfg.body_ids), 3, device=self.device).detach().clone()

@torch.jit.script
def compute_rewards(
    goal_distance: torch.Tensor,
    bending: torch.Tensor,
    goal_reached: torch.Tensor,
    rew_scale_positional_reward: float,
    rew_scale_bending: float,
    rew_scale_goal_reached: float
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """Calculate the rewards."""
    r_pos = -rew_scale_positional_reward * (goal_distance)**2
    r_bend = rew_scale_bending * bending.float()
    r_goal = rew_scale_goal_reached * goal_reached.float()
    return r_pos + r_bend + r_goal, r_pos

@torch.jit.script
def compute_torque_force(
    actions: torch.Tensor,
    action_scale: float,
    len_body_ids: int
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """Compute the torque and force based on actions."""
    zeros = torch.zeros_like(actions)
    torque = torch.cat([zeros, zeros, actions], dim=-1)
    torque = (torque * action_scale)
    torque = torque.unsqueeze(1).expand(-1, len_body_ids, -1)
    force = torch.zeros_like(torque)
    return torque, force

@torch.jit.script
def compute_goal_position(
    num_envs: int,
    length_catheter: float = 0.0445
) -> torch.Tensor:
    """Compute the goal position for the environment."""
    angles = torch.rand(num_envs, device="cuda") * torch.pi / 2 + torch.pi / 4
    x = length_catheter * torch.cos(angles)
    y = length_catheter * torch.sin(angles) - 0.0223
    z = torch.ones_like(x) * 0.0013
    return torch.stack((x, y, z), dim=-1)

@torch.jit.script
def add_position_noise(
    positions: torch.Tensor,
    noise_std: float = 0.0001
) -> torch.Tensor:
    """Add Gaussian noise to the positions."""
    noise = torch.randn_like(positions) * noise_std
    return positions + noise
