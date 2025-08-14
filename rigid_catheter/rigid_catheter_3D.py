from __future__ import annotations
import torch
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.rigidcatheter import RIGID_CATHETER_COLORED_CFG, GOAL_CFG, FIELD_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.magnetic import RLMagneticEntity
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
import wandb
import isaaclab.utils.math as math_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.envs.ui import ViewportCameraController


@configclass
class EventCfg:
    randomize_frictions = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", body_names=".*"),
            "static_friction_range": (0.84, 0.91),
            "dynamic_friction_range": (0.01, 0.022),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_weight = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1), 
            "operation": "scale",
        },
    )
    randomize_collider_offset = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", body_names=".*"),
            "contact_offset_distribution_params": (8.4e-05, 9.0e-05),
            "rest_offset_distribution_params": (1.8e-05, 2.3e-05),
        },
    )
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("rigid_catheter", joint_names=".*"),
            "friction_distribution_params": (16.0e3, 20.0e3),
            "armature_distribution_params": (4.9e-8, 5.2e-8),
            "operation": "abs",
        },
    )
    
@configclass
class RigidCatheterEnvCfg(DirectRLEnvCfg):
    # Environment parameters
    episode_length_s = 20.0
    action_space = 2
    observation_space = 8
    state_space = 0
    sim_dt: float = 0.005
    goal_radius: float=0.003

    camera_dt: float = 1.0/10.0
    field_magnitude: float = 0.02
    max_angle_divider_field_phi: float = 3.0
    max_angle_divider_field_theta: float = 10.0
    max_angle_divider_goal_phi: float = 5.0
    max_angle_divider_goal_theta: float = 3.0
    position_noise_std: float = 0.1 # Noise in cm
    current_noise_std: float = 0.2 # Noise in A

    decimation = int(camera_dt / sim_dt)  # Decimation factor for rendering and simulation steps
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=sim_dt,
                                        render_interval=decimation,
                                        physx = sim_utils.PhysxCfg(gpu_max_rigid_patch_count=1200000,
                                                                   solver_type=0))


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.2, replicate_physics=False)
    rigid_chateter_cfg = RIGID_CATHETER_COLORED_CFG.replace(prim_path="/World/envs/env_.*/Catheter")
    goal_cfg = GOAL_CFG.replace(prim_path="/World/envs/Goals")
    field_cfg = FIELD_CFG.replace(prim_path="/World/envs/Field_arrows")
    
    #reward
    rew_scale_positional_reward = 50.0
    rew_scale_bending = -0.0
    rew_scale_goal_reached = 1.0


    body_ids: list[int] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 33, 34, 35]

    events: EventCfg = EventCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 0.15),
        lookat=(0.0, 0.0, 0.0013),
        resolution=(3840, 2160)
    )


class RigidCatheterEnv(DirectRLEnv):
    """Direct workflow environment for moving a cube to a goal position."""
    
    def __init__(self, cfg: RigidCatheterEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.cfg = cfg
        if self.cfg.max_angle_divider_goal_phi < self.cfg.max_angle_divider_field_phi:
            raise ValueError("max_angle_divider_goal must be greater than or equal to max_angle_divider_field")
        # Initialize buffers
        self._length_catheter = 0.0434
        self._goal_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_rigid_catheter_pos = torch.zeros(self.num_envs, 1, device=self.device)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.max_angle_goal_phi = torch.pi / cfg.max_angle_divider_goal_phi
        self.max_angle_goal_theta = torch.pi / cfg.max_angle_divider_goal_theta
        self.max_angle_field_phi = torch.pi / cfg.max_angle_divider_field_phi
        self.max_angle_field_theta = torch.pi / cfg.max_angle_divider_field_theta
        self.first_it = True
        wandb.log({
            "config/episode_length_s": cfg.episode_length_s,
            "config/goal_radius": cfg.goal_radius,
            "config/rew_scale_positional_reward": cfg.rew_scale_positional_reward,
            "config/rew_scale_bending": cfg.rew_scale_bending,
            "config/rew_scale_goal_reached": cfg.rew_scale_goal_reached,
            "config/field_magnitude": cfg.field_magnitude,
            "config/max_angle_divider_field_phi": cfg.max_angle_divider_field_phi,
            "config/max_angle_divider_field_theta": cfg.max_angle_divider_field_theta,
            "config/max_angle_divider_goal_phi": cfg.max_angle_divider_goal_phi,
            "config/max_angle_divider_goal_theta": cfg.max_angle_divider_goal_theta,
            "config/cam_dt": cfg.camera_dt,
            "config/current_noise_std": cfg.current_noise_std,
            "config/position_noise_std": cfg.position_noise_std,
        })

    def get_xy_orientation_direction(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the quaternion from the field vector using the shortest arc method."""
        # Normalize the field vector
        field_norm = vec / vec.norm(dim=1, keepdim=True)
        
        # Set up the arrow vector (starting direction)
        arrow_vector = torch.zeros_like(field_norm)
        arrow_vector[:, 0] = 1.0
        
        # Find the cross product and dot product
        cross_prod = torch.cross(arrow_vector, field_norm, dim=1)
        dot_prod = torch.sum(arrow_vector * field_norm, dim=1)
        

        quat = torch.zeros((vec.shape[0], 4), device=self.device)
        quat[:, 0] = 1.0 + dot_prod  # w component
        quat[:, 1:] = cross_prod     # x, y, z components
        
        # Normalize the quaternions
        quat = quat / quat.norm(dim=1, keepdim=True)
        
        # Scale factors for visualization
        scales = torch.tensor([0.02, 0.01, 0.01], device=self.device).unsqueeze(0).repeat(vec.shape[0], 1)
        
        return quat, scales
    
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
        self.field_arrows = VisualizationMarkers(self.cfg.field_cfg)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # Add cube to the scene
        self.scene.articulations["rigid_catheter"] = self.rigid_catheter

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
            
    def _pre_physics_step(self, actions):
        """Apply actions before physics step."""
        self.actions = actions.clone()
        fields = compute_field_from_angles(
            actions, field_magnitude=self.cfg.field_magnitude, max_angle_phi=self.max_angle_field_phi, max_angle_theta=self.max_angle_field_theta,
        )
        fields[:, 2] = 0.0  # Set z-component to zero for 2D field
        pos = self.scene.env_origins# + torch.tensor([0.0, 0.0, 0.001], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        quat, scales = self.get_xy_orientation_direction(fields)
        self.field_arrows.visualize(pos, quat, scales)
        self.currents = self.magnet_entity.get_currents_from_field(fields)

    def _apply_action(self):
        """Apply actions to the cube."""
        currents = add_noise(self.currents, noise_std=self.cfg.current_noise_std)  # Add noise to the currents
        forces, torques = self.magnet_entity.get_force_torque_from_currents(currents)
        self.rigid_catheter.set_external_force_and_torque(forces, torques, body_ids=self.cfg.body_ids, env_ids=self.env_ids)
        self.rigid_catheter.write_data_to_sim()
    
    def _get_observations(self):
        rigid_catheter_x = self.rigid_catheter.data.body_link_pos_w[:, -1, 0] * 1e2
        rigid_catheter_z = self.rigid_catheter.data.body_link_pos_w[:, -1, 2] * 1e2 * 0.0
        rigid_catheter_pos = torch.cat([rigid_catheter_x.unsqueeze(-1), rigid_catheter_z.unsqueeze(-1)], dim=-1)  # Shape: [num_envs, 2]
        rigid_catheter_pos = add_noise(rigid_catheter_pos, noise_std=self.cfg.position_noise_std)  # Add noise to the catheter position
        last_rigid_catheter_pos = self.last_rigid_catheter_pos.clone()
        goal_pos_x = self._goal_position[:, 0] * 1e2  # Shape: [num_envs]
        goal_pos_z = self._goal_position[:, 2] * 1e2 * 0.0
        goal_pos = torch.stack([goal_pos_x, goal_pos_z], dim=-1)
        obs = torch.cat([rigid_catheter_pos, last_rigid_catheter_pos, goal_pos, self.actions], dim=-1)
        self.last_rigid_catheter_pos = rigid_catheter_pos.clone()
        if torch.any(obs.isnan()):
            print("observations are NAN\n", obs)
            raise ValueError("Observations cannot be NAN")
        return {"policy": obs}
    
    def _get_rewards(self):
        goal_distance_x = torch.abs(self.rigid_catheter.data.body_link_pos_w[:, -1, 0] - self._goal_position[:, 0])
        goal_distance_z = torch.abs(self.rigid_catheter.data.body_link_pos_w[:, -1, 2] - self._goal_position[:, 2]) * 0.0 
        self.goal_distance = torch.sqrt(goal_distance_x**2 + goal_distance_z**2)
        bending = ((self.rigid_catheter.data.body_link_pos_w[:, -1, 1] - self.scene.env_origins[:, 1]) < -0.005)
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
            print("Goal distance:", self.goal_distance)
            print("Bending:", bending)
            print("Goal reached:", goal_reached)
            print("positions", self.rigid_catheter.data.body_link_pos_w[:, -1, 1])
            raise ValueError("Rewards cannot be NAN")
        return reward
    
    def _get_dones(self):
        """Check if the episode is done."""
        goal_reached = self.goal_distance < self.cfg.goal_radius
        time_out = self.episode_length_buf > self.cfg.episode_length_s
        bending = ((self.rigid_catheter.data.body_link_pos_w[:, -1, 1] - self.scene.env_origins[:, 1]) < -0.005)
        done_mask = goal_reached | bending
        return bending, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Reset the environment indices
        if env_ids is None:
            env_ids = self.rigid_catheter._ALL_INDICES
        super()._reset_idx(env_ids)
        if self.first_it:
            self.first_it = False
            direction = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
            self.magnet_entity = RLMagneticEntity(volume=4.231e-9, remanence=1.45, direction=direction, origins=self.scene.env_origins, magnet=self.rigid_catheter, body_ids=self.cfg.body_ids)

            
        reset_env_ids = torch.argwhere(((self.rigid_catheter.data.body_link_pos_w[:, -1, 1] - self.scene.env_origins[:, 1]) < -0.005) == 1.0).squeeze(-1)
        if reset_env_ids.numel() != 0:
            self._reset_articulation_state(reset_env_ids)

        self._reset_goal_positions(env_ids)
        rigid_catheter_x = self.rigid_catheter.data.body_link_pos_w[:, -1, 0] * 1e2
        rigid_catheter_z = self.rigid_catheter.data.body_link_pos_w[:, -1, 2] * 1e2 * 0.0
        self.last_rigid_catheter_pos = torch.cat([rigid_catheter_x.unsqueeze(-1), rigid_catheter_z.unsqueeze(-1)], dim=-1)  # Shape: [num_envs, 2]


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
        self._goal_position[env_ids] = compute_goal_position(num_reset, self._length_catheter, self.max_angle_goal_phi, self.max_angle_goal_theta) + self.scene.env_origins[env_ids]
        self.goal.visualize(translations=self._goal_position)
        goal_distance_x = torch.abs(self.rigid_catheter.data.body_link_pos_w[:, -1, 0] - self._goal_position[:, 0])
        goal_distance_z = torch.abs(self.rigid_catheter.data.body_link_pos_w[:, -1, 2] - self._goal_position[:, 2]) * 0.0
        self.goal_distance = torch.sqrt(goal_distance_x**2 + goal_distance_z**2)


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
def compute_goal_position(
    num_envs: int,
    length_catheter: float,
    max_angle_phi: float,
    max_angle_theta: float,
    minimal_theta: float = torch.pi / 2.0016565828721946
) -> torch.Tensor:
    """Compute the goal position for the environment."""
    angles = torch.rand(num_envs, device="cuda:0") *2*max_angle_phi - max_angle_phi  # Random angles in the range [-max_angle, max_angle]
    angles2 = torch.rand(num_envs, device="cuda:0") * (max_angle_theta - minimal_theta) + minimal_theta  # Random angles in the range [minimal_theta, max_angle_theta]
    x = length_catheter * torch.sin(angles) * torch.sin(angles2)
    y = length_catheter * torch.cos(angles) * torch.sin(angles2) - 0.0203
    z = length_catheter* torch.cos(angles2)
    return torch.stack((x, y, z), dim=-1)

@torch.jit.script
def add_noise(
    input: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    """Add Gaussian noise to the positions."""
    noise = torch.randn_like(input) * noise_std
    return input + noise

@torch.jit.script
def compute_field_from_angles(
    angles: torch.Tensor,
    field_magnitude: float,
    max_angle_phi: float,
    max_angle_theta: float,
    minimal_theta: float = torch.pi / 2.0
) -> torch.Tensor:
    """Compute the magnetic field from angles."""
    angles1 = max_angle_phi * torch.tanh(angles[:, 0])
    angles2 = (torch.tanh(angles[:, 1])+1.0) * (max_angle_theta - minimal_theta)/2 + minimal_theta
    x = field_magnitude * torch.sin(angles1) * torch.sin(angles2)
    y = field_magnitude * torch.cos(angles1) * torch.sin(angles2)
    z = field_magnitude * torch.cos(angles2)
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # Reshape to [num_envs, 3]