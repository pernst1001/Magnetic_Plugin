import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# Add this near the top of RigidCatheter_Sim2Real.py
parser.add_argument("--joint_friction", type=float, default=18.0e3, help="Joint friction coefficient")
parser.add_argument("--joint_armature", type=float, default=5.035780253531126e-8, help="Joint armature value")
parser.add_argument("--static_friction", type=float, default=0.8772987175254492, help="Static friction for ground plane")
parser.add_argument("--dynamic_friction", type=float, default=0.017636336305183548, help="Dynamic friction for ground plane")
parser.add_argument("--restitution", type=float, default=0.0, help="Restitution for ground plane")
parser.add_argument("--linear_damping", type=float, default=0.05, help="Linear dampening for the rigid body")
parser.add_argument("--angular_damping", type=float, default=0.05, help="Angular dampening for the rigid body")
parser.add_argument("--output_file", type=str, default='None', help="Path to save optimization results")
parser.add_argument("--sim_dt", type=float, default=0.00501, help="Simulation time step")
parser.add_argument("--position_iteration_count", type=int, default=11)
parser.add_argument("--velocity_iteration_count", type=int, default=0)
parser.add_argument("--compliant_contact_damping", type=float, default=18.891483360903592, help="Compliant contact damping for the rigid body")
parser.add_argument("--compliant_contact_stiffness", type=float, default=8315.361151528119, help="Compliant contact stiffness for the rigid body")
parser.add_argument("--contact_offset", type=float, default=8.739257490691612e-05, help="Contact offset for the rigid body")
parser.add_argument("--rest_offset", type=float, default=2.078435319546033e-05, help="Rest offset for the rigid body")
parser.add_argument("--bag_number", type=int, default=1, help="Bag number to use for the simulation")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# args_cli.position_iteration_count = 0.0
# args_cli.velocity_iteration_count = 0.0

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import omni
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
import omni.physics.tensors as tensors # Used for RigidBodyView
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
import torch.nn.functional as F         # new
from isaaclab.magnetic import MagneticEntity
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.utils.math as math_utils


import tqdm as tqdm
import math

# def get_xy_orientation_direction(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     vec_angle = torch.atan2(vec[:, 1], vec[:, 0])
#     zeros = torch.zeros_like(vec_angle)
#     quat = math_utils.quat_from_euler_xyz(zeros, zeros, vec_angle)
#     scales = torch.tensor([0.02, 0.003, 0.003]).unsqueeze(0).repeat(vec.shape[0], 1)
#     return quat, scales

# def get_xyz_orientation_direction(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     b : (...,3) target directions, ‖b‖>0
#     returns (...,4) unit quaternions (w, x, y, z)
#     """
#     b = b/ (torch.linalg.norm(b, dim=1, keepdim=True)+1e-9)  # normalize b
#     a = torch.tensor([1.0, 0.0, 0.0], device=b.device).expand_as(b)
#     v  = torch.cross(a, b, dim=-1)                     # rotation axis ∥v∥ = sinθ
#     w  = (a * b).sum(-1, keepdim=True) + 1.0          # w  = 1+cosθ , ≥0
#     quat = torch.cat([w, v], dim=-1)
#     quat = F.normalize(quat, dim=-1)
#     # Handle ‖b‖≈0 → identity quaternion
#     quat[w.squeeze(-1) < 1.0e-8] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=b.device, dtype=quat.dtype)
#     scales = torch.tensor([0.02, 0.003, 0.003], device=b.device).unsqueeze(0).repeat(b.shape[0], 1)
#     scales = scales * torch.linalg.norm(b, dim=1, keepdim=True)  # Scale the arrow length by the norm of b
#     return quat, scales

def set_articulation_collider_offset(articulation:Articulation, rest_offset_param: float, contact_offset_param: float):
    """Set the contact and rest offsets for all shapes in the articulation.
    
    Args:
        articulation: The articulation to apply offsets to
        rest_offset_param: Rest offset value to set for all shapes
        contact_offset_param: Contact offset value to set for all shapes
    """

    root_physx_view = articulation.root_physx_view
    rest_offset = root_physx_view.get_rest_offsets().clone()
    contact_offset = root_physx_view.get_contact_offsets().clone()

    # Update the rest and contact offsets with the provided parameters
    rest_offset = torch.ones_like(rest_offset) * rest_offset_param
    contact_offset = torch.ones_like(contact_offset) * contact_offset_param

    # Apply the updated offsets to the simulation
    env_ids = torch.arange(0, device=articulation.device)
    root_physx_view.set_rest_offsets(rest_offset, env_ids)
    root_physx_view.set_contact_offsets(contact_offset, env_ids)

def set_articulation_material_friction(articulation:Articulation, static_friction:float, dynamic_friction:float, restitution:float):
    """Set the physics material properties (friction) for all shapes in an articulation.
    
    Args:
        articulation: The articulation to apply friction values to
        static_friction: Static friction coefficient value
        dynamic_friction: Dynamic friction coefficient value
    """
    # Get physics view for the articulation
    root_physx_view = articulation.root_physx_view
    
    # Get the current material properties
    materials = root_physx_view.get_material_properties()
    
    # Set static and dynamic friction for all shapes
    # Materials tensor has shape: (num_envs, num_shapes, 3)
    # Where the 3 values are: static friction, dynamic friction, restitution
    materials[:, :, 0] = static_friction  # Set static friction
    materials[:, :, 1] = dynamic_friction  # Set dynamic friction
    materials[:, :, 2] = restitution  # Set restitution to 0.0 (or any other value you want)
    
    # Apply the updated materials to the simulation
    env_ids = torch.arange(0, device=articulation.device)
    root_physx_view.set_material_properties(materials, env_ids)    
    
def create_currents_tensor():
    csv_file_name = f"2025_06_05_v{args_cli.bag_number}"
    print("Using csv file: ", csv_file_name)
    csv_file = f"/home/pascal/Documents/Rosbags_Accuracy/bags/{csv_file_name}/{csv_file_name}.csv"

    #read out the times and current tensors of each row and safe them in a tensor
    # header looks like this: timestamp,current_0,current_1,current_2,current_3,current_4,current_5,current_6,current_7
    #everything should be saved as tensor

    #open the file
    with open(csv_file, "r") as f:
        #skip the first line
        next(f)
        #read the rest of the lines
        lines = f.readlines()
        #create a tensor for the currents
        currents = torch.zeros((len(lines), 3), device="cuda")
        #create a tensor for the times
        times = torch.zeros((len(lines), 1), device="cuda")
        #create a tensor for the positions
        ground_truth_positions = torch.zeros((len(lines), 8), device="cuda")
        #iterate over the lines
        for i, line in enumerate(lines):
            #split the line by comma
            values = line.split(",")
            #save the values in the tensors
            times[i] = float(values[0])
            currents[i] = torch.tensor([float(x) for x in values[1:4]])
            ground_truth_positions[i] = torch.tensor([float(x) for x in values[4:]])
        return currents, times, ground_truth_positions

def design_scene():
    # Ground-plane
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    # Lights
    # Load the RigidCatheter
    prim_utils.create_prim("/World/Origin1", "Xform", translation=(0.0, -0.02015, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711))
    rigid_catheter_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/pascal/Downloads/My_Joints/RigidCatheterColored.usd", #TODO: Add the usd path to the repo
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                sleep_threshold=0.0,
                stabilization_threshold=1e-6,
                solver_position_iteration_count=args_cli.position_iteration_count,
                solver_velocity_iteration_count=args_cli.velocity_iteration_count,
                # enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=args_cli.position_iteration_count,
                solver_velocity_iteration_count=args_cli.velocity_iteration_count,
                sleep_threshold=0.0,
                stabilization_threshold=1e-6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
        }
    )
    # rigid_catheter_cfg = RIGID_CATHETER_CFG
    rigid_catheter_cfg.prim_path = "/World/Origin1/rigid_catheter"
    rigid_catheter = Articulation(cfg = rigid_catheter_cfg)


    cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.static_friction,
            dynamic_friction=args_cli.dynamic_friction,
            restitution=args_cli.restitution,
            compliant_contact_damping=args_cli.compliant_contact_damping,
            compliant_contact_stiffness=args_cli.compliant_contact_stiffness,
        )
    )
    cfg.func("/World/defaultGroundPlane", cfg)

    CONES_CFG =VisualizationMarkersCfg(
        prim_path="/World/Goal/Cones",
        markers={
            "cone_0": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    ground_truth_catheter = VisualizationMarkers(CONES_CFG)
    CONES_CFG =VisualizationMarkersCfg(
        prim_path="/World/Goal/Cones",
        markers={
            "cone_0": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )
    colored_magnet = VisualizationMarkers(CONES_CFG)
    # TORQUE_ARROW_CFG = VisualizationMarkersCfg(
    #     prim_path="/World/Goal/TorqueArrows",
    #     markers={
    #         "torque_arrow": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",   # points +X  :contentReference[oaicite:0]{index=0}
    #             scale=(1.0, 0.2, 0.2),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         ),
    #     },
    # )
    # torque_arrow_markers = VisualizationMarkers(TORQUE_ARROW_CFG)
    # FIELD_ARROW_CONFIG = VisualizationMarkersCfg(
    #     prim_path="/World/Goal/FieldArrow",
    #     markers={
    #         "field_arrow": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",   # points +X  :contentReference[oaicite:0]{index=0}
    #             scale=(1.0, 0.2, 0.2),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    #         ),
    #     },
    # )
    # field_arrow = VisualizationMarkers(FIELD_ARROW_CONFIG)
    scene_entities = {"rigid_catheter": rigid_catheter, "ground_truth_catheter": ground_truth_catheter, "colored_magnet": colored_magnet}
    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    rigid_catheter = entities["rigid_catheter"]
    ground_truth_vis = entities["ground_truth_catheter"]
    colored_magnet_vis = entities["colored_magnet"]
    rigid_catheter.write_joint_friction_coefficient_to_sim(args_cli.joint_friction)
    rigid_catheter.write_joint_armature_to_sim(args_cli.joint_armature)
    set_articulation_material_friction(rigid_catheter, args_cli.static_friction, args_cli.dynamic_friction, args_cli.restitution)
    set_articulation_collider_offset(rigid_catheter, args_cli.rest_offset, args_cli.contact_offset)

    sim_time = 0.0
    # sim.set_simulation_dt(physics_dt=1.0/(40.0*3), rendering_dt=2)
    sim_dt = sim.get_physics_dt()
    count = 0
    sim_view = tensors.create_simulation_view("torch")
    magnet_view = sim_view.create_rigid_body_view("/World/Origin1/rigid_catheter/Magnet_*")
    direction = torch.tensor([[0.0, 1.0, 0.0]]*magnet_view.count, device="cuda")
    magnet_entity = MagneticEntity(volume=4.231e-9, remanence=1.45, direction=direction, magnet=magnet_view)
    currents, times, ground_truth_positions = create_currents_tensor()
    euclidean_norms = []
    print("Time steps per current set: ", int(0.02 / sim_dt))
    # sim_view.initialize_kinematic_bodies()
    # for i in range(3):
    #     sim.step()
    #     sim_time += sim_dt
    #     count += 1
    for current_set, ground_truth_position in tqdm.tqdm(zip(currents, ground_truth_positions), total=currents.shape[0]):
        ones = torch.ones(size=(4,1), device="cuda")*0.0013
        ground_truth = torch.concat((torch.reshape(ground_truth_position, (4,2)), ones), dim=1)
        ground_truth_vis.visualize(translations=ground_truth)
        magnet_position = magnet_entity.get_magnets_position()
        colored_magnet_position = torch.cat((magnet_position[11, :3], magnet_position[7, :3], magnet_position[4, :3], magnet_position[1, :3]), dim=0)
        colored_magnet = colored_magnet_position.reshape(4, 3)
        colored_magnet_vis.visualize(translations=colored_magnet)
        difference = colored_magnet[:, :2] - ground_truth[:, :2]
        euclidean_norm = torch.linalg.norm(difference, dim=1)
        if torch.isnan(euclidean_norm).any():
            break
        euclidean_norms.append(euclidean_norm)
        for _ in range(int(0.02 / sim_dt)):
            forces, torques = magnet_entity.apply_force_torque_on_magnet(currents=current_set)
            positions = magnet_entity.get_magnets_position()
            positions[:, 2] = 0.003
            sim.step()
            sim_time += sim_dt
            count += 1
            # rigid_catheter.update(sim_dt)
            # sim_view.update_articulations_kinematic()
    euclidean_norms = torch.stack(euclidean_norms, dim=0)
    positional_error = torch.sum(torch.mean(euclidean_norms, dim=0))
    mean_magnet1, std_magnet1 = torch.mean(euclidean_norms[:, 0]), torch.std(euclidean_norms[:, 0])
    max_magnet1, min_magnet1 = torch.max(euclidean_norms[:, 0]), torch.min(euclidean_norms[:, 0])
    mean_magnet2, std_magnet2 = torch.mean(euclidean_norms[:, 1]), torch.std(euclidean_norms[:, 1])
    max_magnet2, min_magnet2 = torch.max(euclidean_norms[:, 1]), torch.min(euclidean_norms[:, 1])
    mean_magnet3, std_magnet3 = torch.mean(euclidean_norms[:, 2]), torch.std(euclidean_norms[:, 2])
    max_magnet3, min_magnet3 = torch.max(euclidean_norms[:, 2]), torch.min(euclidean_norms[:, 2])
    mean_magnet4, std_magnet4 = torch.mean(euclidean_norms[:, 3]), torch.std(euclidean_norms[:, 3])
    max_magnet4, min_magnet4 = torch.max(euclidean_norms[:, 3]), torch.min(euclidean_norms[:, 3])
    print(f"Mean positional error for magnet 1: {mean_magnet1:.6f} ± {std_magnet1:.6f}, max: {max_magnet1:.6f}, min: {min_magnet1:.6f}")
    print(f"Mean positional error for magnet 2: {mean_magnet2:.6f} ± {std_magnet2:.6f}, max: {max_magnet2:.6f}, min: {min_magnet2:.6f}")
    print(f"Mean positional error for magnet 3: {mean_magnet3:.6f} ± {std_magnet3:.6f}, max: {max_magnet3:.6f}, min: {min_magnet3:.6f}")
    print(f"Mean positional error for magnet 4: {mean_magnet4:.6f} ± {std_magnet4:.6f}, max: {max_magnet4:.6f}, min: {min_magnet4:.6f}")
    print(f"  & red & {mean_magnet1*1e3:.2f} ± {std_magnet1*1e3:.2f} & {max_magnet1*1e3:.2f} & {min_magnet1*1e3:.2f} \\\\")
    print(f"& & & blue & {mean_magnet2*1e3:.2f} ± {std_magnet2*1e3:.2f} & {max_magnet2*1e3:.2f} & {min_magnet2*1e3:.2f} \\\\")
    print(f"& & & yellow & {mean_magnet3*1e3:.2f} ± {std_magnet3*1e3:.2f} & {max_magnet3*1e3:.2f} & {min_magnet3*1e3:.2f} \\\\")
    print(f"& & & turquoise & {mean_magnet4*1e3:.2f} ± {std_magnet4*1e3:.2f} & {max_magnet4*1e3:.2f} & {min_magnet4*1e3:.2f} \\\\")
    print(f"Positional error: {positional_error:.6f} m")
    return positional_error
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device,
                                      dt=args_cli.sim_dt,
                                      render_interval=1)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view(eye=[-8, 0, 8], target=[0.0, 0.0, 2])
    sim.set_camera_view(eye=[0.0, 0.0, 0.1], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    result = run_simulator(sim, scene_entities)
    print("Result: ", result.item())
    return result.item()


if __name__ == "__main__":
    # run the main function
    result = main()
    if math.isnan(result):
        result = 0.05 + (torch.rand(1).item() -0.5) * 0.001  # Add some random value to avoid NaN
    if args_cli.output_file != 'None':
        # Write result to file if output_file is provided
        if args_cli.output_file:
            print(f"Writing result to {args_cli.output_file}")
            with open(args_cli.output_file, 'w') as f:
                f.write(str(result))
    # close sim app
    simulation_app.close()
