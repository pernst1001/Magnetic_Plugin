import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import omni
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim import SimulationContext
from Magnetic_Plugin.magnetic_entity import MagneticEntity
from pxr import PhysxSchema  # import PhysX USD schema classes



def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.005, 0.005, 0.005),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # cube_cfg = RigidObjectCfg(
    #     prim_path="/World/RigidCube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.001, 0.001, 0.001),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
    #         physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0005), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
    # )
    # cube_object = RigidObject(cfg=cube_cfg)

    # Define and spawn the deformable (soft) cuboid
    soft_cfg = DeformableObjectCfg(
        prim_path="/World/SoftCube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.02, 0.02, 0.05),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001, solver_position_iteration_count=10),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1e-3),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0105, 0.05))  # center 1.35m above ground (1.35 = 1.0 + 0.25 + 0.10)
    )
    soft_cuboid = DeformableObject(soft_cfg)  # create the deformable cuboid in the scene
    # scene_entities = {"rigid_cube": cube_object, "soft_cube": soft_cuboid}
    # scene_entities = {"rigid_cube": cube_object}
    scene_entities = {"soft_cube": soft_cuboid}
    
    return scene_entities

def create_attachement():
    '''Create a magnetic attachement'''
    # Create a PhysX attachment to rigidly bind the soft cuboid to the cube
    stage = omni.usd.get_context().get_stage()
    soft_prim = stage.GetPrimAtPath("/World/SoftCube/geometry/mesh")
    rigid_prim = stage.GetPrimAtPath("/World/RigidCube/geometry/mesh")
    attachment_path = soft_prim.GetPath().AppendElementString("attachment")  # child prim path for attachment
    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
    attachment.GetActor0Rel().SetTargets([soft_prim.GetPath()])   # actor0: soft body prim
    attachment.GetActor1Rel().SetTargets([rigid_prim.GetPath()])  # actor1: rigid body prim
    PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())  # enable auto-attachment (binds soft to rigid)&#8203;:contentReference[oaicite:4]{index=4}


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    # create_attachement()
    # rigid_cube = entities["rigid_cube"]
    soft_cube = entities["soft_cube"]
    # magnet = MagneticEntity(volume=0.001**2, remanence=1.32, direction=[0.0, 0.0, 1.0], magnet=rigid_cube)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    field = torch.tensor([0.0, 0.0, 0.01])
    force = torch.tensor([1.0, 0.0, 0.0])*1e-3
    # Simulate physics
    while simulation_app.is_running():
        if count < 5:
            input("Press Enter to continue...")
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        # rigid_cube.update(sim_dt)
        soft_cube.update(sim_dt)
        # input("Press Enter to continue...")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view(eye=[-8, 0, 8], target=[0.0, 0.0, 2])
    sim.set_camera_view(eye=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
