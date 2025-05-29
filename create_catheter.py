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

import omni.physx
import torch

import isaacsim.core.utils.prims as prim_utils
import omni
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim import SimulationContext, PhysxCfg
from Magnetic_Plugin.magnetic_entity import MagneticEntity
from pxr import PhysxSchema  # import PhysX USD schema classes



def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    cylinder = RigidObjectCfg(
        prim_path="/World/Magnets/Magnet1",
        spawn=sim_utils.CylinderCfg(
            radius=0.00075,
            height=0.015,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0002,
                rest_offset=0.0001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002), rot=(0.70711, 0.0, 0.70711, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
    )
    magnet_1 = RigidObject(cfg=cylinder)

    cylinder = RigidObjectCfg(
        prim_path="/World/Magnets/Magnet2",
        spawn=sim_utils.CylinderCfg(
            radius=0.00075,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0375, 0.0, 0.001), rot=(0.70711, 0.0, 0.70711, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
    )
    magnet_2 = RigidObject(cfg=cylinder)

    cylinder = RigidObjectCfg(
        prim_path="/World/Magnets/Magnet3",
        spawn=sim_utils.CylinderCfg(
            radius=0.00075,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0675, 0.0, 0.001), rot=(0.70711, 0.0, 0.70711, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
    )
    # magnet_3 = RigidObject(cfg=cylinder)
    
    # Define and spawn the deformable (soft) cuboid
    soft_cylinder = DeformableObjectCfg(
        prim_path="/World/Deformable/Connection1",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.0005,
            height=0.025,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0003, contact_offset=0.0002, simulation_hexahedral_resolution=8, solver_position_iteration_count=50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.45, youngs_modulus=40e6),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.02, 0.0, 0.002), rot=(0.70711, 0.0, 0.70711, 0.0)),
    )
    connection1 = DeformableObject(soft_cylinder)  # create the deformable cuboid in the scene

    soft_cylinder = DeformableObjectCfg(
        prim_path="/World/Deformable/Connection2",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.0005,
            height=0.02,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0009, contact_offset=0.001, simulation_hexahedral_resolution=8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.35, youngs_modulus=40e6, elasticity_damping=0.01, damping_scale=1.0),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.1e-3),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0525, 0.0, 0.001), rot=(0.70711, 0.0, 0.70711, 0.0)),
    )
    # connection2 = DeformableObject(soft_cylinder)  # create the deformable cuboid in the scene

    soft_cylinder = DeformableObjectCfg(
        prim_path="/World/Deformable/Connection3",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.0005,
            height=0.03,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0004, contact_offset=0.0005, simulation_hexahedral_resolution=10),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.35, youngs_modulus=40e6, elasticity_damping=0.01, damping_scale=1.0),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.1e-3),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0875, 0.0, 0.001), rot=(0.70711, 0.0, 0.70711, 0.0)),
    )
    # connection3 = DeformableObject(soft_cylinder)  # create the deformable cuboid in the scene
    # scene_entities = {"magnet1": magnet_1, "magnet2": magnet_2, "magnet3": magnet_3, "connection1": connection1, "connection2": connection2, "connection3": connection3}
    # scene_entities = {"connection1": connection1}
    scene_entities = {"connection1": connection1, "magnet1": magnet_1}
    return scene_entities

def create_attachement(prim1: str, prim2: str):
    '''Create a magnetic attachement'''
    # Create a PhysX attachment to rigidly bind the soft cuboid to the cube
    stage = omni.usd.get_context().get_stage()
    soft_prim = stage.GetPrimAtPath(prim1)
    rigid_prim = stage.GetPrimAtPath(prim2)
    attachment_path = soft_prim.GetPath().AppendElementString("attachment")  # child prim path for attachment
    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
    attachment.GetActor0Rel().SetTargets([soft_prim.GetPath()])   # actor0: soft body prim
    attachment.GetActor1Rel().SetTargets([rigid_prim.GetPath()])  # actor1: rigid body prim
    # Apply auto attachment API first
    auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    # Then create the attribute on the API instance
    # auto_attachment_api.CreateDeformableVertexOverlapOffsetAttr(0.002, False)
    return auto_attachment_api


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    attachement_api = create_attachement("/World/Magnets/Magnet1/geometry/mesh", "/World/Deformable/Connection1/geometry/mesh")
    attachement_api_2 = create_attachement("/World/Magnets/Magnet2/geometry/mesh", "/World/Deformable/Connection1/geometry/mesh")
    magnet1 = entities["magnet1"]
    connection1 = entities["connection1"]
    # magnet2 = entities["magnet2"]
    # create_attachement("/World/Magnets/Magnet2/geometry/mesh", "/World/Deformable/Connection2/geometry/mesh")
    # create_attachement("/World/Deformable/Connection2/geometry/mesh", "/World/Magnets/Magnet3/geometry/mesh")
    # create_attachement("/World/Magnets/Magnet3/geometry/mesh", "/World/Deformable/Connection3/geometry/mesh")
    # Get simulation time-step
    sim_dt = sim.get_physics_dt()
    rendering_dt = sim.get_rendering_dt()
    print(f"[INFO]: Simulation time-step: {sim_dt}, Rendering time-step: {rendering_dt}")
    sim_time = 0.0
    count = 0
    # Simulate physics
    print("[INFO]: Starting simulation...")
    while True:
        # perform step     
        if count == 1:
            attachement_api.CreateDeformableVertexOverlapOffsetAttr(0.005, False)
            attachement_api_2.CreateDeformableVertexOverlapOffsetAttr(0.005, False)
        #     pass
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        magnet1.update(sim_dt)
        connection1.update(sim_dt)
        # magnet1.update(sim_dt)
        # update buffers
        # input("Press Enter to continue...")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, 
                                      dt=1 / 1000, 
                                      render_interval=2,
                                      physx=PhysxCfg(bounce_threshold_velocity=0.0,
                                                     enable_stabilization=False)
                                      )
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view(eye=[-8, 0, 8], target=[0.0, 0.0, 2])
    sim.set_camera_view(eye=[0.0, -0.05, 0.2], target=[0.0, 0.0, 0.01])
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
