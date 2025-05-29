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

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
import omni.physx.scripts.utils as physx_utils
from pxr import PhysxSchema, UsdPhysics
import omni

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    connector = RigidObjectCfg(
        prim_path="/World/Connector",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/pascal/Downloads/Joint_model/joint_model.usd"
        ))
    connector = RigidObject(connector)

    magnet = RigidObjectCfg(
        prim_path="/World/Magnets/Magnet1",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/pascal/Downloads/Magnet_model/magnet_model.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ))
    magnet = RigidObject(magnet)
    stage = omni.usd.get_context().get_stage()
    articulation_root = stage.GetPrimAtPath("/World/Magnets/Magnet1")
    UsdPhysics.ArticulationRootAPI.Apply(articulation_root)
    PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)

    scene_entities = {"magnet": magnet, "connector": connector}
    
    return scene_entities
   
def create_joint(prim1: str, prim2: str, joint_name: str = "Fixed"):
        stage = omni.usd.get_context().get_stage()
        prim1 = stage.GetPrimAtPath(prim1)
        prim2 = stage.GetPrimAtPath(prim2)
        joint = physx_utils.createJoint(stage=stage, joint_type=joint_name, from_prim=prim1, to_prim=prim2)
        # if joint_name == "Spherical":
        #     write_joint_friction_coefficient_to_sim

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Define simulation stepping
    create_joint("/World/Magnets/Magnet1/geometry", "/World/Connector/geometry", joint_name="Fixed")
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
        # Simulate physics
    while simulation_app.is_running():
        # input("Press Enter to start the simulation...")
        sim_time += sim_dt
        sim.step()
        # Update the cube


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    sim.set_camera_view(eye=[0.0, -0.05, 0.05], target=[0.0, 0.0, 0.0])
    # sim.set_camera_view(eye=[0.0, -0.5, 0.5], target=[0.0, 0.0, 0.0])
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