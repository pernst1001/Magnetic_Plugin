import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
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
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

def design_scene():
    """Designs the scene."""
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    needle_asset_path = "/home/pascal/IsaacLab/MagnetSim/Calibration_data/isaac_sim_deformable.usd"
    needle_asset = RigidObjectCfg(
        prim_path="/World/needle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=needle_asset_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0, 0)),
    )
    needle = RigidObject(needle_asset)

    # return the scene information
    scene_entities = {"cube_object": needle}
    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    # cube_object = entities["cube_object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies

    # Simulate physics
    while simulation_app.is_running():
        # Step the simulation
        sim.step()
        sim_time += sim_dt
        count += 1
        if count % 100 == 0:
            print(f"Time: {sim_time:.2f}")
        sim.render()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device,
                                      dt=1/1000,
                                      render_interval=2)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.3, 0.0, 0.2], target=[0.0, 0.0, 0.0])
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
