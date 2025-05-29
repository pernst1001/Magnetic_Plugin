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
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim import SimulationContext
from Magnetic_Plugin.magnetic_entity import MagneticEntity
from isaaclab.utils.math import quat_apply


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

    scene_entities = {}

    distance = 0.001
    z_distance = 0.0005
    origins = [[distance, distance, z_distance], [-distance, distance, z_distance], [distance, -distance, z_distance], [-distance, -distance, z_distance]]
    for i, origin in enumerate(origins):
        cube_cfg = RigidObjectCfg(
            prim_path=f"/World/Cube{i}",
            spawn=sim_utils.CuboidCfg(
                size=(0.001, 0.001, 0.001),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=origin, rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
        )
        cube_object = RigidObject(cfg=cube_cfg)
        scene_entities.update({f"cube_{i}": cube_object})
    
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cube1 = entities["cube_1"]
    cube2 = entities["cube_2"]
    cube3 = entities["cube_3"]
    cube4 = entities["cube_0"]
    cubes = [cube1, cube2, cube3, cube4]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    field = torch.tensor([0.001, 0.0, 0.0])
    force = torch.tensor([0.0, 0.0, 0.0])
    magnetic_cube_1 = MagneticEntity(volume=0.001**3, remanence=1.27, direction=torch.tensor([0.0, 0.0, 1.0]), magnet=cube1)
    magnetic_cube_2 = MagneticEntity(volume=0.001**3, remanence=1.27, direction=torch.tensor([0.0, 0.0, 1.0]), magnet=cube2)
    magnetic_cube_3 = MagneticEntity(volume=0.001**3, remanence=1.27, direction=torch.tensor([0.0, 0.0, 1.0]), magnet=cube3)
    magnetic_cube_4 = MagneticEntity(volume=0.001**3, remanence=1.27, direction=torch.tensor([0.0, 0.0, 1.0]), magnet=cube4)
    magnetic_cubes = [magnetic_cube_1, magnetic_cube_2, magnetic_cube_3, magnetic_cube_4]

    for i, cube in enumerate(cubes):
        magnetic_cube_i = MagneticEntity(volume=0.001**3, remanence=1.27, direction=torch.tensor([0.0, 0.0, 1.0]), magnet=cube)
    while simulation_app.is_running():
        if count == 0:
            input("Press Enter to continue...")
        currents = magnetic_cube_1.get_currents_from_field_gradient3(field, force)
        for magnet in magnetic_cubes:
            magnet.apply_force_torque_on_magnet(currents)
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for cube in cubes: cube.update(sim_dt)
        # input("Press Enter to continue...")

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    sim.set_camera_view(eye=[0.0, -0.0, 0.05], target=[0.0, 0.0, 0.0])
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
