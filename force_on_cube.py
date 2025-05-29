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
# from Magnetic_Plugin.magnetic_entity import MagneticEntity
from isaaclab.magnetic import MagneticEntity
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

    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
    )
    cube_object = RigidObject(cfg=cube_cfg)
    scene_entities = {"cube": cube_object}
    
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cube = entities["cube"]
    my_visualizer = define_markers()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    magnetic_cube = MagneticEntity(volume=0.001**3, remanence=1.32, direction=[1.0, 0.0, 0.0], magnet=cube)
    force = torch.tensor([0.0017, 0.001, 0.0], device=cube.device)
    field = torch.tensor([30.0e-3, 10.0e-3, 10.0e-3], device=cube.device)
    initial_state = cube.data.root_state_w
    print(f"Initial state: {initial_state}")
    initial_state[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=cube.device)
    initial_state[:, 7:] = torch.zeros_like(initial_state[:, 7:])
    print(f"Initial state: {initial_state}")
    cube.write_root_state_to_sim(initial_state)
    print(f"Initial state: {cube.data.root_state_w}")
    cube.update(dt=sim_dt)
        # Simulate physics
    while simulation_app.is_running():
        position = magnetic_cube.get_magnets_position()
        print(f"Current position: {position}")
        # current_dipole = magnetic_cube.get_current_dipole_moment()
        my_visualizer.visualize(translations=position, orientations=magnetic_cube.get_quad())
        # if count % 2 == 0:
        #     my_visualizer.visualize(translations=position, orientations=magnetic_cube.get_quad())
        # else:
        #     my_visualizer.visualize(translations=position, orientations=magnetic_cube.get_quad())
        # print(f"Current dipole moment: {quat_apply(magnetic_cube.get_quad(), current_dipole)}")
        # print(f"Scipy Current dipole moment: {quat_apply(magnetic_cube.get_quad(), current_dipole)}")

        currents = magnetic_cube.get_currents_from_field_gradient3(field=field, gradient=force)
        magnetic_cube.apply_force_torque_on_magnet(currents=currents)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cube.update(sim_dt)
        # input("Press Enter to continue...")
        if count % 2 == 0:
            field = torch.roll(field, 1)

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
