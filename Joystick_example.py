import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on controlling a rigid object over joystick.")
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
import pygame
import numpy as np
import threading

class JoystickController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Joystick Controller")
        self.clock = pygame.time.Clock()
        self.center = (200, 200)
        self.joystick_pos = self.center
        self.direction = np.array([0, 0])
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.joystick_pos = event.pos
                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:
                        self.joystick_pos = event.pos
                        self.direction = np.array([self.joystick_pos[0] - self.center[0], self.joystick_pos[1] - self.center[1]])
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.joystick_pos = self.center
                    self.direction = np.array([0, 0])

            self.screen.fill((255, 255, 255))
            pygame.draw.circle(self.screen, (0, 0, 0), self.center, 50, 2)
            pygame.draw.circle(self.screen, (0, 0, 255), self.joystick_pos, 20)
            self.clock.tick(60)

        pygame.quit()

    def get_direction(self):
        if np.linalg.norm(self.direction) == 0:
            return np.array([0, 0])
        return self.direction / np.linalg.norm(self.direction)

def run_gui(controller):
    controller.run()

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
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    # cube_cfg = RigidObjectCfg(
    #     prim_path="/World/Cube",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.0005,
    #         height=0.001,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
    #         physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(rot=(0.70711, 0.0, 0.70711, 0.0)),
    # )
    cube_object = RigidObject(cfg=cube_cfg)
    scene_entities = {"cube": cube_object}
    
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cube = entities["cube"]
    # my_visualizer = define_markers()
    # Define simulation stepping
    sim.set_simulation_dt(physics_dt=0.001)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    magnetic_cube = MagneticEntity(volume=0.001**3, remanence=1.32, direction=[0.0, 0.0, 1.0], magnet=cube)
    # force = torch.tensor([0.0019, 0.0019, 0.0], device=cube.device)
    field = torch.tensor([0.0e-3, 0.0e-3, 0.0e-3], device=cube.device)
    root_pose = cube.data.root_state_w
    root_pose[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=cube.device)
    cube.write_root_state_to_sim(root_state=root_pose)
    cube.write_data_to_sim()
    controller = JoystickController()
    gui_thread = threading.Thread(target=run_gui, args=(controller,))
    gui_thread.start()
    input("Press Enter to continue...")
    
    # cube.write_root_pose_to_sim()
    # Simulate physics
    while simulation_app.is_running():
        direction = controller.get_direction()
        force = torch.tensor([direction[0], -direction[1], 0.0], device=cube.device)*0.02
        print(f"force: {force}")
        # my_visualizer.visualize(translations=magnet_position, orientations=current_dipole, marker_indices=torch.tensor([1]))
        currents = magnetic_cube.get_currents_from_field_gradient3(field=field, gradient=force)
        magnetic_cube.apply_force_torque_on_magnet(currents=currents)
        # print(f"position: {magnetic_cube.get_magnets_position()}")
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        # update buffers
        cube.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
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
