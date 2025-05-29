import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporterCfg

# CUBE_ARTICULATION_CFG = ArticulationCfg(
#         prim_path="/World/envs/env_0/Cube",

CUBE_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_0/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.1
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
    )

MAZE_CUBE_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_0/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.1
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
    )

GOAL_CFG =VisualizationMarkersCfg(
        prim_path="/World/Goal/Cones",
        markers={
            "cone_0": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )

MAGNET_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_0/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0005),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.1
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
    )    

MAGNET_GOAL_CFG =VisualizationMarkersCfg(
        prim_path="/World/Goal/Cones",
        markers={
            "cone_0": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )

TERRAIN_CFG = TerrainImporterCfg(
        prim_path="/World/envs/env_0/Terrain",
        terrain_type="usd",
        usd_path="/home/pascal/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cube/Calibration_data/maze.usd",
        env_spacing=4.0,
)

TERRAIN_HARD_CFG = TerrainImporterCfg(
        prim_path="/World/envs/env_0/Terrain",
        terrain_type="usd",
        usd_path="/home/pascal/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cube/Calibration_data/maze_hard.usd",
        env_spacing=8.0,
)