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
from isaaclab.sim.converters.mesh_converter import MeshConverter
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.converters.urdf_converter import UrdfConverter
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
import isaaclab.sim as sim_utils
import os
import glob


# folder = "/home/pascal/Downloads/Joints_Alex"
# files = glob.glob(os.path.join(folder, "2-6mm*.stl"))
# for file in files:
#     cfg = MeshConverterCfg(
#         asset_path=file,
#         force_usd_conversion=True,
#         make_instanceable=True,
#         usd_dir=os.path.join(folder, os.path.basename(file).replace(".stl", "")),
#         usd_file_name=os.path.basename(file).replace(".stl", ".usd"),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
#         collision_props=sim_utils.CollisionPropertiesCfg(),
#         scale=(0.001, 0.001, 0.001),
#     )
#     converter = MeshConverter(cfg=cfg)
# files = glob.glob(os.path.join(folder, "12mm*.stl"))
# for file in files:
#     cfg = MeshConverterCfg(
#         asset_path=file,
#         force_usd_conversion=True,
#         make_instanceable=True,
#         usd_dir=os.path.join(folder, os.path.basename(file).replace(".stl", "")),
#         usd_file_name=os.path.basename(file).replace(".stl", ".usd"),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
#         collision_props=sim_utils.CollisionPropertiesCfg(),
#         scale=(0.001, 0.001, 0.001),
#     )
#     converter = MeshConverter(cfg=cfg)
file = "/home/pascal/Downloads/My_Joints/Magnet.stl"
cfg = MeshConverterCfg(
    asset_path=file,
    force_usd_conversion=True,
    make_instanceable=True,
    usd_dir=os.path.join(os.path.dirname(file), os.path.basename(file).replace(".stl", "")),
    usd_file_name=os.path.basename(file).replace(".stl", ".usd"),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.0317e-3),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    scale=(0.001, 0.001, 0.001),
)
MeshConverter(cfg=cfg)
# folder = "/home/pascal/Downloads/STL_RCATHETER"
# files = glob.glob(os.path.join(folder, "MeshBody*.stl"))
# output_folder = os.path.join(folder, "Catheter")

# for file in files:
#     cfg = MeshConverterCfg(
#         asset_path=file,
#         force_usd_conversion=True,
#         make_instanceable=True,
#         usd_dir=os.path.join(folder, os.path.basename(file).replace(".stl", "")),
#         usd_file_name=os.path.basename(file).replace(".stl", ".usd"),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
#         collision_props=sim_utils.CollisionPropertiesCfg(),
#         scale=(0.001, 0.001, 0.001),
#     )
#     converter = MeshConverter(cfg=cfg)


