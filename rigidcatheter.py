import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

RIGID_CATHETER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/pascal/Downloads/My_Joints/RigidCatheter.usd", #TODO: Add the usd path to the repo
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0.0013)),
    actuators={
        "spherical_joint": ImplicitActuatorCfg(
            joint_names_expr=[r"SphericalJoint[1-9]:\d", r"SphericalJoint1[0-1]:\d"],
            stiffness=0.0,
            damping=0.0,
        )
    }
)
RIGID_CATHETER_SIM2REAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/pascal/Downloads/My_Joints/RigidCatheterSim2Real.usd", #TODO: Add the usd path to the repo
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            sleep_threshold=0.0,
            stabilization_threshold=1e-6,
            linear_damping=0.0,
            angular_damping=0.0,
            # enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=1e-6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(),
    actuators={
        "spherical_joint": ImplicitActuatorCfg(
            joint_names_expr=[r"SphericalJoint[1-9]:\d", r"SphericalJoint1[0-1]:\d"],
            stiffness=0.0,
            damping=0.0,
        )
    }
)