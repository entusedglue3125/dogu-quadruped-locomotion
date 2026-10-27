# Copyright (c) 2026, DOGU Locomotion Project.
"""Configuration for DOGU quadruped robot asset in Isaac Lab."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import os

USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "dogu", "usd", "dogu", "dogu.usda"
)

DOGU_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.108), # nominal standing height is ~0.108m
        rot=(0.0, 0.0, 0.0, 1.0), # Quaternion in (x, y, z, w) order
        joint_pos={
            "fl_hip_joint": 0.0,
            "fr_hip_joint": 0.0,
            "bl_hip_joint": 0.0,
            "br_hip_joint": 0.0,
            "fl_femur_joint": 0.0,
            "fr_femur_joint": 0.0,
            "bl_femur_joint": 0.0,
            "br_femur_joint": 0.0,
            "fl_tibia_joint": 0.0,
            "fr_tibia_joint": 0.0,
            "bl_tibia_joint": 0.0,
            "br_tibia_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=40.0,
            damping=2.0,
            effort_limit=3.0,
            velocity_limit=10.0,
        ),
    },
)
