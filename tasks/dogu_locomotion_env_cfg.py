# Copyright (c) 2026, DOGU Locomotion Project.
"""Stage 3: Forward Locomotion Task Environment Configuration for DOGU."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.core.velocity.mdp.rewards import feet_air_time
from configs.dogu_robot_cfg import DOGU_CFG
from configs.rewards_cfg import LocomotionRewardsConfig
from tasks.dogu_standing_env_cfg import DoguSceneCfg, ObservationsCfg, ActionsCfg, TerminationsCfg

rew_cfg = LocomotionRewardsConfig()

##
# Locomotion Commands (Forward walking curriculum)
##

@configclass
class LocomotionCommandsCfg:
    """Command specifications for forward walking locomotion."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 10.0),
        rel_standing_envs=0.05, # 5% standing envs to maintain stance capability
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.15, 0.40), # Slow, stable forward velocity (0.15 to 0.40 m/s)
            lin_vel_y=(-0.05, 0.05), # Minimal lateral drift
            ang_vel_z=(-0.2, 0.2),   # Gentle turning
            heading=(-0.5, 0.5),
        ),
    )


@configclass
class LocomotionRewardsCfg:
    """Reward terms for Stage 3 Forward Locomotion task."""

    # 1. Command tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=rew_cfg.track_lin_vel_xy_exp_weight,
        params={"command_name": "base_velocity", "std": rew_cfg.vel_error_std},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=rew_cfg.track_ang_vel_z_exp_weight,
        params={"command_name": "base_velocity", "std": rew_cfg.vel_error_std},
    )

    # 2. Base stability & posture
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=rew_cfg.base_height_l2_weight,
        params={"target_height": rew_cfg.target_height},
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=rew_cfg.flat_orientation_l2_weight,
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=rew_cfg.lin_vel_z_l2_weight,
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=rew_cfg.ang_vel_xy_l2_weight,
    )

    # 3. Leg motion & stepping
    feet_air_time = RewTerm(
        func=feet_air_time,
        weight=rew_cfg.feet_air_time_weight,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("feet_contact"),
            "threshold": rew_cfg.feet_air_time_threshold,
        },
    )

    # 4. Action smoothness & energy efficiency
    dof_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=rew_cfg.dof_torques_l2_weight,
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=rew_cfg.action_rate_l2_weight,
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=rew_cfg.joint_pos_limits_weight,
    )

    # 5. Penalize body contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=rew_cfg.undesired_contacts_weight,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact"),
            "threshold": rew_cfg.undesired_contact_threshold,
        },
    )


@configclass
class LocomotionEventsCfg:
    """Randomization events for locomotion training."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.85, 1.15),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class DoguLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Complete environment configuration for Stage 3 Locomotion task."""

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
    )

    scene: DoguSceneCfg = DoguSceneCfg(num_envs=16, env_spacing=2.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: LocomotionCommandsCfg = LocomotionCommandsCfg()
    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: LocomotionEventsCfg = LocomotionEventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        if self.scene.feet_contact is not None:
            self.scene.feet_contact.update_period = self.sim.dt
        if self.scene.body_contact is not None:
            self.scene.body_contact.update_period = self.sim.dt
