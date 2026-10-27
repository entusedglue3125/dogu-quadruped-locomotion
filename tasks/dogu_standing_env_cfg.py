# Copyright (c) 2026, DOGU Locomotion Project.
"""Stage 2: Standing Task Environment Configuration for DOGU."""

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
from configs.dogu_robot_cfg import DOGU_CFG
from configs.rewards_cfg import StandingRewardsConfig

rew_cfg = StandingRewardsConfig()

##
# Scene definition
##

@configclass
class DoguSceneCfg(InteractiveSceneCfg):
    """Scene definition with flat ground and DOGU robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = DOGU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Foot contact sensor
    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*_tibia_link",
        history_length=3,
        track_air_time=True,
    )

    # Base and thigh contact sensor for detecting falls/undesired ground contacts
    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        history_length=3,
        track_air_time=False,
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP Settings
##

@configclass
class StandingCommandsCfg:
    """Command specifications: zero commanded velocity to hold stance."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0, # 100% standing envs (target 0 vel)
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications: normalized joint position targets with conservative scaling."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.20, # +/- 0.20 rad maximum offset per action step
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for policy group (48 dimensions)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Base kinematics
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # Joint state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StandingRewardsCfg:
    """Reward terms for Stage 2 Standing task."""

    # Height and posture stability
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

    # Zero velocity tracking
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

    # Action quality and energy
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

    # Ground contact penalties
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=rew_cfg.undesired_contacts_weight,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact"),
            "threshold": rew_cfg.undesired_contact_threshold,
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the MDP."""

    # Max episode duration: 20 seconds = 1000 decimation steps
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Fall detection: base drops below 5cm (half standing height)
    base_height_below_min = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.055},
    )

    # Tilt detection: body rolls or pitches more than 45 degrees
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.80}, # ~46 degrees
    )

    # Body ground collision
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact"),
            "threshold": 0.5,
        },
    )


@configclass
class EventsCfg:
    """Randomization events at reset and interval."""

    # Startup physics material
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

    # Reset root state
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "yaw": (-0.1, 0.1),
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

    # Reset joints to near nominal standing
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2), # slight perturbation around default (0.0)
            "velocity_range": (0.0, 0.0),
        },
    )

    # Gentle perturbation push during standing to build balance robustness
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
    )


@configclass
class DoguStandingEnvCfg(ManagerBasedRLEnvCfg):
    """Complete environment configuration for Stage 2 Standing task."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, # 200 Hz physics
        render_interval=4,
    )

    # Scene settings (16 parallel envs for laptop 6GB VRAM safety)
    scene: DoguSceneCfg = DoguSceneCfg(num_envs=16, env_spacing=1.5)

    # Components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: StandingCommandsCfg = StandingCommandsCfg()
    rewards: StandingRewardsCfg = StandingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization settings."""
        self.decimation = 4 # 50 Hz control loop
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        if self.scene.feet_contact is not None:
            self.scene.feet_contact.update_period = self.sim.dt
        if self.scene.body_contact is not None:
            self.scene.body_contact.update_period = self.sim.dt
