# Copyright (c) 2026, DOGU Locomotion Project.
"""Configurable reward settings for DOGU Locomotion & Standing tasks."""

from dataclasses import dataclass
import math

@dataclass
class StandingRewardsConfig:
    """Weights and parameters for Stage 2: Standing Task."""
    # Stability & Height
    base_height_l2_weight: float = -15.0       # Penalize deviation from target height
    target_height: float = 0.108              # Target standing height in meters
    flat_orientation_l2_weight: float = -5.0   # Penalize pitch & roll
    lin_vel_z_l2_weight: float = -1.5          # Penalize vertical bouncing
    ang_vel_xy_l2_weight: float = -0.1         # Penalize body angular rocking

    # Zero velocity tracking (stand still)
    track_lin_vel_xy_exp_weight: float = 1.0   # Reward keeping zero XY velocity
    track_ang_vel_z_exp_weight: float = 0.5    # Reward keeping zero yaw velocity
    vel_error_std: float = math.sqrt(0.25)     # Kernel bandwidth for velocity tracking

    # Action & Effort quality
    dof_torques_l2_weight: float = -1.0e-4     # Penalize high actuator effort
    dof_acc_l2_weight: float = -2.5e-7         # Penalize high joint accelerations
    action_rate_l2_weight: float = -0.02       # Penalize rapid action changes (smooth control)
    joint_pos_limits_weight: float = -5.0      # Penalize getting close to joint limits

    # Contact penalties
    undesired_contacts_weight: float = -5.0    # Penalize base or thigh touching ground
    undesired_contact_threshold: float = 0.5   # Contact force threshold in Newtons


@dataclass
class LocomotionRewardsConfig:
    """Weights and parameters for Stage 3: Forward Walking Task."""
    # Velocity tracking
    track_lin_vel_xy_exp_weight: float = 2.0   # High reward for forward velocity tracking
    track_ang_vel_z_exp_weight: float = 0.75   # Reward for yaw command tracking
    vel_error_std: float = math.sqrt(0.25)     # Exponential kernel std

    # Base stability
    base_height_l2_weight: float = -10.0      # Maintain height during gait
    target_height: float = 0.108              # Target height
    flat_orientation_l2_weight: float = -3.5  # Upright torso orientation
    lin_vel_z_l2_weight: float = -1.5         # Penalize vertical bouncing
    ang_vel_xy_l2_weight: float = -0.05       # Penalize roll/pitch angular velocity

    # Stepping & Foot contacts
    feet_air_time_weight: float = 0.5         # Encourage stepping with positive air time
    feet_air_time_threshold: float = 0.15     # Minimum step swing time threshold for Dogu (0.15s)
    feet_slide_weight: float = -0.1           # Penalize foot sliding during stance

    # Action & Energy quality
    dof_torques_l2_weight: float = -5.0e-5    # Energy efficiency
    dof_acc_l2_weight: float = -2.5e-7        # Smooth acceleration
    action_rate_l2_weight: float = -0.01      # Smooth actions
    joint_pos_limits_weight: float = -2.0     # Stay inside joint bounds

    # Contacts
    undesired_contacts_weight: float = -5.0   # Penalize belly or thigh ground contacts
    undesired_contact_threshold: float = 0.5
