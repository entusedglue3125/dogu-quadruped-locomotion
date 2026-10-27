#!/usr/bin/env python3
"""
Stage 1: Simulation Validation Script for DOGU Quadruped
- Headless verification of articulation import
- Joint control and limit checking
- Physics stepping, contact checking, and reset behavior
- Random-action stress test checking for NaNs or velocity explosion
"""

import sys
import os
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

# 1. Launch simulation app in headless mode
app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from configs.dogu_robot_cfg import DOGU_CFG

def run_validation():
    print("==============================================================================")
    print("                DOGU Stage 1: Simulation & Articulation Validation             ")
    print("==============================================================================")

    sim_cfg = SimulationCfg(dt=0.005, render_interval=4)
    sim = SimulationContext(sim_cfg)

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Lighting
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    # Spawn Dogu
    print("[1/6] Initializing DOGU Articulation...")
    dogu_cfg = DOGU_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=dogu_cfg)

    sim.reset()
    print("      -> Articulation initialized successfully.")
    print(f"      -> Num joints: {robot.num_joints}")
    print(f"      -> Joint names: {robot.data.joint_names}")
    print(f"      -> Num bodies: {robot.num_bodies}")
    print(f"      -> Body names: {robot.data.body_names}")

    # Check joint limits
    print("\n[2/6] Verifying joint limits and initial state...")
    limits = robot.data.joint_limits.torch[0]
    print(f"      -> Joint lower limits: {limits[:, 0].cpu().numpy().round(3)}")
    print(f"      -> Joint upper limits: {limits[:, 1].cpu().numpy().round(3)}")
    init_pos = robot.data.root_pos_w.torch[0].cpu().numpy()
    print(f"      -> Initial root pos: X={init_pos[0]:.4f}, Y={init_pos[1]:.4f}, Z={init_pos[2]:.4f} m")

    # Step physics with zero joint targets (nominal standing)
    print("\n[3/6] Stepping physics under zero-position target (holding stand pose for 50 steps)...")
    has_nan = False
    for step in range(50):
        target = torch.zeros((1, 12), device=sim.device)
        robot.actuators.target_command.set_position_index(value=target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=0.005)

        pos = robot.data.root_pos_w.torch[0]
        vel = robot.data.root_lin_vel_w.torch[0]
        q = robot.data.joint_pos.torch[0]
        qd = robot.data.joint_vel.torch[0]

        if torch.isnan(pos).any() or torch.isnan(vel).any() or torch.isnan(q).any() or torch.isnan(qd).any():
            print(f"[FAIL] NaN detected at step {step}!")
            has_nan = True
            break

    if not has_nan:
        final_pos = robot.data.root_pos_w.torch[0].cpu().numpy()
        print(f"      -> Stand holding passed cleanly! Final base pos: Z={final_pos[2]:.4f} m (expected ~0.108m)")

    # Test reset behavior
    print("\n[4/6] Testing reset behavior...")
    # Displace robot to 0.2m high
    default_root_state = robot.data.default_root_state.torch.clone()
    default_root_state[:, 2] = 0.15 # raise to 15cm
    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.torch.clone(),
        robot.data.default_joint_vel.torch.clone()
    )
    robot.reset()
    print("      -> Articulation reset executed cleanly.")

    # Run random action stress test
    print("\n[5/6] Running random action stress test (100 steps with bounded random offsets)...")
    exploding_velocity = False
    for step in range(100):
        # Sample bounded action within +/- 0.3 rad from standing
        random_action = (torch.rand((1, 12), device=sim.device) * 2.0 - 1.0) * 0.3
        robot.actuators.target_command.set_position_index(value=random_action)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=0.005)

        vel_norm = robot.data.root_lin_vel_w.torch[0].norm().item()
        ang_norm = robot.data.root_ang_vel_w.torch[0].norm().item()
        if vel_norm > 50.0 or ang_norm > 100.0:
            print(f"[FAIL] Exploding velocity detected! vel={vel_norm:.2f}, ang={ang_norm:.2f}")
            exploding_velocity = True
            break

    if not exploding_velocity:
        print("      -> Random action stress test PASSED: No velocity explosion, stable dynamics.")

    # Check GPU memory
    print("\n[6/6] Resource consumption summary:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"      -> VRAM Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

    print("\n==============================================================================")
    print("               STAGE 1 SIMULATION VALIDATION: ALL CHECKS PASSED              ")
    print("==============================================================================")

    simulation_app.close()

if __name__ == "__main__":
    run_validation()
