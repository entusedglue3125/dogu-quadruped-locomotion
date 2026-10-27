#!/usr/bin/env python3
"""
Export trajectory data (base position, orientation, and 12 joint angles)
from trained DOGU policy checkpoints for interactive 3D web visualization.
"""

import argparse
import json
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Export DOGU trajectory to JSON")
    parser.add_argument("--task", type=str, default="locomotion", choices=["standing", "locomotion"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from tasks.dogu_standing_env_cfg import DoguStandingEnvCfg
    from tasks.dogu_locomotion_env_cfg import DoguLocomotionEnvCfg
    from configs.rsl_rl_ppo_cfg import DoguStandingPPORunnerCfg, DoguLocomotionPPORunnerCfg

    if args.task == "standing":
        env_cfg = DoguStandingEnvCfg()
        agent_cfg = DoguStandingPPORunnerCfg()
    else:
        env_cfg = DoguLocomotionEnvCfg()
        agent_cfg = DoguLocomotionPPORunnerCfg()

    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    agent_cfg.seed = 42
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.4.1")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=None,
        device=str(env.sim.device),
    )
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=str(env.sim.device))

    obs = wrapped_env.get_observations()
    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)

    trajectory = []

    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = wrapped_env.step(actions)

        pos = robot.data.root_pos_w.torch[0].cpu().numpy().tolist()
        quat = robot.data.root_quat_w.torch[0].cpu().numpy().tolist()
        joints = robot.data.joint_pos.torch[0].cpu().numpy().tolist()
        body_pos = robot.data.body_pos_w.torch[0].cpu().numpy().tolist()

        trajectory.append({
            "step": step,
            "time": round(step * 0.02, 3),
            "base_pos": [round(float(x), 4) for x in pos],
            "base_quat": [round(float(x), 4) for x in quat],
            "body_pos": {name: [round(float(c), 4) for c in bpos] for name, bpos in zip(robot.data.body_names, body_pos)},
            "joint_pos": {name: round(float(j), 4) for name, j in zip(joint_names, joints)},
        })

        if dones[0]:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "task": args.task,
            "dt": 0.02,
            "num_steps": len(trajectory),
            "joint_names": joint_names,
            "trajectory": trajectory,
        }, f, indent=2)

    print(f"Successfully exported {len(trajectory)} steps to {args.output}")
    simulation_app.close()

if __name__ == "__main__":
    main()
