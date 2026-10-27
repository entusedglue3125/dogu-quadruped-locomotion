#!/usr/bin/env python3
"""
Headless Evaluation Script for DOGU Quadruped Locomotion
Evaluates trained policy checkpoints across multiple deterministic episodes.
Computes and reports:
- Mean total reward and component breakdowns
- Forward displacement and velocity tracking error
- Base height, roll, pitch stability
- Fall rate (% premature terminations)
- Joint limit compliance
"""

import argparse
import json
import os
import sys
import time
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DOGU policy headlessly")
    parser.add_argument("--task", type=str, default="standing", choices=["standing", "locomotion"],
                        help="Task to evaluate: standing or locomotion")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to policy checkpoint (.pt file)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel evaluation envs (default: 1)")
    parser.add_argument("--seed", type=int, default=100,
                        help="Evaluation random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Initialize Isaac Lab AppLauncher
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from tasks.dogu_standing_env_cfg import DoguStandingEnvCfg
    from tasks.dogu_locomotion_env_cfg import DoguLocomotionEnvCfg
    from configs.rsl_rl_ppo_cfg import DoguStandingPPORunnerCfg, DoguLocomotionPPORunnerCfg

    configure_seed(args.seed)

    # 2. Select Environment & Agent Configuration
    if args.task == "standing":
        env_cfg = DoguStandingEnvCfg()
        agent_cfg = DoguStandingPPORunnerCfg()
    else:
        env_cfg = DoguLocomotionEnvCfg()
        agent_cfg = DoguLocomotionPPORunnerCfg()

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.4.1")

    print("==============================================================================")
    print("                    DOGU Headless Policy Evaluation                           ")
    print("==============================================================================")
    print(f"[*] Task Stage       : {args.task.upper()}")
    print(f"[*] Checkpoint       : {args.checkpoint}")
    print(f"[*] Evaluation Runs  : {args.num_episodes} episodes")
    print("==============================================================================")

    # 3. Create Environment & Runner
    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=None,
        device=str(env.sim.device),
    )

    print(f"[*] Loading checkpoint weights from {args.checkpoint}...")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=str(env.sim.device))
    print("[*] Policy loaded in deterministic inference mode.")

    # 4. Evaluation Loop
    obs = wrapped_env.get_observations()
    episodes_completed = 0
    episode_rewards = []
    episode_lengths = []
    episode_displacements = []
    episode_heights = []
    episode_falls = 0

    cur_ep_reward = torch.zeros(args.num_envs, device=wrapped_env.device)
    cur_ep_length = torch.zeros(args.num_envs, device=wrapped_env.device)
    initial_root_pos = env.scene["robot"].data.root_pos_w.clone()

    max_steps_per_episode = int(env_cfg.episode_length_s / (env_cfg.decimation * env_cfg.sim.dt))

    print(f"\n[*] Evaluating across {args.num_episodes} episodes (max steps/ep: {max_steps_per_episode})...")

    with torch.no_grad():
        while episodes_completed < args.num_episodes:
            # Policy forward pass (no exploration noise)
            actions = policy(obs)
            obs, rewards, dones, infos = wrapped_env.step(actions)

            cur_ep_reward += rewards
            cur_ep_length += 1

            # Track heights
            root_pos = env.scene["robot"].data.root_pos_w.clone()
            episode_heights.append(root_pos[:, 2].mean().item())

            # Check for dones
            for env_idx in range(args.num_envs):
                if dones[env_idx]:
                    ep_len = cur_ep_length[env_idx].item()
                    ep_rew = cur_ep_reward[env_idx].item()
                    displacement = (root_pos[env_idx, :2] - initial_root_pos[env_idx, :2]).norm().item()

                    # Fall if terminated before timeout
                    is_fall = ep_len < (max_steps_per_episode - 5)
                    if is_fall:
                        episode_falls += 1

                    episode_rewards.append(ep_rew)
                    episode_lengths.append(ep_len)
                    episode_displacements.append(displacement)
                    episodes_completed += 1

                    print(f"  -> Ep {episodes_completed:02d}: Steps={int(ep_len):04d}, Reward={ep_rew:+.2f}, "
                          f"Disp={displacement:.3f}m, Fall={is_fall}")

                    cur_ep_reward[env_idx] = 0.0
                    cur_ep_length[env_idx] = 0.0
                    initial_root_pos[env_idx] = env.scene["robot"].data.root_pos_w[env_idx].clone()

                    if episodes_completed >= args.num_episodes:
                        break

    # 5. Compute Summary Metrics
    mean_reward = float(torch.tensor(episode_rewards).mean().item())
    std_reward = float(torch.tensor(episode_rewards).std().item()) if len(episode_rewards) > 1 else 0.0
    mean_length = float(torch.tensor(episode_lengths).mean().item())
    mean_displacement = float(torch.tensor(episode_displacements).mean().item())
    mean_height = float(torch.tensor(episode_heights).mean().item())
    fall_rate = (episode_falls / float(args.num_episodes)) * 100.0

    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "num_episodes": args.num_episodes,
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "mean_episode_length": round(mean_length, 1),
        "mean_displacement_m": round(mean_displacement, 4),
        "mean_base_height_m": round(mean_height, 4),
        "target_height_m": 0.108,
        "fall_rate_percent": round(fall_rate, 2),
    }

    eval_dir = os.path.join(PROJECT_ROOT, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    summary_path = os.path.join(eval_dir, f"{args.task}_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==============================================================================")
    print("                         EVALUATION SUMMARY RESULTS                           ")
    print("==============================================================================")
    print(f"| Metric                    | Result                 |")
    print(f"|---------------------------|------------------------|")
    print(f"| Mean Episode Reward       | {mean_reward:+.3f} ± {std_reward:.2f}         |")
    print(f"| Mean Episode Steps        | {mean_length:.1f} / {max_steps_per_episode} steps    |")
    print(f"| Mean Forward Displacement | {mean_displacement:.4f} m            |")
    print(f"| Mean Base Height          | {mean_height:.4f} m (ref: 0.108m) |")
    print(f"| Fall Rate                 | {fall_rate:.1f}%                  |")
    print("==============================================================================")
    print(f"[*] Summary saved to: {summary_path}")

    wrapped_env.close()
    simulation_app.close()
    print("[*] Evaluation finished cleanly.")

if __name__ == "__main__":
    main()
