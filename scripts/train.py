#!/usr/bin/env python3
"""
Headless Training Script for DOGU Quadruped Locomotion
Supports Stage 2 (Standing) and Stage 3 (Locomotion)
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Train DOGU robot with RSL-RL PPO headlessly")
    parser.add_argument("--task", type=str, default="standing", choices=["standing", "locomotion"],
                        help="Task stage to train: standing (Stage 2) or locomotion (Stage 3)")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments (default: 16 for 6GB VRAM)")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Maximum training iterations (override config default)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to warm-start or resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Initialize Isaac Lab AppLauncher in headless mode
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    # 2. Imports after simulation app startup
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from tasks.dogu_standing_env_cfg import DoguStandingEnvCfg
    from tasks.dogu_locomotion_env_cfg import DoguLocomotionEnvCfg
    from configs.rsl_rl_ppo_cfg import DoguStandingPPORunnerCfg, DoguLocomotionPPORunnerCfg

    configure_seed(args.seed)

    # 3. Select Task and Agent Configuration
    if args.task == "standing":
        env_cfg = DoguStandingEnvCfg()
        agent_cfg = DoguStandingPPORunnerCfg()
    else:
        env_cfg = DoguLocomotionEnvCfg()
        agent_cfg = DoguLocomotionPPORunnerCfg()

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed

    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    # Apply version migration
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.4.1")

    # 4. Setup Logging Directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(PROJECT_ROOT, "logs", agent_cfg.experiment_name, timestamp)
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", agent_cfg.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("==============================================================================")
    print("                     DOGU Locomotion Headless Training                        ")
    print("==============================================================================")
    print(f"[*] Task Stage       : {args.task.upper()}")
    print(f"[*] Parallel Envs    : {env_cfg.scene.num_envs}")
    print(f"[*] Max Iterations   : {agent_cfg.max_iterations}")
    print(f"[*] Random Seed      : {args.seed}")
    print(f"[*] Log Directory    : {run_dir}")
    print(f"[*] Checkpoint Dir   : {ckpt_dir}")
    if args.checkpoint:
        print(f"[*] Loading Checkpoint: {args.checkpoint}")
    print("==============================================================================")

    # 5. Build Environment
    print("[*] Instantiating ManagerBasedRLEnv...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)
    print("[*] Environment wrapped with RslRlVecEnvWrapper successfully.")

    # 6. Setup RSL-RL Runner
    train_cfg_dict = agent_cfg.to_dict()
    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=train_cfg_dict,
        log_dir=run_dir,
        device=str(env.sim.device),
    )

    if args.checkpoint:
        print(f"[*] Loading weights from {args.checkpoint}...")
        runner.load(args.checkpoint)
        print("[*] Weights loaded successfully.")

    # 7. Execute Training Loop
    print(f"\n[*] Commencing PPO training for {agent_cfg.max_iterations} iterations...\n")
    start_time = time.time()
    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        print("\n[*] Training interrupted by user.")
    finally:
        total_time = time.time() - start_time
        print(f"\n[*] Training completed in {total_time:.2f} seconds ({total_time / 60.0:.2f} mins).")

        # Save final and best checkpoints
        final_ckpt = os.path.join(ckpt_dir, f"{args.task}_final.pt")
        runner.save(final_ckpt)
        print(f"[*] Saved final checkpoint: {final_ckpt}")

        # Also copy latest model from run_dir to checkpoints
        latest_models = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
        if latest_models:
            latest_models.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            best_src = os.path.join(run_dir, latest_models[-1])
            best_dst = os.path.join(ckpt_dir, f"{args.task}_best.pt")
            import shutil
            shutil.copyfile(best_src, best_dst)
            print(f"[*] Saved best checkpoint copy: {best_dst}")

        # Export policy to TorchScript JIT for zero-dependency deployment
        try:
            jit_path = os.path.join(ckpt_dir, f"{args.task}_policy.jit")
            runner.export_policy_to_jit(jit_path)
            print(f"[*] Exported TorchScript policy: {jit_path}")
        except Exception as e:
            print(f"[NOTE] JIT export note: {e}")

        # 8. Clean Shutdown
        print("[*] Closing simulation app...")
        wrapped_env.close()
        simulation_app.close()
        print("[*] Isaac Sim exited cleanly.")

if __name__ == "__main__":
    main()
