# DOGU Quadruped Locomotion — Headless Isaac Lab Reinforcement Learning

This repository contains the complete reinforcement learning pipeline for the **DOGU quadruped robot** using **NVIDIA Isaac Sim / Isaac Lab (v17.0.2)** with **RSL-RL PPO (v5.4.1)** running headlessly on an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).

---

## 1. Project Organization

```text
~/robotics/dogu_locomotion/
├── assets/
│   └── dogu/
│       ├── meshes/          # 13 STL link meshes (base, hips, femurs, tibias)
│       ├── urdf/            # Validated URDF with calibrated mass, COM, inertia, and collisions
│       └── usd/             # Converted Omniverse USD asset (dogu.usda)
├── configs/
│   ├── dogu_robot_cfg.py    # Articulation configuration (actuator gains, stiffness, damping)
│   ├── rewards_cfg.py       # Configurable reward terms and weights
│   └── rsl_rl_ppo_cfg.py    # PPO agent network and hyperparameter configurations
├── tasks/
│   ├── dogu_standing_env_cfg.py    # Stage 2: Standing & balance task MDP
│   └── dogu_locomotion_env_cfg.py  # Stage 3: Forward walking locomotion task MDP
├── scripts/
│   ├── validate_simulation.py      # Stage 1: Headless articulation and physics sanity check
│   ├── train.py                    # Headless PPO training script
│   ├── evaluate.py                 # Deterministic headless evaluation script
│   └── monitor_resources.sh        # Real-time GPU VRAM, RAM, and thermals monitor
├── checkpoints/
│   ├── dogu_standing/              # Stage 2 policy checkpoints & TorchScript JIT
│   └── dogu_locomotion/            # Stage 3 policy checkpoints & TorchScript JIT
├── logs/
│   └── rsl_rl/                     # TensorBoard training metrics & logs
├── evaluations/                    # JSON evaluation summaries
└── docs/
    └── DOGU_LOCOMOTION_REPORT.md   # Comprehensive engineering report
```

---

## 2. Hardware and Software Specifications

- **OS**: Zorin OS 18.1 (Ubuntu 24.04 noble base)
- **CPU**: Intel Core i7 13th Gen HX
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM)
- **NVIDIA Driver**: 580.178.04
- **PyTorch**: 2.11.0+cu128
- **Isaac Sim / Isaac Lab**: 17.0.2
- **RL Framework**: RSL-RL 5.4.1

---

## 3. Robot Model Verification Summary

- **Total Links**: 13 (1 base + 4 hips + 4 femurs + 4 tibias)
- **Total Revolute Joints**: 12 (3 DOF per leg: abduction/hip, femur/pitch, tibia/knee)
- **Total Robot Mass**: 0.880 kg (calibrated from STL mesh volume and PLA/servo densities)
- **Nominal Standing Height**: 0.108 m (10.8 cm)
- **Actuator Model**: PD implicit position drive ($K_p = 40.0\text{ N}\cdot\text{m/rad}$, $K_d = 2.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$, effort limit $3.0\text{ N}\cdot\text{m}$, velocity limit $10.0\text{ rad/s}$).
- **Joint Limits**: $-1.5708\text{ to }+1.5708\text{ rad}$ ($\pm 90^\circ$).

---

## 4. Quickstart Commands

### Step 1: Run Stage 1 Simulation Validation
Validates articulation import, joint limits, physics stepping, and checks for NaNs or exploding velocities:
```bash
cd ~/robotics/dogu_locomotion
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
/home/chakradhar/robotics/isaac_environment/IsaacLab/.venv/bin/python scripts/validate_simulation.py
```

### Step 2: System Resource Monitor
Launch in a separate terminal to watch VRAM and system memory:
```bash
cd ~/robotics/dogu_locomotion
./scripts/monitor_resources.sh
```

### Step 3: Train Stage 2 (Standing Task)
Trains the policy to hold an upright standing posture without drifting or collapsing:
```bash
/home/chakradhar/robotics/isaac_environment/IsaacLab/.venv/bin/python scripts/train.py \
    --task standing \
    --num_envs 16 \
    --max_iterations 60
```

### Step 4: Evaluate Stage 2 Policy
Runs 10 deterministic episodes (1000 steps each):
```bash
/home/chakradhar/robotics/isaac_environment/IsaacLab/.venv/bin/python scripts/evaluate.py \
    --task standing \
    --checkpoint checkpoints/dogu_standing/standing_best.pt \
    --num_episodes 10
```

### Step 5: Train Stage 3 (Forward Walking Locomotion)
Warm-starts from the trained standing policy and trains forward velocity tracking:
```bash
/home/chakradhar/robotics/isaac_environment/IsaacLab/.venv/bin/python scripts/train.py \
    --task locomotion \
    --num_envs 16 \
    --max_iterations 150 \
    --checkpoint checkpoints/dogu_standing/standing_best.pt
```

### Step 6: Evaluate Stage 3 Policy
Evaluates forward displacement, velocity tracking, and fall rate across 10 deterministic episodes:
```bash
/home/chakradhar/robotics/isaac_environment/IsaacLab/.venv/bin/python scripts/evaluate.py \
    --task locomotion \
    --checkpoint checkpoints/dogu_locomotion/locomotion_best.pt \
    --num_episodes 10
```

---

## 5. TensorBoard Metrics

To view the training curves, reward terms, and loss metrics:
```bash
/home/chakradhar/robotics/isaac_environment/IsaacLab/.venv/bin/python -m tensorboard.main \
    --logdir logs/rsl_rl/ --port 6006
```
Open `http://localhost:6006` in your browser.
