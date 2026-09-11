"""
RL training/evaluation entry point for non-prehensile push task with YHRG S1.

Glue code for Genesis simulator.  External dependencies (used as-is):
  - rsl_rl (rsl-rl-lib 4.0.1): OnPolicyRunner + PPO (production-grade)
  - genesis: Full simulator implementation (production-grade)

Usage (in sam3 conda environment):
    # Training
    python rl_integration.py
    python rl_integration.py --num_envs 32 --max_iterations 2000
    python rl_integration.py --vis   # with viewer

    # Training with domain randomization
    python rl_integration.py --dr size,mass,friction
    python rl_integration.py --dr-config config/dr_push.yaml

    # Evaluation
    python rl_integration.py --eval --checkpoint model_2999.pt
    python rl_integration.py --eval --checkpoint model_2999.pt --num_episodes 10

Note: the DR configuration must be identical between training and evaluation;
omitting --dr / --dr-config reproduces the pre-DR (nominal) baseline.
"""

import argparse
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from rl_push_env import PushEnv
from model.domain_randomization import (
    DomainRandomizationConfig,
    RANDOMIZER_REGISTRY,
    get_rl_push_dr_config,
)


# ─────────────────── rsl_rl 4.x training config ───────────────────

def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    """PPO training configuration for rsl_rl 4.0.1 OnPolicyRunner.

    Key differences from rsl_rl 2.2.4:
      - 'actor' and 'critic' replace the old 'policy' dict
      - class_name references MLPModel (not ActorCritic)
      - obs_groups maps observation sets to TensorDict keys
      - algorithm.rnd_cfg must exist (can be None)
    """
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "fixed",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "rnd_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "init_noise_std": 0.5,
            "stochastic": True,
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 10,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 48,
        "save_interval": 500,
        "seed": 1,
        "multi_gpu": None,
    }


# ─────────────────── domain randomization config ───────────────────

def build_dr_config(args) -> DomainRandomizationConfig:
    """Assemble the DR config: defaults < --dr-config file < --dr name list.

    Randomization stays opt-in: without any flag every randomizer keeps its
    ``enabled: False`` default, i.e. behaviour is bit-identical to pre-DR runs.
    """
    specs = get_rl_push_dr_config().to_dict()

    if args.dr_config:  # full spec (ranges + enable flags) from YAML
        specs.update(DomainRandomizationConfig.from_yaml(args.dr_config).to_dict())

    requested = [s.strip() for s in args.dr.split(",")]
    requested = [s for s in requested if s]
    if "all" in requested:
        requested = list(specs.keys())

    for name in requested:
        if name not in RANDOMIZER_REGISTRY:
            raise ValueError(
                f"Unknown domain randomizer: '{name}'. "
                f"Available: {RANDOMIZER_REGISTRY.get_available()}"
            )
        specs.setdefault(name, {})["enabled"] = True

    return DomainRandomizationConfig.from_dict(specs)


# ─────────────────── main ───────────────────

def main():
    parser = argparse.ArgumentParser(description="YHRG S1 Push Task – RL Training/Evaluation")
    parser.add_argument("-e", "--exp_name", type=str, default="yhrg_s1_push")
    parser.add_argument("-B", "--num_envs", type=int, default=16)
    parser.add_argument("--max_iterations", type=int, default=13000)
    parser.add_argument("--vis", action="store_true", default=False)
    
    # Evaluation arguments
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Run evaluation instead of training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file for evaluation (e.g., model_2999.pt)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of evaluation episodes")

    # Domain randomization arguments
    parser.add_argument("--dr", type=str, default="",
                        help="Comma-separated randomizers to enable "
                             "(size,mass,friction or 'all'); empty disables DR")
    parser.add_argument("--dr-config", type=str, default=None,
                        help="YAML file with the full DR spec, e.g. config/dr_push.yaml")
    args = parser.parse_args()

    # ── initialise Genesis ──
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=1,
        performance_mode=True,
    )

    # ── domain randomization ──
    dr_config = build_dr_config(args)
    enabled_dr = [name for name, spec in dr_config.specs.items() if spec.get("enabled")]
    print(f"[DR] enabled: {enabled_dr if enabled_dr else 'none (nominal baseline)'}")

    # ── create environment ──
    env = PushEnv(
        num_envs=args.num_envs,
        show_viewer=args.vis,
        dr_config=dr_config,
    )

    # ── training config ──
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # ── log directory ──
    log_dir = os.path.join("logs", args.exp_name)
    if not args.eval:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
            pickle.dump(train_cfg, f)

        # Persist the DR spec so a run can be reproduced / evaluated exactly
        with open(os.path.join(log_dir, "dr_config.pkl"), "wb") as f:
            pickle.dump(dr_config.to_dict(), f)

    # ── launch OnPolicyRunner ──
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # ── EVALUATION MODE ──
    if args.eval:
        import torch
        
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluation mode")
        
        checkpoint_path = os.path.join(log_dir, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*80}")
        print(f"EVALUATION MODE")
        print(f"{'='*80}")
        print(f"Loading checkpoint: {checkpoint_path}")
        runner.load(checkpoint_path)
        print(f"Checkpoint loaded successfully!")
        print(f"Running {args.num_episodes} evaluation episodes...")
        print(f"{'='*80}\n")
        
        # Get inference policy once
        policy = runner.get_inference_policy()
        
        # Run evaluation episodes
        max_eval_steps = env.max_episode_length + 10  # safety margin

        # Per-env tracking (all tensors on device)
        env_episode_reward = torch.zeros(env.num_envs, device=env.device)
        env_episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        # Pre-allocate result buffers
        eval_success_buf = torch.zeros(args.num_episodes, device=env.device)
        eval_reward_buf = torch.zeros(args.num_episodes, device=env.device)
        eval_steps_buf = torch.zeros(args.num_episodes, dtype=torch.long, device=env.device)
        completed_episodes = 0

        obs = env.reset()

        while completed_episodes < args.num_episodes:
            with torch.no_grad():
                actions = policy(obs)

            obs, rew, dones, extras = env.step(actions)
            env_episode_reward += rew
            env_episode_steps += 1

            # ── Process completed envs (pure tensor ops) ──
            done_mask = dones.clone()
            n_done = done_mask.sum().item()

            if n_done > 0:
                remaining = args.num_episodes - completed_episodes
                n_record = min(int(n_done), remaining)

                # Batch-read done env indices (only first n_record)
                done_indices = done_mask.nonzero(as_tuple=True)[0][:n_record]

                # Batch-read success, reward, steps via tensor indexing
                successes = env.last_success[done_indices].float()
                rewards = env_episode_reward[done_indices]
                steps = env_episode_steps[done_indices]

                # Store into pre-allocated buffers (slice assignment)
                sl = slice(completed_episodes, completed_episodes + n_record)
                eval_success_buf[sl] = successes
                eval_reward_buf[sl] = rewards
                eval_steps_buf[sl] = steps
                completed_episodes += n_record

                # Print results (lightweight loop for display only)
                for i in range(n_record):
                    ep = completed_episodes - n_record + i + 1
                    s = successes[i].item()
                    r = rewards[i].item()
                    st = steps[i].item()
                    print(f"Episode {ep}/{args.num_episodes}: "
                          f"Steps={st}, Reward={r:.2f}, "
                          f"Success={'100.00' if s > 0.5 else '0.00'}%")

                # Reset per-env trackers (tensor op, no loop)
                env_episode_reward[done_indices] = 0.0
                env_episode_steps[done_indices] = 0

            # Safety: prevent infinite loop
            if env_episode_steps.min().item() > max_eval_steps:
                break

        # Convert to Python lists for summary
        eval_successes = eval_success_buf[:completed_episodes].tolist()
        eval_rewards = eval_reward_buf[:completed_episodes].tolist()
        
        # Print evaluation summary
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total episodes: {args.num_episodes}")
        avg_reward = sum(eval_rewards)/len(eval_rewards) if eval_rewards else 0.0
        avg_success = sum(eval_successes)/len(eval_successes) if eval_successes else 0.0
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {avg_success:.2%}")
        print(f"{'='*80}\n")
        
        # Save evaluation results
        eval_results = {
            "checkpoint": args.checkpoint,
            "num_episodes": args.num_episodes,
            "successes": eval_successes,
            "rewards": eval_rewards,
            "avg_success": avg_success,
            "avg_reward": avg_reward,
        }
        
        eval_file = os.path.join(log_dir, f"eval_{os.path.basename(args.checkpoint).replace('.pt', '')}.pkl")
        with open(eval_file, "wb") as f:
            pickle.dump(eval_results, f)
        print(f"Evaluation results saved to: {eval_file}")
        
        env.close()
        return
    
    # ── TRAINING MODE ──
    if not args.eval:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
