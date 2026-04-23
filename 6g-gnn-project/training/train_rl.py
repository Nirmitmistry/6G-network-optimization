"""
train_rl.py
DQN training loop with:
  - Pretrained (frozen) GNN encoder
  - Periodic reward/loss logging
  - Checkpoint saving
  - Reward plot
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from envs.network_env import NetworkEnv
from agents.dqn_agent import DQNAgent
from models.encoder import GraphEncoder


def train(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tr = config["training"]
    gnn_cfg = config.get("gnn", {})
    episodes = tr["episodes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RL Training] device={device}")

    # Build encoder
    encoder = GraphEncoder(
        in_channels=gnn_cfg.get("in_channels", 5),
        hidden_channels=gnn_cfg.get("hidden_channels", 128),
        out_channels=gnn_cfg.get("out_channels", 64),
        num_layers=gnn_cfg.get("num_layers", 4),
        dropout=gnn_cfg.get("dropout", 0.2),
    ).to(device)

    gnn_path = "results/gnn_pretrained.pt"
    if os.path.exists(gnn_path):
        encoder.gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        print(f"  Loaded pretrained GNN from {gnn_path}")
    else:
        print("  WARNING: No pretrained GNN found, using random weights.")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    env = NetworkEnv(config)
    env.reset()
    state_dim = env.get_gnn_state(encoder, device).shape[0]
    action_dim = env.action_space.n
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    agent = DQNAgent(state_dim, action_dim, config)

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    episode_rewards = []
    episode_losses = []
    best_avg = -float("inf")

    for ep in range(1, episodes + 1):
        env.reset()
        state = env.get_gnn_state(encoder, device)
        total_reward = 0.0
        losses = []
        done = False

        while not done:
            action = agent.select_action(state)
            _, reward, done, _ = env.step(action)
            next_state = env.get_gnn_state(encoder, device)
            agent.store(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward

        agent.step_scheduler()
        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(losses) if losses else 0.0)

        if ep % 50 == 0:
            agent.save(f"results/dqn_ep{ep}.pt")

        if ep % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_l = np.mean(episode_losses[-10:])
            print(f"  Ep {ep:4d}/{episodes} | avg_reward={avg_r:.4f} | avg_loss={avg_l:.4f} | eps={agent.epsilon:.3f}")

            if avg_r > best_avg:
                best_avg = avg_r
                agent.save("results/dqn_best.pt")

    # Final save + plot
    agent.save(f"results/dqn_ep{episodes}.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(episode_rewards, alpha=0.4, label="reward")
    window = min(20, len(episode_rewards))
    smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(range(window - 1, len(episode_rewards)), smoothed, label=f"MA-{window}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("DQN Training Rewards")
    axes[0].legend()

    axes[1].plot(episode_losses, alpha=0.6)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("DQN Training Loss")

    plt.tight_layout()
    plt.savefig("results/plots/dqn_rewards.png", dpi=150)
    plt.close()
    print(f"  Plot saved to results/plots/dqn_rewards.png")
    print(f"  RL training done. Best avg reward={best_avg:.4f}")


if __name__ == "__main__":
    train()
