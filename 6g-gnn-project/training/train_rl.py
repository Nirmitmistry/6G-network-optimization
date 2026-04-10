"""
train_rl.py
Trains the DQN agent on the NetworkEnv using GNN-encoded states.
Loads the pretrained GNN encoder and uses get_gnn_state() for observations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs.network_env import NetworkEnv
from agents.dqn_agent import DQNAgent
from models.encoder import GraphEncoder
from utils.metrics import summarize_episode


def get_state_dim(env, encoder, device):
    """Compute the GNN state dimension from a live env reset."""
    env.reset()
    state = env.get_gnn_state(encoder, device)
    return state.shape[0]


def train(config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained GNN encoder
    encoder = GraphEncoder(in_channels=3, hidden_channels=64, out_channels=32).to(device)
    gnn_path = "results/gnn_pretrained.pt"
    if os.path.exists(gnn_path):
        encoder.gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        print(f"Loaded pretrained GNN from {gnn_path}")
    else:
        print("No pretrained GNN found — using random encoder weights.")
    encoder.eval()

    env = NetworkEnv(config)
    state_dim = get_state_dim(env, encoder, device)
    action_dim = env.action_space.n
    print(f"GNN state dim: {state_dim} | Actions: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, config)
    episodes = config["training"]["episodes"]
    episode_rewards = []

    for ep in range(1, episodes + 1):
        env.reset()
        state = env.get_gnn_state(encoder, device)
        ep_rewards = []
        done = False

        while not done:
            action = agent.select_action(state)
            _, reward, done, _ = env.step(action)
            next_state = env.get_gnn_state(encoder, device)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            ep_rewards.append(reward)

        if ep % 10 == 0:
            agent.update_target()

        stats = summarize_episode(ep_rewards)
        episode_rewards.append(stats["total_reward"])

        if ep % 50 == 0:
            os.makedirs("results", exist_ok=True)
            agent.save(f"results/dqn_ep{ep}.pt")
            print(f"Episode {ep}/{episodes} | Total Reward: {stats['total_reward']:.3f} | ε: {agent.epsilon:.3f}")

    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training — 6G Network Optimization (GNN State)")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/dqn_rewards.png")
    print("Training complete. Plot saved to results/plots/dqn_rewards.png")


if __name__ == "__main__":
    train()
