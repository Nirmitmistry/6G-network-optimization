"""
train_rl.py
Trains the DQN agent on the NetworkEnv.
Logs rewards and saves checkpoints every 50 episodes.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import matplotlib.pyplot as plt
from envs.network_env import NetworkEnv
from agents.dqn_agent import DQNAgent
from utils.metrics import summarize_episode


def train(config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = NetworkEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config)
    episodes = config["training"]["episodes"]
    episode_rewards = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_rewards = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
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
    plt.title("DQN Training — 6G Network Optimization")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/dqn_rewards.png")
    print("Training complete. Plot saved to results/plots/dqn_rewards.png")


if __name__ == "__main__":
    train()
