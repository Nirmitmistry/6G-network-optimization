"""
app.py
Streamlit dashboard for visualizing 6G network state and agent decisions.
Run with: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
from envs.network_env import NetworkEnv
from agents.dqn_agent import DQNAgent

st.set_page_config(page_title="6G Network Optimizer", layout="wide")
st.title("6G GNN-DRL Network Optimization Dashboard")

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

env = NetworkEnv(config)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, config)

# Optionally load a trained model
model_path = "results/dqn_ep500.pt"
if os.path.exists(model_path):
    agent.load(model_path)
    st.success(f"Loaded trained model from {model_path}")
else:
    st.warning("No trained model found — using random policy. Run training/train_rl.py first.")

if st.button("Run Episode"):
    state = env.reset()
    rewards, steps = [], 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        steps += 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reward", f"{sum(rewards):.3f}")
        st.metric("Steps", steps)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Per-Step Reward")
        st.pyplot(fig)

    # Network topology plot
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    bs = env.bs_positions
    users = env.user_positions
    ax2.scatter(bs[:, 0], bs[:, 1], c="red", marker="^", s=100, label="Base Stations")
    ax2.scatter(users[:, 0], users[:, 1], c="blue", marker="o", s=40, label="Users")
    for u in range(env.num_users):
        b = env.allocations[u]
        ax2.plot([users[u, 0], bs[b, 0]], [users[u, 1], bs[b, 1]], "gray", alpha=0.3)
    ax2.set_title("Network Topology & Allocations")
    ax2.legend()
    st.pyplot(fig2)
