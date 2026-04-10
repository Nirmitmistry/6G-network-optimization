import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from envs.network_env import NetworkEnv
from agents.dqn_agent import DQNAgent
from models.encoder import GraphEncoder

st.set_page_config(page_title="6G Network Optimizer", layout="wide")
st.title("6G GNN-DRL Network Optimization Dashboard")

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder = GraphEncoder(in_channels=3, hidden_channels=64, out_channels=32).to(device)
gnn_path = "results/gnn_pretrained.pt"
if os.path.exists(gnn_path):
    encoder.gnn.load_state_dict(torch.load(gnn_path, map_location=device))
    st.info(f"GNN encoder loaded from {gnn_path}")
else:
    st.warning("No pretrained GNN found — using random encoder weights.")
encoder.eval()

env = NetworkEnv(config)
env.reset()
state_dim = env.get_gnn_state(encoder, device).shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim, config)
model_path = "results/dqn_ep500.pt"
if os.path.exists(model_path):
    agent.load(model_path)
    st.success(f"Loaded trained DQN model from {model_path}")
else:
    st.warning("No trained DQN model found — using random policy. Run main.py first.")

if st.button("Run Episode"):
    env.reset()
    state = env.get_gnn_state(encoder, device)
    rewards, steps = [], 0
    done = False

    while not done:
        action = agent.select_action(state)
        _, reward, done, _ = env.step(action)
        state = env.get_gnn_state(encoder, device)
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
