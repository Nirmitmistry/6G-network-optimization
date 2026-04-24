from envs.network_env import NetworkEnv
from agents.dqn_agent import DQNAgent
from models.encoder import GraphEncoder
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Ensure custom modules can be found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# --- Page Configuration ---
st.set_page_config(
    page_title="6G Network Optimizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Models for Performance ---


@st.cache_resource
def load_configuration_and_models():
    """Loads configs and models once to prevent reloading on every UI interaction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Config
    config_path = os.path.join(BASE_DIR, "configs/config.yaml")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found at {config_path}")
        st.stop()

    # Initialize Environment
    env = NetworkEnv(config)
    env.reset()

    # Initialize & Load GNN Encoder
    encoder = GraphEncoder(
        in_channels=config['gnn']['in_channels'],
        hidden_channels=config['gnn']['hidden_channels'],
        out_channels=config['gnn']['out_channels'],
        num_layers=config['gnn']['num_layers'],
        dropout=config['gnn']['dropout'],
    ).to(device)

    gnn_path = os.path.join(BASE_DIR, "results/gnn_pretrained.pt")
    gnn_status = "No pretrained GNN found — using random weights."
    if os.path.exists(gnn_path):
        encoder.gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        gnn_status = f"GNN loaded successfully from {gnn_path}"
    encoder.eval()

    # Initialize & Load DQN Agent
    state_dim = env.get_gnn_state(encoder, device).shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, config)

    model_path = os.path.join(BASE_DIR, "results/dqn_ep500.pt")
    dqn_status = "No trained DQN model found — using random policy."
    if os.path.exists(model_path):
        agent.load(model_path)
        dqn_status = f"DQN loaded successfully from {model_path}"

    return config, env, encoder, agent, device, gnn_status, dqn_status


# --- Load Resources ---
config, env, encoder, agent, device, gnn_status, dqn_status = load_configuration_and_models()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System Status")

    st.subheader("Compute")
    st.write(f"**Device:** `{device}`")

    st.subheader("Model Status")
    if "successfully" in gnn_status:
        st.success("✅ " + gnn_status)
    else:
        st.warning("⚠️ " + gnn_status)

    if "successfully" in dqn_status:
        st.success("✅ " + dqn_status)
    else:
        st.warning("⚠️ " + dqn_status)

    st.markdown("---")
    st.markdown(
        "Press **Run Episode** in the main window to simulate a network allocation cycle.")

# --- Main Dashboard ---
st.title("📡 6G GNN-DRL Network Optimization Dashboard")
st.markdown("""
This dashboard simulates a 6G network environment. It uses a **Graph Neural Network (GNN)** to encode the network's current state and a **Deep Reinforcement Learning (DRL)** agent to optimize resource allocation between users and base stations.
""")

st.divider()

# --- Simulation Logic ---
if st.button(" Run Simulation Episode", type="primary", use_container_width=True):
    with st.spinner("Simulating network environment..."):
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

    # --- Results Presentation ---
    st.header("Simulation Results")

    # Metrics Row
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Total Episode Reward", f"{sum(rewards):.3f}")
    col_metric2.metric("Total Steps", steps)
    col_metric3.metric("Average Reward/Step",
                       f"{(sum(rewards)/steps):.3f}" if steps > 0 else "0")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_chart1, col_chart2 = st.columns(2)

    # 1. Reward Chart
    with col_chart1:
        st.subheader(" Performance over Time")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(rewards, color="#1f77b4", linewidth=2)
        ax1.set_xlabel("Step", fontweight='bold')
        ax1.set_ylabel("Reward", fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

    # 2. Topology Chart
    with col_chart2:
        st.subheader(" Network Topology & Allocations")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bs = env.bs_positions
        users = env.user_positions

        # Plot Base Stations & Users
        ax2.scatter(bs[:, 0], bs[:, 1], c="#d62728", marker="^",
                    s=150, label="Base Stations", zorder=3)
        ax2.scatter(users[:, 0], users[:, 1], c="#2ca02c",
                    marker="o", s=60, label="Users", zorder=3)

        # Plot Connections
        for u in range(env.num_users):
            b = env.allocations[u]
            ax2.plot([users[u, 0], bs[b, 0]], [users[u, 1], bs[b, 1]],
                     color="gray", alpha=0.4, linewidth=1, zorder=1)

        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.legend(loc="upper right")
        st.pyplot(fig2)
