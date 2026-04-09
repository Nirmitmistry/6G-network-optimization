"""
train_gnn.py
Pre-trains the GNN encoder in a supervised or self-supervised fashion
using synthetic network topology data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch_geometric.loader import DataLoader
from models.gnn import GNNModel
from utils.graph_builder import build_network_graph


def generate_dataset(config, num_samples=200):
    net = config["network"]
    graphs = []
    for _ in range(num_samples):
        bs_pos = np.random.uniform(0, net["area_size"], (net["num_base_stations"], 2))
        user_pos = np.random.uniform(0, net["area_size"], (net["num_users"], 2))
        g = build_network_graph(bs_pos, user_pos)
        # Dummy regression target: mean distance to nearest BS per node
        g.y = torch.rand(g.num_nodes, 1)
        graphs.append(g)
    return graphs


def train(config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = generate_dataset(config)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = GNNModel(in_channels=3, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()
    for epoch in range(1, 51):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)                          # (N, 32)
            loss = F.mse_loss(out[:, :1], batch.y)     # dummy supervised loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/gnn_pretrained.pt")
    print("GNN saved to results/gnn_pretrained.pt")


if __name__ == "__main__":
    train()
