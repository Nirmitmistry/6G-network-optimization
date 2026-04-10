"""
encoder.py
Graph encoder that wraps the GNN and produces a flat state vector
suitable for RL agents.
"""

import torch
import torch.nn as nn
from models.gnn import GNNModel


class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.gnn = GNNModel(in_channels, hidden_channels, out_channels)
        self.pool_fc = nn.Linear(out_channels, out_channels)

    def forward(self, data):
        node_emb = self.gnn(data)                          # (N, out_channels)
        graph_emb = torch.mean(node_emb, dim=0, keepdim=True)  # global mean pool
        graph_emb = torch.relu(self.pool_fc(graph_emb))
        return graph_emb, node_emb
