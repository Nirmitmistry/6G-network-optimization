"""
gnn.py
Graph Neural Network for learning node embeddings from the 6G network graph.
Uses GCN layers (swap to GATConv or SAGEConv as needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GNNModel(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, out_channels: int = 32):
        """
        Args:
            in_channels:     node feature dimension (x, y, is_bs)
            hidden_channels: intermediate embedding size
            out_channels:    final node embedding size
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        return x  # (num_nodes, out_channels)
