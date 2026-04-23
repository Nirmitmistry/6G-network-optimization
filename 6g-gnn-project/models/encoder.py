

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn import GNNModel


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn = GNNModel(in_channels, hidden_channels, out_channels, num_layers, dropout)

        # Graph-level projection: mean+max concat → out_channels
        self.graph_proj = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, data):
        node_emb = self.gnn(data)                              # (N, out_channels)
        mean_pool = node_emb.mean(dim=0, keepdim=True)         # (1, out_channels)
        max_pool = node_emb.max(dim=0).values.unsqueeze(0)     # (1, out_channels)
        graph_emb = self.graph_proj(
            torch.cat([mean_pool, max_pool], dim=-1)
        )                                                       # (1, out_channels)
        return graph_emb, node_emb
