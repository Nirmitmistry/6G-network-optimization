

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, NNConv, global_mean_pool, global_max_pool


class ResGATBlock(nn.Module):
    """GATv2 block with residual connection and layer norm."""

    def __init__(self, in_ch: int, out_ch: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv = GATv2Conv(in_ch, out_ch // heads, heads=heads, dropout=dropout, edge_dim=3)
        self.norm = nn.LayerNorm(out_ch)
        self.proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.gelu(self.norm(h + self.proj(x)))


class GNNModel(nn.Module):
    """
    Multi-layer GNN with residual GATv2 blocks.

    Args:
        in_channels:     node feature dim (default 5: x,y,is_bs,degree,load)
        hidden_channels: internal embedding size
        out_channels:    final node embedding size
        num_layers:      number of ResGATBlocks
        dropout:         dropout probability
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # Stack of residual GAT blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.blocks.append(ResGATBlock(hidden_channels, hidden_channels, heads=4, dropout=dropout))

        # Final output block (reduce to out_channels)
        self.out_block = ResGATBlock(hidden_channels, out_channels, heads=4, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        x = self.out_block(x, edge_index, edge_attr)
        return x  # (N, out_channels)
