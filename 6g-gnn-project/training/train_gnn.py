"""
train_gnn.py
Self-supervised GNN pre-training using:
  - Node feature reconstruction (autoencoder-style)
  - Graph-level contrastive loss (SimCLR-style augmentation)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch

from models.encoder import GraphEncoder
from utils.graph_builder import build_network_graph


def random_graph(config: dict) -> Data:
    net = config["network"]
    num_bs = net["num_base_stations"]
    num_users = net["num_users"]
    area = net["area_size"]
    max_dist = net.get("max_edge_distance", area * 0.3)

    bs_pos = np.random.uniform(0, area, (num_bs, 2)).astype(np.float32)
    user_pos = np.random.uniform(0, area, (num_users, 2)).astype(np.float32)
    alloc = np.random.randint(0, num_bs, num_users)
    return build_network_graph(bs_pos, user_pos, alloc, max_dist, area)


def augment_graph(data: Data, noise_std: float = 0.02, drop_edge_p: float = 0.1) -> Data:
    """Light augmentation: add node feature noise + randomly drop edges."""
    x = data.x + torch.randn_like(data.x) * noise_std
    x = x.clamp(0, 1)

    if data.edge_index.size(1) > 0 and drop_edge_p > 0:
        mask = torch.rand(data.edge_index.size(1)) > drop_edge_p
        edge_index = data.edge_index[:, mask]
        edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None
    else:
        edge_index = data.edge_index
        edge_attr = data.edge_attr

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent loss for a batch of paired embeddings."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                          # (2B, D)
    sim = torch.mm(z, z.T) / temperature                    # (2B, 2B)
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


def train(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gnn_cfg = config.get("gnn", {})
    epochs = gnn_cfg.get("pretrain_epochs", 100)
    lr = gnn_cfg.get("pretrain_lr", 0.0005)
    batch_size = gnn_cfg.get("pretrain_batch", 32)
    num_graphs = gnn_cfg.get("num_pretrain_graphs", 500)
    in_ch = gnn_cfg.get("in_channels", 5)
    hidden = gnn_cfg.get("hidden_channels", 128)
    out_ch = gnn_cfg.get("out_channels", 64)
    num_layers = gnn_cfg.get("num_layers", 4)
    dropout = gnn_cfg.get("dropout", 0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GNN Pre-training] device={device}")

    encoder = GraphEncoder(in_ch, hidden, out_ch, num_layers, dropout).to(device)
    proj_head = ProjectionHead(out_ch, proj_dim=64).to(device)

    # Node reconstruction decoder
    decoder = nn.Sequential(
        nn.Linear(out_ch, hidden),
        nn.GELU(),
        nn.Linear(hidden, in_ch),
    ).to(device)

    params = list(encoder.parameters()) + list(proj_head.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Pre-generate graphs
    print(f"  Generating {num_graphs} random graphs...")
    graphs = [random_graph(config) for _ in range(num_graphs)]

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        decoder.train()

        indices = np.random.permutation(num_graphs)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, num_graphs - batch_size + 1, batch_size):
            batch_graphs = [graphs[i] for i in indices[start: start + batch_size]]

            # Two augmented views
            aug1 = Batch.from_data_list([augment_graph(g) for g in batch_graphs]).to(device)
            aug2 = Batch.from_data_list([augment_graph(g) for g in batch_graphs]).to(device)

            # Forward
            _, node_emb1 = encoder(aug1)
            _, node_emb2 = encoder(aug2)

            # Graph-level embeddings via mean pool per graph
            batch_vec1 = aug1.batch
            batch_vec2 = aug2.batch
            from torch_geometric.nn import global_mean_pool
            g_emb1 = proj_head(global_mean_pool(node_emb1, batch_vec1))  # (B, proj_dim)
            g_emb2 = proj_head(global_mean_pool(node_emb2, batch_vec2))

            loss_contrast = contrastive_loss(g_emb1, g_emb2)

            # Node reconstruction on original (non-augmented) graphs
            orig_batch = Batch.from_data_list(batch_graphs).to(device)
            _, node_emb_orig = encoder(orig_batch)
            recon = decoder(node_emb_orig)
            loss_recon = F.mse_loss(recon, orig_batch.x)

            loss = loss_contrast + loss_recon

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("results", exist_ok=True)
            torch.save(encoder.gnn.state_dict(), "results/gnn_pretrained.pt")

    print(f"  GNN pre-training done. Best loss={best_loss:.4f}. Saved to results/gnn_pretrained.pt")


if __name__ == "__main__":
    train()
