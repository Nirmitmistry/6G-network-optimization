import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def build_network_graph(
    bs_positions: np.ndarray,
    user_positions: np.ndarray,
    allocations: np.ndarray = None,
    max_distance: float = 900.0,
    area_size: float = 3000.0,
):
    
    num_bs = len(bs_positions)
    num_users = len(user_positions)
    all_positions = np.vstack([bs_positions, user_positions]).astype(np.float32)
    n = len(all_positions)

    # Normalise positions to [0, 1]
    pos_norm = all_positions / area_size

    # is_bs flag
    is_bs = np.array([[1.0]] * num_bs + [[0.0]] * num_users, dtype=np.float32)

    # Load ratio: fraction of users assigned to each BS (0 for user nodes)
    load = np.zeros((n, 1), dtype=np.float32)
    if allocations is not None:
        for u_idx, b_idx in enumerate(allocations):
            load[b_idx, 0] += 1.0
        load[:num_bs] /= max(num_users, 1)

    # Build edges
    edge_src, edge_dst = [], []
    edge_attr_list = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(all_positions[i] - all_positions[j])
            if dist <= max_distance:
                edge_src += [i, j]
                edge_dst += [j, i]
                norm_dist = dist / max_distance
                dx = (all_positions[j, 0] - all_positions[i, 0]) / area_size
                dy = (all_positions[j, 1] - all_positions[i, 1]) / area_size
                edge_attr_list += [[norm_dist, dx, dy], [norm_dist, -dx, -dy]]

    # Degree as a node feature (normalised)
    degree = np.zeros((n, 1), dtype=np.float32)
    for s in edge_src:
        degree[s, 0] += 1.0
    max_deg = max(degree.max(), 1.0)
    degree /= max_deg

    node_features = np.hstack([pos_norm, is_bs, degree, load])  # (N, 5)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if edge_attr_list else torch.zeros((0, 3), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_networkx_graph(bs_positions: np.ndarray, user_positions: np.ndarray, max_distance: float = 900.0):
    G = nx.Graph()
    num_bs = len(bs_positions)
    all_positions = np.vstack([bs_positions, user_positions])

    for i, pos in enumerate(all_positions):
        node_type = "bs" if i < num_bs else "user"
        G.add_node(i, pos=pos, type=node_type)

    for i in range(len(all_positions)):
        for j in range(i + 1, len(all_positions)):
            dist = np.linalg.norm(all_positions[i] - all_positions[j])
            if dist <= max_distance:
                G.add_edge(i, j, weight=dist)

    return G
