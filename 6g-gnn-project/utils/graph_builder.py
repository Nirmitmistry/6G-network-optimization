import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def build_network_graph(bs_positions: np.ndarray, user_positions: np.ndarray, max_distance: float = 300.0):
    num_bs = len(bs_positions)
    num_users = len(user_positions)
    all_positions = np.vstack([bs_positions, user_positions])  # (N, 2)

    node_types = np.array([[1]] * num_bs + [[0]] * num_users, dtype=np.float32)
    node_features = np.hstack([all_positions.astype(np.float32), node_types])


    edge_src, edge_dst = [], []
    for i in range(len(all_positions)):
        for j in range(i + 1, len(all_positions)):
            dist = np.linalg.norm(all_positions[i] - all_positions[j])
            if dist <= max_distance:
                edge_src += [i, j]
                edge_dst += [j, i]

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def build_networkx_graph(bs_positions: np.ndarray, user_positions: np.ndarray, max_distance: float = 300.0):
 
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
