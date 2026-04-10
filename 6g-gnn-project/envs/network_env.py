import gym
import numpy as np
import torch
from gym import spaces
from utils.graph_builder import build_network_graph
from utils.metrics import compute_sinr, compute_throughput, compute_fairness_index


class NetworkEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()
        net = config["network"]
        self.num_bs = net["num_base_stations"]
        self.num_users = net["num_users"]
        self.area_size = net["area_size"]
        self.max_bw = net["max_bandwidth_mhz"] * 1e6  # Hz


        self.action_space = spaces.Discrete(self.num_users * self.num_bs)

        obs_dim = (self.num_bs + self.num_users) * 2 + self.num_users
        self.observation_space = spaces.Box(
            low=0.0, high=float(self.area_size), shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.bs_positions = np.random.uniform(0, self.area_size, (self.num_bs, 2))
        self.user_positions = np.random.uniform(0, self.area_size, (self.num_users, 2))
        self.allocations = np.zeros(self.num_users, dtype=np.int32)   # bs index per user
        self.bandwidth_alloc = np.ones(self.num_users) * (self.max_bw / self.num_users)
        self.demands = np.random.uniform(1e6, 10e6, self.num_users)   # 1–10 Mbps demand
        self.step_count = 0
        return self._get_obs()

    def step(self, action: int):
        user_id = action // self.num_bs
        bs_id = action % self.num_bs

        self.allocations[user_id] = bs_id

   
        throughputs = self._compute_throughputs()
        fairness = compute_fairness_index(throughputs)
        demand_satisfaction = np.mean(np.minimum(throughputs / (self.demands + 1e-9), 1.0))
        reward = 0.5 * demand_satisfaction + 0.5 * fairness

        self.step_count += 1
        done = self.step_count >= 200

        self.user_positions += np.random.uniform(-5, 5, self.user_positions.shape)
        self.user_positions = np.clip(self.user_positions, 0, self.area_size)

        return self._get_obs(), reward, done, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        obs = np.concatenate([
            self.bs_positions.flatten(),
            self.user_positions.flatten(),
            self.allocations.astype(np.float32) / self.num_bs,
        ]).astype(np.float32)
        return obs

    def _compute_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for u in range(self.num_users):
            bs = self.allocations[u]
            dist = np.linalg.norm(self.user_positions[u] - self.bs_positions[bs]) + 1e-3
            signal = 1.0 / (dist ** 2)
            interference = sum(
                1.0 / (np.linalg.norm(self.user_positions[u] - self.bs_positions[self.allocations[v]]) ** 2 + 1e-3)
                for v in range(self.num_users) if v != u
            )
            sinr = compute_sinr(signal, interference)
            throughputs[u] = compute_throughput(self.bandwidth_alloc[u], sinr)
        return throughputs

    def render(self, mode="human"):
        print(f"Step {self.step_count} | Allocations: {self.allocations}")

    def get_graph(self):
        return build_network_graph(self.bs_positions, self.user_positions)

    def get_gnn_state(self, encoder, device):
        """Run current topology through GraphEncoder and return a flat numpy state."""
        graph = self.get_graph().to(device)
        with torch.no_grad():
            graph_emb, node_emb = encoder(graph)
        # graph_emb: (1, out_channels), node_emb: (num_nodes, out_channels)
        # Concatenate global embedding + flattened node embeddings + allocation ratios
        alloc = torch.tensor(
            self.allocations.astype(np.float32) / self.num_bs, device=device
        )
        state = torch.cat([
            graph_emb.squeeze(0),          # (32,)
            node_emb.flatten(),            # (num_nodes * 32,)
            alloc,                         # (num_users,)
        ], dim=0)
        return state.cpu().numpy()
