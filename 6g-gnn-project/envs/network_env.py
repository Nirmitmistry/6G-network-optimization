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
        self.max_bw = net["max_bandwidth_mhz"] * 1e6
        self.max_edge_dist = net.get("max_edge_distance", self.area_size * 0.3)

        self.action_space = spaces.Discrete(self.num_users * self.num_bs)

        obs_dim = (self.num_bs + self.num_users) * 2 + self.num_users
        self.observation_space = spaces.Box(
            low=0.0, high=float(self.area_size), shape=(obs_dim,), dtype=np.float32
        )

        # Transmit power per BS (dBm → linear W), path-loss exponent
        self.tx_power = 1.0       # normalised
        self.noise_power = 1e-9
        self.path_loss_exp = 3.5  # typical urban mmWave

        self.reset()

    def reset(self):
        # Place BSs on a rough grid + small jitter for diversity
        grid_n = int(np.ceil(np.sqrt(self.num_bs)))
        xs = np.linspace(0, self.area_size, grid_n + 2)[1:-1]
        ys = np.linspace(0, self.area_size, grid_n + 2)[1:-1]
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)[: self.num_bs]
        jitter = np.random.uniform(-self.area_size * 0.05, self.area_size * 0.05, grid_pts.shape)
        self.bs_positions = np.clip(grid_pts + jitter, 0, self.area_size).astype(np.float32)

        self.user_positions = np.random.uniform(0, self.area_size, (self.num_users, 2)).astype(np.float32)
        self.allocations = np.zeros(self.num_users, dtype=np.int32)
        self.bandwidth_alloc = np.ones(self.num_users) * (self.max_bw / self.num_users)
        self.demands = np.random.uniform(1e6, 50e6, self.num_users)  # 1–50 Mbps
        self.step_count = 0
        self._prev_reward = 0.0
        return self._get_obs()

    def step(self, action: int):
        user_id = int(action) // self.num_bs
        bs_id = int(action) % self.num_bs
        self.allocations[user_id] = bs_id

        # Rebalance bandwidth: equal share per BS load
        bs_loads = np.bincount(self.allocations, minlength=self.num_bs)
        for u in range(self.num_users):
            b = self.allocations[u]
            load = max(bs_loads[b], 1)
            self.bandwidth_alloc[u] = self.max_bw / load

        throughputs = self._compute_throughputs()
        fairness = compute_fairness_index(throughputs)
        demand_sat = np.mean(np.minimum(throughputs / (self.demands + 1e-9), 1.0))

        # Shaped reward: weighted sum + small improvement bonus
        reward = 0.6 * demand_sat + 0.4 * fairness
        delta = reward - self._prev_reward
        shaped_reward = reward + 0.1 * delta
        self._prev_reward = reward

        self.step_count += 1
        done = self.step_count >= 200

        # User mobility: random walk with slight drift toward centre
        centre = np.array([self.area_size / 2, self.area_size / 2])
        drift = (centre - self.user_positions) * 0.001
        self.user_positions += drift + np.random.uniform(-10, 10, self.user_positions.shape)
        self.user_positions = np.clip(self.user_positions, 0, self.area_size)

        return self._get_obs(), float(shaped_reward), done, {"raw_reward": reward}

    def _get_obs(self):
        return np.concatenate([
            self.bs_positions.flatten() / self.area_size,
            self.user_positions.flatten() / self.area_size,
            self.allocations.astype(np.float32) / self.num_bs,
        ]).astype(np.float32)

    def _compute_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for u in range(self.num_users):
            bs = self.allocations[u]
            dist = np.linalg.norm(self.user_positions[u] - self.bs_positions[bs]) + 1e-3
            signal = self.tx_power / (dist ** self.path_loss_exp)
            interference = sum(
                self.tx_power / (np.linalg.norm(self.user_positions[u] - self.bs_positions[self.allocations[v]]) ** self.path_loss_exp + 1e-3)
                for v in range(self.num_users) if v != u
            )
            sinr = compute_sinr(signal, interference, self.noise_power)
            throughputs[u] = compute_throughput(self.bandwidth_alloc[u], sinr)
        return throughputs

    def render(self, mode="human"):
        print(f"Step {self.step_count} | Allocations: {self.allocations}")

    def get_graph(self):
        return build_network_graph(
            self.bs_positions,
            self.user_positions,
            allocations=self.allocations,
            max_distance=self.max_edge_dist,
            area_size=self.area_size,
        )

    def get_gnn_state(self, encoder, device):
        graph = self.get_graph().to(device)
        with torch.no_grad():
            graph_emb, node_emb = encoder(graph)
        alloc = torch.tensor(
            self.allocations.astype(np.float32) / self.num_bs, device=device
        )
        state = torch.cat([
            graph_emb.squeeze(0),
            node_emb.flatten(),
            alloc,
        ], dim=0)
        return state.cpu().numpy()
