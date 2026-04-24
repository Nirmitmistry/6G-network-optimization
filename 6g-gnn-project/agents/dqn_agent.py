"""
dqn_agent.py
Improved DQN with:
  - Dueling network architecture (value + advantage streams)
  - Double DQN target computation
  - Prioritised Experience Replay (PER) with importance-sampling weights
  - Soft target network updates (Polyak averaging)
  - Gradient clipping
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SumTree:
    """Binary sum-tree for O(log n) PER sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritisedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100_000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0
        self.epsilon = 1e-5

    def push(self, transition):
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total / batch_size
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        probs = np.array(priorities) / (self.tree.total + 1e-8)
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()
        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority = (abs(err) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        # Dueling aggregation: Q = V + (A - mean(A))
        return v + a - a.mean(dim=-1, keepdim=True)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.action_dim = action_dim
        tr = config["training"]
        self.gamma = tr["gamma"]
        self.batch_size = tr["batch_size"]
        self.epsilon = tr.get("epsilon_start", 1.0)
        self.epsilon_min = tr.get("epsilon_min", 0.02)
        self.epsilon_decay = tr.get("epsilon_decay", 0.997)
        self.tau = tr.get("tau", 0.005)
        hidden = tr.get("hidden_dim", 512)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=tr["lr"], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tr.get("episodes", 800), eta_min=1e-5
        )
        self.memory = PrioritisedReplayBuffer(
            capacity=tr.get("memory_size", 50_000),
            beta_frames=tr.get("episodes", 800) * 200,
        )

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch, idxs, weights = self.memory.sample(self.batch_size)
        if not batch:
            return None

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Double DQN: action selected by policy net, evaluated by target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            targets = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (q_values - targets).detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Weighted Huber loss
        loss = (weights_t * nn.functional.huber_loss(q_values, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft target update
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_target(self):
        """Hard target update (kept for compatibility)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step_scheduler(self):
        self.scheduler.step()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
