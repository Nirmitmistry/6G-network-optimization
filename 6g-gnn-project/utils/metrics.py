"""
metrics.py
Utility functions for evaluating 6G network performance.
"""

import numpy as np


def compute_sinr(signal_power: float, interference: float, noise_power: float = 1e-9) -> float:
    """Signal-to-Interference-plus-Noise Ratio (linear scale)."""
    return signal_power / (interference + noise_power)


def compute_throughput(bandwidth_hz: float, sinr: float) -> float:
    """Shannon capacity in bits/s."""
    return bandwidth_hz * np.log2(1 + sinr)


def compute_spectral_efficiency(throughput: float, bandwidth_hz: float) -> float:
    """Spectral efficiency in bits/s/Hz."""
    return throughput / bandwidth_hz if bandwidth_hz > 0 else 0.0


def compute_fairness_index(throughputs: np.ndarray) -> float:
    """Jain's Fairness Index across users."""
    n = len(throughputs)
    if n == 0:
        return 0.0
    return (np.sum(throughputs) ** 2) / (n * np.sum(throughputs ** 2) + 1e-12)


def compute_avg_latency(queue_lengths: np.ndarray, service_rate: float) -> float:
    """Approximate average latency via Little's Law."""
    return np.mean(queue_lengths) / (service_rate + 1e-12)


def summarize_episode(rewards: list) -> dict:
    """Returns summary stats for a training episode."""
    rewards = np.array(rewards)
    return {
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }
