"""
main.py
Entry point — runs GNN pre-training followed by DRL training.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from training.train_gnn import train as train_gnn
from training.train_rl import train as train_rl


if __name__ == "__main__":
    print("=" * 50)
    print("Step 1: Pre-training GNN encoder...")
    print("=" * 50)
    train_gnn(config_path="configs/config.yaml")

    print("\n" + "=" * 50)
    print("Step 2: Training DQN agent...")
    print("=" * 50)
    train_rl(config_path="configs/config.yaml")

    print("\nAll done. Check results/ for saved models and plots.")
