"""
ppo_agent.py
Proximal Policy Optimization agent using stable-baselines3.
Wraps the NetworkEnv for easy PPO training.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class PPOAgent:
    def __init__(self, env, config: dict):
        check_env(env, warn=True)
        self.env = env
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config["training"]["lr"],
            gamma=config["training"]["gamma"],
            batch_size=config["training"]["batch_size"],
            verbose=1,
        )

    def train(self, total_timesteps: int = 100_000):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)
