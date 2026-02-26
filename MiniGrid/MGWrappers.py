import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ValidActionsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.valid_actions = [0, 1, 2]
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def step(self, action):
        return self.env.step(self.valid_actions[action])


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]  # type:ignore
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))
