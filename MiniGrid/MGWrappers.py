import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.core.world_object import Wall, Lava, Goal


class ValidActionsWrapper(gym.Wrapper):
    """
    A Gym environment wrapper that restricts the action space to a subset of valid actions.
    This wrapper maps the agent's chosen discrete action index to a predefined set of valid actions.
    
    :param env: The Gym environment to wrap.
    :type env: gym.Env
    """
    def __init__(self, env):
        super().__init__(env)
        self.valid_actions = [0, 1, 2]
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def step(self, action):
        """
        Executes a step in the environment using the mapped valid action.
        
        :param action: The index of the action in the valid_actions list.
        :type action: int
        :return: A tuple of (observation, reward, terminated, truncated, info) from the environment.
        :rtype: tuple
        """
        return self.env.step(self.valid_actions[action])


class FeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN-based feature extractor for MiniGrid environments.
    Converts image-based observations into a flat feature vector suitable for RL algorithms.
    This is based on the MiniGrid Documentation: https://minigrid.farama.org/content/training/
    
    :param observation_space: The observation space of the environment.
    :type observation_space: gym.spaces.Box
    :param features_dim: Dimensionality of the output feature vector.
    :type features_dim: int
    """
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
        """
        Computes the feature representation of an observation.
        
        :param obs: Observation tensor from the environment.
        :type obs: torch.Tensor
        :return: Extracted feature vector.
        :rtype: torch.Tensor
        """
        return self.linear(self.cnn(obs))

class UnWrappers():
    """
    Utility class providing static methods to unwrap environments and flatten MiniGrid grids.
    """
    @staticmethod
    def unwrap_env(env):
        """
        Recursively unwraps a vectorized or wrapped environment to get the base MiniGrid environment.
        
        :param env: A wrapped or vectorized Gym environment.
        :type env: gym.Env
        :return: The unwrapped base environment.
        :rtype: gym.Env
        """
        if hasattr(env, "envs"):
            env = env.envs[0]

        while hasattr(env, "env"):
            env = env.env

        return env

    @staticmethod
    def flatten_env_grid(vec_env):
        """
        Flattens the grid of a MiniGrid environment into a 1D list of integers representing objects.
        
        Mapping:
            None -> 0
            Wall -> 1
            Lava -> 2
            Goal -> 5
        
        :param vec_env: A (possibly vectorized/wrapped) MiniGrid environment.
        :type vec_env: gym.Env
        :return: Flattened grid as a list of integers.
        :rtype: list[int]
        """
        env = UnWrappers.unwrap_env(vec_env)

        obj_to_id = {None: 0, Wall: 1, Lava: 2, Goal: 5}
        flat_env = []

        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid.get(x, y)
                flat_env.append(obj_to_id.get(type(cell), 0))

        return flat_env