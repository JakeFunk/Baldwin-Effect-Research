import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from MGWrappers import FeaturesExtractor, ValidActionsWrapper


def make_env(env_id):
    """
    Creates and wraps a MiniGrid environment with the required wrappers for training
    a reinforcement learning agent. The environment is wrapped to restrict actions,
    convert observations to image format, monitor episode statistics, and vectorize
    the environment for compatibility with Stable-Baselines3.

    :param env_id: The Gymnasium environment ID used to create the MiniGrid environment
    :type env_id: str
    :return: A fully wrapped vectorized environment ready for training
    :rtype: VecTransposeImage
    """
    def init():
        env = gym.make(env_id)
        env = ValidActionsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env

    return VecTransposeImage(DummyVecEnv([init]))


def make_model(env, verbosity=0):
    """
    Creates a PPO reinforcement learning model configured with a custom CNN feature
    extractor and training hyperparameters suitable for MiniGrid environments.

    The model uses a convolutional neural network policy to process image-based
    observations from the environment.

    :param env: The wrapped vectorized environment used for training
    :type env: VecTransposeImage
    :param verbosity: The verbosity level of the training output
    :type verbosity: int
    :return: A configured PPO reinforcement learning model
    :rtype: PPO
    """
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=FeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        ),
        verbose=verbosity,
        n_steps=256,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
    )


def main():
    model = make_model(make_env("MiniGrid-LavaGapS7-v0"), verbosity=1)
    model.learn(total_timesteps=500_000)
    model.save(f"Data/Model/MiniGrid-LavaGapS7-v0_PPO")


if __name__ == "__main__":
    main()