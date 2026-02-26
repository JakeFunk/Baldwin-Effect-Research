import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from MGWrappers import FeaturesExtractor, ValidActionsWrapper


def make_env(env_id):
    def init():
        env = gym.make(env_id)
        env = ValidActionsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env

    return VecTransposeImage(DummyVecEnv([init]))


def make_model(env, verbosity=0):
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
    model.save(f"MiniGrid-LavaGapS7-v0_CNN")


if __name__ == "__main__":
    main()
