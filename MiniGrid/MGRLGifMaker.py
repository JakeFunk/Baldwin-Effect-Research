import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper
from MGWrappers import ValidActionsWrapper


MODEL_PATH = "Data/Model/MiniGrid-LavaGapS7-v0_PPO.zip"
ENV_ID = "MiniGrid-LavaGapS7-v0"
NUM_EPISODES = 10
OUTPUT_GIF = "Data/Figures/RL_Run.gif"

def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = ValidActionsWrapper(env)
    env = ImgObsWrapper(env)
    return env

def main():
    model = PPO.load(MODEL_PATH)

    frames = []

    for ep in range(NUM_EPISODES):
        env = make_env()
        obs, _ = env.reset()
        done = False

        while not done:
            # Render frame
            frame = env.render()
            frames.append(frame)

            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()

        if ep < NUM_EPISODES - 1:
            frames.append(np.zeros_like(frames[-1]))

    imageio.mimsave(
        OUTPUT_GIF,
        frames,
        fps=10,
        loop=0 # infinite loop
    )
    print(f"Saved GIF to {OUTPUT_GIF}")


if __name__ == "__main__":
    main()