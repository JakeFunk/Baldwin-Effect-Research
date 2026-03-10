import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from MGReplayEnv import ReplayEnv
from MGWrappers import ValidActionsWrapper
from MGGeneticAlgorithm import MGGeneticAlgorithm


ENV_ID = "MiniGrid-LavaGapS7-v0"
GENERATIONS = 60
MAX_STEPS = 500


def wrap_env(env):
    """
    Wraps a MiniGrid environment with custom wrappers for action restriction, image observations, 
    monitoring, and vectorization.

    :param env: The raw Gym MiniGrid environment.
    :type env: gym.Env
    :return: A wrapped and vectorized environment ready for Stable-Baselines3.
    :rtype: stable_baselines3.common.vec_env.VecEnv
    """
    env = ValidActionsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return VecTransposeImage(DummyVecEnv([lambda: env]))


def run_expert(expert, env):
    """
    Runs a pre-trained expert model on a given environment and returns the reward obtained.

    :param expert: Pre-trained RL agent (e.g., PPO) to evaluate.
    :type expert: stable_baselines3.common.base_class.BaseAlgorithm
    :param env: Wrapped environment for evaluation.
    :type env: stable_baselines3.common.vec_env.VecEnv
    :return: Final reward obtained by the expert in the environment.
    :rtype: float
    """
    obs = env.reset()
    final_reward = 0.0

    for _ in range(MAX_STEPS):
        action, _ = expert.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        final_reward = float(reward[0])

        if done[0]:
            break

    return final_reward


def collect_visits(model, env):
    """
    Collects the number of visits each grid cell receives when a model interacts with the environment.

    :param model: RL agent (e.g., PPO or genetic algorithm agent) whose behavior will be tracked.
    :type model: object
    :param env: Wrapped MiniGrid environment.
    :type env: stable_baselines3.common.vec_env.VecEnv
    :return: 2D array representing the visit counts for each cell in the environment grid.
    :rtype: np.ndarray
    """
    base_env = env.envs[0].unwrapped
    width = base_env.width
    height = base_env.height

    visits = np.zeros((height, width))
    obs = env.reset()

    for _ in range(MAX_STEPS):
        x, y = base_env.agent_pos
        visits[y, x] += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if done[0]:
            break

    return visits


def draw_heatmap(env, visits, generation):
    """
    Draws and saves a heatmap of agent visits over a MiniGrid environment grid.

    The intensity of each cell is proportional to the logarithm of the number of visits.

    :param env: Wrapped MiniGrid environment.
    :type env: stable_baselines3.common.vec_env.VecEnv
    :param visits: 2D array of visit counts per grid cell.
    :type visits: np.ndarray
    :param generation: Current generation number (used in filename for saving heatmap).
    :type generation: int
    """
    base_env = env.envs[0].unwrapped
    grid_size = base_env.height

    img = base_env.grid.render(
        tile_size=64,
        agent_pos=None,
        agent_dir=None
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(img)

    max_visits = visits.max()
    if max_visits == 0:
        max_visits = 1

    cell_w = img.shape[1] / grid_size
    cell_h = img.shape[0] / grid_size

    for y in range(grid_size):
        for x in range(grid_size):

            intensity = np.log1p(visits[y, x]) / np.log1p(max_visits)

            if intensity > 0:
                rect = patches.Rectangle(
                    (x * cell_w, y * cell_h),
                    cell_w,
                    cell_h,
                    linewidth=0,
                    facecolor=(1, 0, 0, intensity * 0.6),
                )
                ax.add_patch(rect)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        f"Data/Heatmaps/heatmap_gen_{generation:.2d}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


def main():
    # Load data from the previously trained data
    df = pd.read_csv("Data/Statistics/minigrid_stats.csv")
    environments = df["agreement_env_flattened"]

    # Select a random environment and create it
    rand_idx = random.randint(0, len(environments) - 1)
    selected_env = json.loads(environments[rand_idx])
    env = ReplayEnv(selected_env, render_mode="rgb_array")
    env = wrap_env(env)

    # Load the agent, and run through the environment with it
    expert = PPO.load(f"Data/Model/MiniGrid-LavaGapS7-v0_PPO")
    optimal_reward = run_expert(expert, env)

    print(f"Operating on Environment: {rand_idx}")
    print(selected_env)
    print(f"Optimal Reward: {optimal_reward:.2f}")

    # Perform the training on the environment and obtain the heat map
    ga = MGGeneticAlgorithm(ENV_ID)
    for gen in range(GENERATIONS):
        scored = []
        pre_rewards = []
        post_rewards = []

        for i, ind in enumerate(ga.population):
            pre_reward = ga.rollout_reward(ind, env)
            ind.learn(total_timesteps=ga.learning_steps(gen))
            post_reward = ga.rollout_reward(ind, env)

            gain = post_reward - pre_reward
            pre_rewards.append(pre_reward)
            post_rewards.append(post_reward)
            scored.append((gain, ind))

        print(f"Generation {gen} | Pre: {np.mean(pre_rewards)} | Post: {np.mean(post_rewards)}")

        best_agent = max(scored, key=lambda x: x[0])[1]
        visits = collect_visits(best_agent, env)
        draw_heatmap(env, visits, gen)

        ga.evolve(scored)


if __name__ == "__main__":
    main()