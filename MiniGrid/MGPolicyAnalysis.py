import json
import random
import os
import re
import pandas as pd
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from MGReplayEnv import ReplayEnv
from MGWrappers import ValidActionsWrapper
from MGGeneticAlgorithm import MGGeneticAlgorithm
from MGModelCreation import make_env
from MGPlotting import plot_policy_analysis


ENV_ID = "MiniGrid-LavaGapS7-v0"
GENERATIONS = 60
N_TEST_ENVS = 10


def wrap_env(env):
    env = ValidActionsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return VecTransposeImage(DummyVecEnv([lambda: env]))


def load_replay_env(flat_env):
    env = ReplayEnv(flat_env, render_mode="rgb_array")
    return wrap_env(env)


def select_models():
    all_models = os.listdir("Data/tmpModels")
    selected_models = []

    for model in all_models:
        match = re.match(r"gen_(\d+)_ind_\d+_(before|after)\.zip", model)
        if match:
            generation = int(match.group(1))

            if generation % 10 == 0 or generation == 59:
                selected_models.append(model)

    selected_models.sort(
        key=lambda x: (
            int(re.search(r"gen_(\d+)", x).group(1)),
            0 if "before" in x else 1
        )
    )

    return selected_models


def build_gen_pairs():
    selected_models = select_models()
    
    grouped = {}
    
    for model in selected_models:
        gen = int(re.search(r"gen_(\d+)", model).group(1))
        
        if gen not in grouped:
            grouped[gen] = {}
            
        if "before" in model:
            grouped[gen]["before"] = os.path.join("Data/tmpModels", model)
        else:
            grouped[gen]["after"] = os.path.join("Data/tmpModels", model)
            
    return grouped


def compare_models(model_a_path, model_b_path, env):
    obs = env.reset()
    done = [False]

    same = 0
    total = 0
    
    model_a = PPO.load(model_a_path)
    model_b = PPO.load(model_b_path)

    while not done[0]:
        act_a, _ = model_a.predict(obs, deterministic=True)
        act_b, _ = model_b.predict(obs, deterministic=True)

        if int(act_a[0]) == int(act_b[0]):
            same += 1

        total += 1

        obs, _, done, _ = env.step(act_a)

    return 1.0 if total == 0 else same / total


def perform_analysis(selected_envs):
    selected_models = build_gen_pairs()
    
    final_gen = max(selected_models.keys())
    final_model = selected_models[final_gen]["after"]
    
    history = []
    
    for gen in sorted(selected_models.keys()):
        for phase in ["before", "after"]:
            compare_path = selected_models[gen][phase]
            
            agreements = []
            
            for env_num, flat_env in enumerate(selected_envs):
                eval_env = load_replay_env(flat_env)
                
                score = compare_models(final_model, compare_path, eval_env)
                
                agreements.append(score)
                
                history.append({
                    "reference_gen": final_gen,
                    "target_gen": gen,
                    "target_phase": phase,
                    "env_num": env_num,
                    "agreement": score
                })
                
    return history


def perform_training():
    ga = MGGeneticAlgorithm(ENV_ID)

    best_models = []
    best_rewards = []

    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen}")

        scored = []
        all_saved_files = []

        for i, ind in enumerate(ga.population):
            env = make_env(ENV_ID)
            
            save_before_path = f"Data/tmpModels/gen_{gen}_ind_{i}_before.zip"
            save_after_path = f"Data/tmpModels/gen_{gen}_ind_{i}_after.zip"
            
            all_saved_files.append(save_before_path)
            all_saved_files.append(save_after_path)

            ind.save(save_before_path)

            pre_reward = ga.rollout_reward(ind, env)
            ind.learn(total_timesteps=ga.learning_steps(gen))
            post_reward = ga.rollout_reward(ind, env)
            gain = post_reward - pre_reward

            ind.save(save_after_path)

            scored.append((gain, save_before_path, save_after_path, ind))
            print(f" Ind {i:02d} | Gain {gain:.4f}")

        scored.sort(key=lambda x: x[0], reverse=True)

        best_gain, best_before, best_after, best_model = scored[0]

        best_models.append((best_before, best_after))
        best_rewards.append(best_gain)
        
        keep_files = {best_before, best_after}
        for file in all_saved_files:
            if file not in keep_files:
                if os.path.exists(file):
                    os.remove(file)

        ga.evolve([(g, ind) for g, _, _, ind in scored])
        

def main():
    # Obtain all environments created in the initial training set.
    df = pd.read_csv("Data/Statistics/minigrid_stats.csv")
    all_envs = df["agreement_env_flattened"].tolist()
    
    # Randomly select N_TEST_ENVS number of environments.
    selected_indices = random.sample(
        range(len(all_envs)),
        min(N_TEST_ENVS, len(all_envs))
    )
    selected_envs = [json.loads(all_envs[i]) for i in selected_indices]

    # This creates all of the models in the tmpModels folder
    perform_training()
    
    history = perform_analysis(selected_envs)

    df_out = pd.DataFrame(history)
    df_out.to_csv("Data/Statistics/minigrid_policy_analysis.csv", index=False)
    
    plot_policy_analysis("Data/Statistics/minigrid_policy_analysis.csv")


if __name__ == "__main__":
    main()