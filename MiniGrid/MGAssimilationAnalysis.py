import os
import re
import json
import random
import pandas as pd
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from MGReplayEnv import ReplayEnv
from MGWrappers import ValidActionsWrapper
from MGGeneticAlgorithm import MGGeneticAlgorithm
from MGPlotting import plot_assimilation_stats
from MGModelCreation import make_env


ENV_ID = "MiniGrid-LavaGapS7-v0"
GENERATIONS = 60
N_TEST_ENVS = 10
DATA_DIR = "Data/Statistics/Assimilation"
MODEL_DIR = "Data/tmpModels"
TRAINING_CSV = f"{DATA_DIR}/training.csv"
RAW_ACTIONS_CSV = f"{DATA_DIR}/actions.csv"
MODES_CSV = f"{DATA_DIR}/modes.csv"
EVENTS_CSV = f"{DATA_DIR}/events.csv"
SUMMARY_CSV = f"{DATA_DIR}/summary.csv"


def wrap_env(env):
    env = ValidActionsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return VecTransposeImage(DummyVecEnv([lambda: env]))


def load_replay_env(flat_env):
    return wrap_env(ReplayEnv(flat_env, render_mode="rgb_array"))


def perform_training():
    ga = MGGeneticAlgorithm(ENV_ID)

    training_rows = []

    for gen in range(GENERATIONS):

        print(f"Generation {gen}:")

        scored = []
        generation_rows = []

        steps = ga.learning_steps(gen)

        for i, ind in enumerate(ga.population):

            env = make_env(ENV_ID)

            before_path = f"{MODEL_DIR}/gen_{gen}_ind_{i}_before.zip"
            after_path = f"{MODEL_DIR}/gen_{gen}_ind_{i}_after.zip"

            ind.save(before_path)

            pre_reward = ga.rollout_reward(ind, env)

            ind.learn(total_timesteps=steps)

            post_reward = ga.rollout_reward(ind, env)

            gain = post_reward - pre_reward

            ind.save(after_path)

            scored.append((gain, ind))

            generation_rows.append({

                "generation": gen,
                "individual": i,

                "pre_reward": pre_reward,
                "post_reward": post_reward,
                "reward_gain": gain,

                "learning_steps": steps,

                "before_model": before_path,
                "after_model": after_path
            })

            print(f"  Ind {i:02d} | gain={gain:.4f}")

        # rank individuals
        scored.sort(key=lambda x: x[0], reverse=True)
        
        for row in generation_rows:
            training_rows.append(row)

        pd.DataFrame(training_rows).to_csv(TRAINING_CSV, index=False)

        ga.evolve(scored)

    return pd.DataFrame(training_rows)


def wrap_replay(flat_env):
    return wrap_env(ReplayEnv(flat_env, render_mode="rgb_array"))


def collect_states(envs):
    states = {}
    unique = []

    for flat_env in envs:

        env = wrap_replay(flat_env)

        obs = env.reset()
        done = [False]

        while not done[0]:

            key = obs.tobytes()

            if key not in states:
                states[key] = True
                unique.append(obs.copy())

            action = [env.action_space.sample()]
            obs, _, done, _ = env.step(action)

    print(f"Collected {len(unique)} unique states")
    return unique


def build_model_dict():
    models = {}

    pattern = r"gen_(\d+)_ind_(\d+)_(before|after)\.zip"

    for f in os.listdir(MODEL_DIR):

        m = re.match(pattern, f)
        if not m:
            continue

        gen = int(m.group(1))
        ind = int(m.group(2))
        phase = m.group(3)

        models.setdefault(gen, {}).setdefault(phase, {})
        models[gen][phase][ind] = os.path.join(MODEL_DIR, f)

    return models


def save_policy_actions(models, states):
    rows = []

    for gen in sorted(models):
        for phase in ["before", "after"]:

            if phase not in models[gen]:
                continue

            for ind, path in models[gen][phase].items():

                model = PPO.load(path)

                for state_id, state in enumerate(states):

                    action, _ = model.predict(state, deterministic=True)

                    rows.append({

                        "generation": gen,
                        "phase": phase,
                        "individual": ind,
                        "state_id": state_id,
                        "action": int(action[0])

                    })

    df = pd.DataFrame(rows)
    df.to_csv(RAW_ACTIONS_CSV, index=False)

    return df


def compute_modes(df):
    rows = []

    grouped = df.groupby(["generation", "phase", "state_id"])

    for (gen, phase, state), group in grouped:

        counts = group["action"].value_counts()

        rows.append({

            "generation": gen,
            "phase": phase,
            "state_id": state,

            "mode_action": counts.index[0],
            "mode_count": counts.iloc[0],
            "population_size": len(group),
            "agreement": counts.iloc[0] / len(group)
        })

    modes = pd.DataFrame(rows)
    modes.to_csv(MODES_CSV, index=False)

    return modes


def compute_assimilation(modes):
    events = []

    generations = sorted(modes["generation"].unique())
    states = sorted(modes["state_id"].unique())

    for state in states:

        before = modes[modes.phase == "before"]
        after = modes[modes.phase == "after"]

        before = before[before.state_id == state].set_index("generation")
        after = after[after.state_id == state].set_index("generation")

        for gen in generations:

            if gen not in before.index or gen not in after.index:
                continue

            instinct = before.loc[gen, "mode_action"]
            learned = after.loc[gen, "mode_action"]

            if instinct == learned:
                continue

            assimilated_gen = None

            for future in generations:

                if future <= gen:
                    continue

                if future not in before.index:
                    continue

                if before.loc[future, "mode_action"] == learned:

                    persistent = all(
                        before.loc[g, "mode_action"] == learned
                        for g in generations
                        if g >= future and g in before.index
                    )

                    assimilated_gen = future

                    events.append({

                        "state_id": state,
                        "learned_generation": gen,
                        "learned_action": learned,
                        "instinct_action": instinct,
                        "assimilation_generation": future,
                        "lag": future - gen,
                        "persistent": persistent
                    })

                    break

            if assimilated_gen is None:

                events.append({

                    "state_id": state,
                    "learned_generation": gen,
                    "learned_action": learned,
                    "instinct_action": instinct,
                    "assimilation_generation": None,
                    "lag": None,
                    "persistent": False
                })

    df = pd.DataFrame(events)
    df.to_csv(EVENTS_CSV, index=False)

    return df


def save_summary(events):
    total = len(events)
    assimilated = events[events.assimilation_generation.notna()]
    persistent = events[events.persistent]

    summary = pd.DataFrame([{

        "total_events": total,
        "assimilated_events": len(assimilated),

        "assimilation_rate":
            len(assimilated) / total if total else 0,

        "persistent_events": len(persistent),
        "persistent_rate":
            len(persistent) / total if total else 0,

        "mean_lag": assimilated["lag"].mean(),
        "median_lag": assimilated["lag"].median(),
        "max_lag": assimilated["lag"].max(),

        "num_states": events["state_id"].nunique()

    }])

    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary)

    return summary


def main():
    df = pd.read_csv("Data/Statistics/minigrid_stats.csv")

    envs = [
        json.loads(x)
        for x in df["agreement_env_flattened"]
    ]

    selected = random.sample(envs, min(N_TEST_ENVS, len(envs)))

    training_df = perform_training()
    states = collect_states(selected)
    models = build_model_dict()
    actions = save_policy_actions(models, states)
    modes = compute_modes(actions)
    events = compute_assimilation(modes)
    save_summary(events)
    
    plot_assimilation_stats()


if __name__ == "__main__":
    main()