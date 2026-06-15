import os
import re
import json
import random
import pandas as pd
import math
from collections import Counter, defaultdict
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from MGReplayEnv import ReplayEnv
from MGWrappers import ValidActionsWrapper
from MGWrappers import UnWrappers
from MGGeneticAlgorithm import MGGeneticAlgorithm
from MGPlotting import (
    plot_event_locations,
    plot_origin_gen_timeline,
    plot_persistence_distribution
)
from MGModelCreation import make_env

ENV_ID = "MiniGrid-LavaGapS7-v0"
GENERATIONS = 60

FIG_DIR = "Data/Figures/Assimilation"
OUTPUT_DIR = "Data/Statistics/Assimilation"
MODEL_DIR = "Data/tmpModels"
TRAINING_CSV = f"{OUTPUT_DIR}/training.csv"


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

            generation_rows.append(
                {
                    "generation": gen,
                    "individual": i,
                    "pre_reward": pre_reward,
                    "post_reward": post_reward,
                    "reward_gain": gain,
                    "learning_steps": steps,
                    "before_model": before_path,
                    "after_model": after_path,
                }
            )

            print(f"  Ind {i:02d} | gain={gain:.4f}")

        scored.sort(key=lambda x: x[0], reverse=True)

        for row in generation_rows:
            training_rows.append(row)

        pd.DataFrame(training_rows).to_csv(TRAINING_CSV, index=False)

        ga.evolve(scored)

    return pd.DataFrame(training_rows)


def get_valid_states(flat_env, width, height):
    states = []
    for y in range(height):
        for x in range(width):
            cell = flat_env[y * width + x]
            # If cell is wall or lava
            if cell in (1, 2):
                continue
            for d in range(4):
                states.append((x, y, d))
    return states


def get_action_table(model_path, flat_env, states):
    model = PPO.load(model_path)
    env = load_replay_env(flat_env)
    env.reset()
    base_env = UnWrappers.unwrap_env(env)

    action_table = {}
    for x, y, d in states:
        base_env.agent_pos = (x, y)
        base_env.agent_dir = d
        img = base_env.gen_obs()["image"]
        img = img.transpose(2, 0, 1)[None, ...]
        action, _ = model.predict(img, deterministic=True)
        action_table[(x, y, d)] = int(action[0])

    return action_table


def population_mode_table(action_tables):
    states = action_tables[0].keys()
    return {
        s: Counter(t[s] for t in action_tables).most_common(1)[0][0] for s in states
    }


def build_generation_tables(model_paths, flat_env, states):
    before_tables = [
        get_action_table(p["before"], flat_env, states) for p in model_paths.values()
    ]
    after_tables = [
        get_action_table(p["after"], flat_env, states) for p in model_paths.values()
    ]

    return population_mode_table(before_tables), population_mode_table(after_tables)


def track_assimilation(gen_instinct, gen_learned, min_persist=3):
    gens = sorted(gen_instinct.keys())
    events = []

    for i, g in enumerate(gens[:-1]):
        instinct_g = gen_instinct[g]
        learned_g = gen_learned[g]

        for s, a_inst in instinct_g.items():
            a_learn = learned_g[s]

            if a_inst == a_learn:
                continue

            streak = 0
            for g2 in gens[i + 1 :]:
                if gen_instinct[g2].get(s) == a_learn:
                    streak += 1
                else:
                    break

            events.append(
                {
                    "state": s,
                    "origin_gen": g,
                    "instinct_before": a_inst,
                    "learned_action": a_learn,
                    "persistence": streak,
                    "assimilated": streak >= min_persist,
                }
            )

    return events


def action_marginals(instinct_by_gen):
    counts = Counter()
    total = 0
    for table in instinct_by_gen.values():
        for a in table.values():
            counts[a] += 1
            total += 1
    return {a: c / total for a, c in counts.items()}


def breakdown_by_action(events, marginals):
    by_action = defaultdict(list)
    for e in events:
        by_action[e["learned_action"]].append(e)

    results = {}
    for a, evs in by_action.items():
        n = len(evs)
        n_assim = sum(e["assimilated"] for e in evs)
        results[a] = {
            "n_events": n,
            "n_assimilated": n_assim,
            "rate": n_assim / n if n else 0.0,
            "marginal_freq": marginals.get(a, 0.0),
        }
    return results


def permutation_null(gen_instinct, observed_events, min_persist=1, n_perms=1000, seed=0):
    rng = random.Random(seed)
    gens = sorted(gen_instinct.keys())

    null_rates = []
    for _ in range(n_perms):
        shuffled_gens = gens.copy()
        rng.shuffle(shuffled_gens)
        # map each real generation position -> a randomly chosen table
        shuffled_instinct = {
            g: gen_instinct[shuffled_gens[i]] for i, g in enumerate(gens)
        }

        n_assim = 0
        for e in observed_events:
            s = e["state"]
            a_learn = e["learned_action"]
            g = e["origin_gen"]
            idx = gens.index(g)

            streak = 0
            for g2 in gens[idx + 1 :]:
                if shuffled_instinct[g2].get(s) == a_learn:
                    streak += 1
                else:
                    break

            if streak >= min_persist:
                n_assim += 1

        null_rates.append(n_assim / len(observed_events))

    return null_rates


def get_lava_cells(flat_env, width, height):
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if flat_env[y * width + x] == 2
    ]


def annotate_distance_to_lava(events, lava_cells):
    for e in events:
        x, y, _d = e["state"]
        if lava_cells:
            e["dist_to_lava"] = min(abs(x - lx) + abs(y - ly) for lx, ly in lava_cells)
        else:
            e["dist_to_lava"] = None
    return events


def raw_counts_per_state_action(events):
    grouped = defaultdict(lambda: {"n_events": 0, "n_assimilated": 0,
                                     "gens": []})

    for e in events:
        key = (e["state"], e["learned_action"])
        grouped[key]["n_events"] += 1
        grouped[key]["n_assimilated"] += int(e["assimilated"])
        grouped[key]["gens"].append(e["origin_gen"])

    rows = []
    for (state, action), stats in grouped.items():
        rows.append({
            "state": state,
            "learned_action": action,
            "n_events": stats["n_events"],
            "n_assimilated": stats["n_assimilated"],
            "rate": stats["n_assimilated"] / stats["n_events"],
            "origin_gens": stats["gens"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["n_events", "rate"], ascending=[False, False])


def min_persist_sensitivity(gen_instinct, gen_learned, persist_values=(1, 2, 3, 4, 5)):
    rows = []
    for mp in persist_values:
        events_mp = track_assimilation(gen_instinct, gen_learned, min_persist=mp)
        marginals = action_marginals(gen_instinct)
        breakdown = breakdown_by_action(events_mp, marginals)

        for action, stats in breakdown.items():
            rows.append({
                "min_persist": mp,
                "action": action,
                "n_events": stats["n_events"],
                "n_assimilated": stats["n_assimilated"],
                "rate": stats["rate"],
                "marginal_freq": stats["marginal_freq"],
            })

    return pd.DataFrame(rows)


def split_events_by_origin_gen(events, marginals, threshold=5):
    early = [e for e in events if e["origin_gen"] < threshold]
    late = [e for e in events if e["origin_gen"] >= threshold]

    early_breakdown = breakdown_by_action(early, marginals) if early else {}
    late_breakdown = breakdown_by_action(late, marginals) if late else {}

    rows = []
    for group_name, group_events, group_breakdown in [
        ("early", early, early_breakdown),
        ("late", late, late_breakdown),
    ]:
        for action, stats in group_breakdown.items():
            rows.append({
                "group": group_name,
                "action": action,
                "n_events": stats["n_events"],
                "n_assimilated": stats["n_assimilated"],
                "rate": stats["rate"],
                "marginal_freq": stats["marginal_freq"],
            })

    return pd.DataFrame(rows), early, late
    
    
def permutation_test_for_subset(gen_instinct, events_subset, min_persist=3, n_perms=1000, seed=0):
    if not events_subset:
        return None

    observed_rate = sum(e["assimilated"] for e in events_subset) / len(events_subset)
    null_rates = permutation_null(gen_instinct, events_subset, min_persist=min_persist, n_perms=n_perms, seed=seed)
    p_value = sum(r >= observed_rate for r in null_rates) / len(null_rates)

    return {
        "n_events": len(events_subset),
        "observed_rate": observed_rate,
        "null_mean": sum(null_rates) / len(null_rates),
        "p_value": p_value,
    }
    

def min_persist_sensitivity_by_group(gen_instinct, gen_learned, threshold=5, persist_values=(1, 2, 3, 4, 5)):
    marginals = action_marginals(gen_instinct)
    rows = []

    for mp in persist_values:
        events_mp = track_assimilation(gen_instinct, gen_learned, min_persist=mp)
        early = [e for e in events_mp if e["origin_gen"] < threshold]
        late = [e for e in events_mp if e["origin_gen"] >= threshold]

        for group_name, group_events in [("early", early), ("late", late)]:
            breakdown = breakdown_by_action(group_events, marginals)
            for action, stats in breakdown.items():
                rows.append({
                    "min_persist": mp,
                    "group": group_name,
                    "action": action,
                    "n_events": stats["n_events"],
                    "n_assimilated": stats["n_assimilated"],
                    "rate": stats["rate"],
                    "marginal_freq": stats["marginal_freq"],
                })

    return pd.DataFrame(rows)


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


def main():
    selected_env = [1, 1, 1, 1, 1, 1, 1,
                    1, 0, 0, 2, 0, 0, 1,
                    1, 0, 0, 2, 0, 0, 1,
                    1, 0, 0, 0, 0, 0, 1,
                    1, 0, 0, 2, 0, 0, 1,
                    1, 0, 0, 2, 0, 5, 1,
                    1, 1, 1, 1, 1, 1, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if os.path.exists(TRAINING_CSV):
        training_df = pd.read_csv(TRAINING_CSV)
    else:
        training_df = perform_training()

    models = build_model_dict()

    side = int(math.sqrt(len(selected_env)))
    states = get_valid_states(selected_env, side, side)

    gen_instinct = {}
    gen_learned = {}

    for gen, phases in sorted(models.items()):
        before_paths = phases.get("before", {})
        after_paths = phases.get("after", {})

        common_inds = sorted(set(before_paths) & set(after_paths))
        if not common_inds:
            continue

        before_tables = [
            get_action_table(before_paths[ind], selected_env, states)
            for ind in common_inds
        ]
        after_tables = [
            get_action_table(after_paths[ind], selected_env, states)
            for ind in common_inds
        ]

        gen_instinct[gen] = population_mode_table(before_tables)
        gen_learned[gen] = population_mode_table(after_tables)

    events = track_assimilation(gen_instinct, gen_learned, min_persist=3)
    marginals = action_marginals(gen_instinct)

    # Proximity to lava (annotate before saving)
    lava_cells = get_lava_cells(selected_env, side, side)
    events = annotate_distance_to_lava(events, lava_cells)

    events_df = pd.DataFrame(events)
    events_df.to_csv(f"{OUTPUT_DIR}/assimilation_events.csv", index=False)

    # Per-action breakdown
    breakdown = breakdown_by_action(events, marginals)
    breakdown_df = pd.DataFrame([
        {"action": a, **stats} for a, stats in sorted(breakdown.items())
    ])
    breakdown_df.to_csv(f"{OUTPUT_DIR}/action_breakdown.csv", index=False)

    # Overall permutation null test
    observed_rate = sum(e["assimilated"] for e in events) / len(events)
    null_rates = permutation_null(gen_instinct, events, min_persist=3, n_perms=1000)
    p_value = sum(r >= observed_rate for r in null_rates) / len(null_rates)

    overall_test_df = pd.DataFrame([{
        "n_events": len(events),
        "observed_rate": observed_rate,
        "null_mean": sum(null_rates) / len(null_rates),
        "p_value": p_value,
    }])
    overall_test_df.to_csv(f"{OUTPUT_DIR}/overall_permutation_test.csv", index=False)

    # Assimilation rate by distance to lava
    dist_lava_df = events_df.groupby("dist_to_lava")["assimilated"].mean().reset_index()
    dist_lava_df.to_csv(f"{OUTPUT_DIR}/assimilation_by_dist_to_lava.csv", index=False)

    # Raw counts per (state, action)
    state_action_df = raw_counts_per_state_action(events)
    state_action_df.to_csv(f"{OUTPUT_DIR}/assimilation_by_state_action.csv", index=False)

    # Sensitivity analysis
    sensitivity_df = min_persist_sensitivity(gen_instinct, gen_learned,
                                              persist_values=(1, 2, 3, 4, 5))
    sensitivity_df.to_csv(f"{OUTPUT_DIR}/min_persist_sensitivity.csv", index=False)

    # Split by origin generation
    split_df, early_events, late_events = split_events_by_origin_gen(events, marginals, threshold=5)
    split_df.to_csv(f"{OUTPUT_DIR}/events_by_origin_split.csv", index=False)

    # Location plots: late vs early assimilated events
    high_persist_late = [e for e in late_events if e["assimilated"]]
    plot_event_locations(
        high_persist_late, selected_env, side,
        out_path=f"{FIG_DIR}/MiniGrid_Late_Assimilated_Locations.png",
        title="Assimilated states (origin_gen >= 5)"
    )

    high_persist_early = [e for e in early_events if e["assimilated"]]
    plot_event_locations(
        high_persist_early, selected_env, side,
        out_path=f"{FIG_DIR}/MiniGrid_Early_Assimilated_Locations.png",
        title="Assimilated states (origin_gen < 5)"
    )

    # Late-group permutation tests, per action
    late_action_tests = []
    for action in (0, 1, 2):
        late_action_events = [e for e in late_events if e["learned_action"] == action]
        result = permutation_test_for_subset(
            gen_instinct, late_action_events, min_persist=3, n_perms=1000
        )
        if result:
            late_action_tests.append({"action": action, **result})

    pd.DataFrame(late_action_tests).to_csv(f"{OUTPUT_DIR}/late_group_permutation_tests.csv", index=False)

    # Sensitivity analysis split by early/late
    group_sensitivity_df = min_persist_sensitivity_by_group(gen_instinct, gen_learned, threshold=5, persist_values=(1, 2, 3, 4, 5))
    group_sensitivity_df.to_csv(f"{OUTPUT_DIR}/min_persist_sensitivity_by_group.csv", index=False)

    # Persistence distribution and origin-gen timeline
    plot_persistence_distribution(events, out_path=f"{FIG_DIR}/MiniGrid_Persistence_Distribution.png")
    plot_origin_gen_timeline(events, out_path=f"{FIG_DIR}/MiniGrid_Assimilation_Origin_Gen_Timeline.png", threshold=5)


if __name__ == "__main__":
    main()
    
# Why focus on action 0 so much? generalize it for all actions? Make a heatmap for that?
# 