import minigrid
import torch
import copy
import random
import csv
import json
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from MGModelCreation import make_env, make_model
from stable_baselines3 import PPO


ENV_ID = "MiniGrid-LavaGapS7-v0"
POP_SIZE = 20
GENERATIONS = 60
ELITES = 4
MUTATION_STD = 0.02
LEARNING_STEPS = 25_000
MAX_STEPS = 500


class MGGeneticAlgorithm:
    def __init__(self, env_id: str, pop_size: int = POP_SIZE, mutation_std: float = MUTATION_STD):
        self.env_id = env_id
        self.pop_size = pop_size
        self.mutation_std = mutation_std
        self.population = [make_model(make_env(self.env_id)) for _ in range(self.pop_size)]

    def mutate_weights(self, model, rate=0.1):
        for p in model.policy.parameters():
            mask = torch.rand_like(p) < rate
            p.data += mask * torch.randn_like(p) * self.mutation_std

    def evolve(self, scored):
        # GA Selection.
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in scored[:ELITES]]

        # GA Reproduction.
        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            parent = random.choice(elites)
            child = make_model(make_env(ENV_ID))
            child.policy.load_state_dict(copy.deepcopy(parent.policy.state_dict()))

            self.mutate_weights(child)
            new_pop.append(child)

        self.population = new_pop

    def rollout_reward(self, model, env):
        # Ensure the environment is set to a start state.
        obs = env.reset()

        for _ in range(MAX_STEPS):
            # Predict the best action, then act in the environment.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            # Done can be either a goal or lava.
            if done[0]:
                break

        return reward  # type: ignore

    def agreement(self, agent, expert, env):
        obs = env.reset()
        matches = 0
        steps = 0

        expert_actions = []
        agent_actions = []

        for _ in range(MAX_STEPS):
            # Perform predictions on both the expert's model and the individual agents model.
            agent_action, _ = agent.predict(obs, deterministic=True)
            expert_action, _ = expert.predict(obs, deterministic=True)

            # If the agent and expert chose the same action, agreeability goes up.
            matches += int(agent_action[0] == expert_action[0])
            steps += 1

            # Tracking actions taken by both agents.
            agent_actions.append(int(agent_action[0]))
            expert_actions.append(int(expert_action[0]))

            # Perform the expert's action (this leads to high variance of agreement in early generations)
            obs, _, done, _ = env.step(expert_action)

            if done[0]:
                break

        return matches / steps if steps else 0.0, expert_actions, agent_actions, obs[0].flatten().tolist()

    def policy_entropy(self, model, env):
        # Ensure the environment is set to a start state.
        obs = env.reset()

        # Track model entropy
        entropies = []

        for _ in range(MAX_STEPS):
            # Obtain entropy from policy.
            obs_tensor = torch.as_tensor(obs).to(model.device)

            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
                entropy = dist.entropy()

            entropies.append(entropy.mean().item())

            # Predict and act in the environment
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            if done[0]:
                break

        return float(np.mean(entropies)) if entropies else 0.0


def plot(avg_pre_rewards, avg_post_rewards, avg_gains, avg_agreements, avg_entropies):
    generations = range(len(avg_pre_rewards))

    # Pre-learning reward
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_pre_rewards, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Pre-Learn Reward")
    plt.title("Average Pre-Learning Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Post-learning reward
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_post_rewards, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Post-Learn Reward")
    plt.title("Average Post-Learning Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Learning gain
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_gains, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Learning Gain")
    plt.title("Average Learning Gain")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Agreement
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_agreements, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Agreement")
    plt.title("Average Agreement with Expert")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Entropy
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_entropies, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Policy Entropy")
    plt.title("Average Pre-Learning Policy Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def learning_steps(gen: int):
    return int(LEARNING_STEPS * (1 - gen / GENERATIONS))


def start_file_tracking():
    csv_file_avg = open("ppo_baldwin_avgs.csv", "w", newline="")
    avg_writer = csv.writer(csv_file_avg)
    avg_writer.writerow(
        [
            "generation",
            "avg_pre_learn_reward",
            "avg_post_learn_reward",
            "avg_gain",
            "avg_agreement",
            "avg_entropy",
        ]
    )

    csv_file_gen = open("ppo_baldwin_stats.csv", "w", newline="")
    gen_writer = csv.writer(csv_file_gen)
    gen_writer.writerow(
        [
            "generation",
            "individual",
            "pre_learn_reward",
            "post_learn_reward",
            "agreement",
            "entropy",
            "agreement_agent_actions",
            "agreement_expert_actions",
            "agreement_env_flattened",
        ]
    )

    return csv_file_avg, csv_file_gen, avg_writer, gen_writer


def main():
    ga = MGGeneticAlgorithm(ENV_ID)
    expert = PPO.load(f"{ENV_ID}_CNN")
    csv_file_avg, csv_file_gen, avg_writer, gen_writer = start_file_tracking()

    # Variables for plotting the averages of each generation.
    avg_pre_rewards = []
    avg_post_rewards = []
    avg_gains = []
    avg_agreements = []
    avg_entropies = []

    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen}:")
        # Store the rewards of this generation. Perform selection on this list.
        scored = []

        # Stores averages for plotting.
        gen_pre_rewards = []
        gen_post_rewards = []
        gen_gains = []
        gen_agreements = []
        gen_entropies = []

        for i, ind in enumerate(ga.population):
            # Generate a new environment for each individual.
            env = make_env(ENV_ID)

            # Perform entropy calculation and agreement evaluation.
            entropy = ga.policy_entropy(ind, env)
            agreement, agreement_expert_actions, agreement_agent_actions, agreement_env_flat = ga.agreement(ind, expert, env)

            # Proceed with reinforement learning.
            pre_reward = ga.rollout_reward(ind, env)
            ind.learn(total_timesteps=learning_steps(gen))

            # Post-learning evaluation.
            post_reward = ga.rollout_reward(ind, env)

            # Append data to local lists for later input into the overall avg tracking.
            gen_pre_rewards.append(pre_reward)
            gen_post_rewards.append(post_reward)
            gen_gains.append(post_reward - pre_reward)
            gen_agreements.append(agreement)
            gen_entropies.append(entropy)
            scored.append((post_reward, ind))

            # Write output to terminal and log in CSV
            print(
                f"  Ind {i:2d} | "
                f"Pre RL Reward: {pre_reward:6.2f} | "
                f"Post RL Reward: {post_reward:6.2f} | "
                f"Agreement: {agreement:5.2f} | "
                f"Entropy: {entropy:5.2f}"
            )

            gen_writer.writerow(
                [
                    gen,
                    i,
                    pre_reward,
                    post_reward,
                    agreement,
                    entropy,
                    json.dumps(agreement_agent_actions),
                    json.dumps(agreement_expert_actions),
                    json.dumps(agreement_env_flat),
                ]
            )

        # Store local generation lists in the overall averages for plotting.
        avg_pre_rewards.append(np.mean(gen_pre_rewards))
        avg_post_rewards.append(np.mean(gen_post_rewards))
        avg_gains.append(np.mean(gen_gains))
        avg_agreements.append(np.mean(gen_agreements))
        avg_entropies.append(np.mean(gen_entropies))

        # Create the next generation of individuals.
        ga.evolve(scored)

        # Print the stats over the generation and log averages in CSV.
        print(
            f"Avg Pre RL Reward={avg_pre_rewards[-1]:.2f}, "
            f"Avg Post RL Reward={avg_post_rewards[-1]:.2f}, "
            f"Avg Learning Gain={avg_gains[-1]:.2f}, "
            f"Avg Agreement={avg_agreements[-1]:.2f}"
        )
        avg_writer.writerow(
            [
                gen,
                avg_pre_rewards[-1],
                avg_post_rewards[-1],
                avg_gains[-1],
                avg_agreements[-1],
                avg_entropies[-1],
            ]
        )

    csv_file_gen.close()
    csv_file_avg.close()
    plot(avg_pre_rewards, avg_post_rewards, avg_gains, avg_agreements, avg_entropies)


if __name__ == "__main__":
    main()
