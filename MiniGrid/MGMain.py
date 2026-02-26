import minigrid
import csv
import json
import numpy as np
from stable_baselines3 import PPO
from MGGeneticAlgorithm import MGGeneticAlgorithm
from MGModelCreation import make_env
from MGPlotting import plot

ENV_ID = "MiniGrid-LavaGapS7-v0"
GENERATIONS = 60

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
            ind.learn(total_timesteps=ga.learning_steps(gen))

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