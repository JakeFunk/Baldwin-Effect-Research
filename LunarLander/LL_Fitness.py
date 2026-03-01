import gymnasium as gym
import numpy as np
from LL_Agent import Agent
from LL_Constants import state_size, number_actions

def fitness_function(ga_instance, solution, solution_idx):
    """
        Function:    fitness_function
        Purpose:     Evaluate the fitness of a genome by testing its learning capability in LunarLander.
        Parameters:
                     - ga_instance: The PyGAD genetic algorithm instance managing evolution.
                     - solution: The genome array to evaluate.
                     - solution_idx: Index of this solution in the current population.
        Returns:     Fitness score (mean reward after training).

        Details:
                    - Creates an agent with architecture specified by the genome.
                    - Measures untrained performance over a set amount of episodes (epsilon=0.01).
                    - Trains agent for a set amount of episodes with epsilon-greedy exploration.
                    - Measures trained performance over 5 episodes (epsilon=0.01).
                    - Fitness is the mean trained performance (post-learning reward).
                    - Logs metrics including untrained, trained, and learning delta to ga_instance.
                    - Designed to detect Baldwin Effect: genomes that learn efficiently have higher fitness.
    """
    agent = Agent(state_size, number_actions, solution)
    env = gym.make("LunarLander-v3")

    untrained_rewards = []
    untrained_actions_list = []
    agent.epsilon = 0.01
    for _ in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        actions_taken = []

        while not done:
            action = agent.get_action(state)
            actions_taken.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        untrained_rewards.append(total_reward)
        untrained_actions_list.append(actions_taken)

    untrained_performance = np.mean(untrained_rewards)

    agent.epsilon = 1.0
    for _ in range(300):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()

    trained_rewards = []
    trained_actions_list = []
    agent.epsilon = 0.01
    for _ in range(10):
        state, _ = env.reset()
        done = False
        total_reward = 0
        actions_taken = []

        while not done:
            action = agent.get_action(state)
            actions_taken.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        trained_rewards.append(total_reward)
        trained_actions_list.append(actions_taken)


    trained_performance = np.mean(trained_rewards)
    learning_delta = trained_performance - untrained_performance

    env.close()

    if not hasattr(ga_instance, 'metrics'):
        ga_instance.metrics = []

    ga_instance.metrics.append({
        'generation': ga_instance.generations_completed,
        'solution_idx': solution_idx,
        'untrained': untrained_performance,
        'trained': trained_performance,
        'learning_delta': learning_delta,
        'trained_actions':trained_actions_list,
        'untrained_actions':untrained_actions_list
    })

    print(f"Gen {ga_instance.generations_completed}, Agent {solution_idx}: "
          f"Untrained={untrained_performance:.1f}, Trained={trained_performance:.1f}, "
          f"Delta={learning_delta:.1f}")

    return trained_performance
