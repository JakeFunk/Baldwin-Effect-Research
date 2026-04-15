import gymnasium as gym
import numpy as np
from LL_Agent import Agent
from LL_Constants import state_size, number_actions
import torch
from LL_Baseline_Agent import DQN as SimpleDQN
from LL_Baseline_Agent import Agent as BaselineAgent

IDEAL_AGENT = "LL_Ideal_Agent.pth"

def load_expert_agent():
    expert = BaselineAgent(state_size, number_actions)
    expert.network.load_state_dict(torch.load(IDEAL_AGENT))
    expert.network.eval()
    expert.epsilon = 0.01
    return expert

expert = load_expert_agent()

def calculate_agreement_expert(agent, expert, env, num_episodes=10):
    total_matches = 0
    total_steps = 0
    
    agent.epsilon = 0.01
    expert.epsilon = 0.01
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            agent_action = agent.get_action(state)
            expert_action = expert.get_action(state)
            
            if agent_action == expert_action:
                total_matches += 1
            total_steps += 1

            next_state, reward, terminated, truncated, _ = env.step(expert_action)
            done = terminated or truncated
            state = next_state
    
    return total_matches / total_steps if total_steps > 0 else 0.0

def calculate_agreement_agent(agent, expert, env, num_episodes=10):
    total_matches = 0
    total_steps = 0

    agent.epsilon = 0.01
    expert.epsilon = 0.01

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            agent_action = agent.get_action(state)
            expert_action = expert.get_action(state)

            if agent_action == expert_action:
                total_matches += 1
            total_steps += 1

            next_state, reward, terminated, truncated, _ = env.step(agent_action)
            done = terminated or truncated
            state = next_state

    return total_matches / total_steps if total_steps > 0 else 0.0


def fitness_function(ga_instance, solution, solution_idx):
    gen = ga_instance.generations_completed
    agent = Agent(state_size, number_actions, solution)
    env = gym.make("LunarLander-v3", max_episode_steps = 1000)
    activation_label = "-".join(agent.network.activation_names)
    training_episodes = 300
    
    agent.epsilon = 0.01
    untrained_agreement_expert = calculate_agreement_expert(agent, expert, env, num_episodes=10)
    untrained_agreement_agent =  calculate_agreement_agent(agent, expert, env, num_episodes=10)

    # Untrained performance
    untrained_rewards = []
    untrained_actions_list = []
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
        untrained_rewards.append(total_reward)
        untrained_actions_list.append(actions_taken)

    untrained_performance = np.mean(untrained_rewards)

    # Training
    agent.epsilon = 1.0
    for _ in range(training_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()

    # Trained performance
    trained_rewards = []
    trained_actions_list = []
    agent.epsilon = 0.01
    trained_agreement_expert = calculate_agreement_expert(agent, expert, env, num_episodes=10)
    trained_agreement_agent =  calculate_agreement_agent(agent, expert, env, num_episodes=10)
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
        'activations': activation_label,
        'num_layers': len(agent.network.activation_names),
        'untrained': untrained_performance,
        'trained': trained_performance,
        'learning_delta': learning_delta,
        'untrained_agreement_agent': untrained_agreement_agent,
        'untrained_agreement_expert': untrained_agreement_expert,
        'trained_actions': trained_actions_list,
        'untrained_actions': untrained_actions_list,
        'trained_agreement_agent': trained_agreement_agent,
        'trained_agreement_expert': trained_agreement_expert
    })


    total_fitness = trained_performance

    print(f"Gen {ga_instance.generations_completed}, Agent {solution_idx}: "
        f"U={untrained_performance:.1f}, T={trained_performance:.1f}, "
        f"Fit={total_fitness:.1f}, Act={activation_label}")

    return total_fitness
