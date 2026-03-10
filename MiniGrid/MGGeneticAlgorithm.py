import torch
import copy
import random
import numpy as np
from MGModelCreation import make_env, make_model
from MGWrappers import UnWrappers


ENV_ID = "MiniGrid-LavaGapS7-v0"
GENERATIONS = 60
POP_SIZE = 20
ELITES = 4
MUTATION_STD = 0.02
MAX_STEPS = 500
LEARNING_STEPS = 25_000


class MGGeneticAlgorithm:
    """
    Implements a genetic algorithm used to evolve reinforcement learning agents
    for a MiniGrid environment. Each individual in the population is a PPO model
    whose weights are mutated and evolved across generations.
    """
    def __init__(self, env_id: str, pop_size: int = POP_SIZE, mutation_std: float = MUTATION_STD):
        """
        Initializes the genetic algorithm and creates the initial population of agents.

        :param env_id: The Gym environment ID used to construct environments
        :type env_id: str
        :param pop_size: The number of individuals maintained in the population
        :type pop_size: int
        :param mutation_std: The standard deviation used when mutating network weights
        :type mutation_std: float
        """
        self.env_id = env_id
        self.pop_size = pop_size
        self.mutation_std = mutation_std
        self.population = [make_model(make_env(self.env_id)) for _ in range(self.pop_size)]

    def mutate_weights(self, model, rate=0.1):
        """
        Applies random mutations to the weights of a model's policy network.

        :param model: The reinforcement learning model whose parameters will be mutated
        :type model: stable_baselines3.PPO
        :param rate: The probability that each parameter element will be mutated
        :type rate: float
        """
        for p in model.policy.parameters():
            mask = torch.rand_like(p) < rate
            p.data += mask * torch.randn_like(p) * self.mutation_std

    def evolve(self, scored):
        """
        Evolves the current population using genetic algorithm selection and mutation.
        The top performing individuals are preserved as elites and used to generate
        the next population through mutation.

        :param scored: A list of tuples containing (fitness_score, model)
        :type scored: list[tuple[float, stable_baselines3.PPO]]
        """
        # GA Selection
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in scored[:ELITES]]

        # GA Reproduction
        new_pop = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = random.choice(elites)
            child = make_model(make_env(ENV_ID))
            child.policy.load_state_dict(copy.deepcopy(parent.policy.state_dict()))

            self.mutate_weights(child)
            new_pop.append(child)

        self.population = new_pop

    def rollout_reward(self, model, env):
        """
        Executes a rollout of a model in the environment and returns the final reward
        achieved by the agent.

        :param model: The reinforcement learning model used to act in the environment
        :type model: stable_baselines3.PPO
        :param env: The vectorized MiniGrid environment
        :type env: gym.Env
        :return: The final reward received during the rollout
        :rtype: float
        """
        # Ensure the environment is set to a start state.
        obs = env.reset()
        final_reward = 0.0

        for _ in range(MAX_STEPS):
            # Predict the best action, then act in the environment.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            final_reward = float(reward[0])

            # Done can be either a goal or lava.
            if done[0]:
                break

        return final_reward

    def agreement(self, agent, expert, env):
        """
        Measures the agreement between an evolving agent and an expert policy by
        comparing their selected actions over a rollout.

        :param agent: The evolving agent being evaluated
        :type agent: stable_baselines3.PPO
        :param expert: The expert policy used as a reference
        :type expert: stable_baselines3.PPO
        :param env: The MiniGrid environment used for evaluation
        :type env: gym.Env
        :return: Agreement ratio, expert actions, agent actions, and flattened environment
        :rtype: tuple[float, list[int], list[int], list[int]]
        """        
        obs = env.reset()
        matches = 0
        steps = 0

        expert_actions = []
        agent_actions = []
        
        flat_obs = UnWrappers.flatten_env_grid(env)

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

        return matches / steps if steps else 0.0, expert_actions, agent_actions, flat_obs

    def policy_entropy(self, model, env):
        """
        Computes the average entropy of a model's policy across a rollout. Higher
        entropy indicates more stochastic or exploratory behavior.

        :param model: The reinforcement learning model whose policy entropy is measured
        :type model: stable_baselines3.PPO
        :param env: The MiniGrid environment used for the rollout
        :type env: gym.Env
        :return: The average entropy of the policy during the rollout
        :rtype: float
        """
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
    
    def learning_steps(self, gen):
        """
        Calculates the number of learning steps an individual should perform in the
        current generation. The learning budget decreases over time.

        :param gen: The current generation number
        :type gen: int
        :return: The number of learning timesteps allocated for the generation
        :rtype: int
        """
        return int(LEARNING_STEPS * (1 - gen / GENERATIONS))