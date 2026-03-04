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
        return int(LEARNING_STEPS * (1 - gen / GENERATIONS))