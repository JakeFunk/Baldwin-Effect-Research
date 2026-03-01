import gymnasium as gym
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 0.05

max_l1 = (state_size * 16) + 16    # 144
max_l2 = (16 * 16) + 16            # 272
max_l3 = (16 * 16) + 16            # 272
max_out = (16 * number_actions) + number_actions # 68

MAX_WEIGHTS = max_l1 + max_l2 + max_l3 + max_out # 756 

gene_space = [
    {'low': 1, 'high': 3},                    # 1-3 layers
    {'low': 12, 'high': 16}, {'low': 0, 'high': 4},  # layer1: 12-16 neurons
    {'low': 12, 'high': 16}, {'low': 0, 'high': 4},  # layer2: 12-16 neurons
    {'low': 12, 'high': 16}, {'low': 0, 'high': 4},  # layer3: 12-16 neurons
] + [{'low': -1.0, 'high': 1.0}] * MAX_WEIGHTS

"""
gene_space = [
    {'low': 2, 'high': 5},                    # num_layers
    {'low': 8, 'high': 64}, {'low': 0, 'high': 4},  # layer1 size, activation
    {'low': 8, 'high': 64}, {'low': 0, 'high': 4},  # layer2 size, activation
    {'low': 8, 'high': 64}, {'low': 0, 'high': 4},  # layer3 size, activation
    {'low': 8, 'high': 64}, {'low': 0, 'high': 4},  # layer4 size, activation
]
"""
