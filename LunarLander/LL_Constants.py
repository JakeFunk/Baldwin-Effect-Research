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

# Max Weights per layer = (inputs * outputs) + biases
max_l1 = (state_size * 32) + 32    # 288
max_l2 = (32 * 32) + 32            # 1056
max_l3 = (32 * 32) + 32            # 1056
max_l4 = (32 * 32) + 32            # 1056
max_out = (32 * number_actions) + number_actions # 132

MAX_WEIGHTS = max_l1 + max_l2 + max_l3 + max_l4 + max_out # 3588 total weight genes

# First 9 genes are architecture, the next 3588 are weights (-1.0 to 1.0)
gene_space = [
    {'low': 2, 'high': 5},
    {'low': 8, 'high': 32}, {'low': 0, 'high': 4},
    {'low': 8, 'high': 32}, {'low': 0, 'high': 4},
    {'low': 8, 'high': 32}, {'low': 0, 'high': 4},
    {'low': 8, 'high': 32}, {'low': 0, 'high': 4},
] + [{'low': -1.0, 'high': 1.0}] * MAX_WEIGHTS
