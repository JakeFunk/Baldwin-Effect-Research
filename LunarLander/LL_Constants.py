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

max_l1 = (state_size * 128) + 128    
max_l2 = (128 * 128) + 128
max_l3 = (128 * 128) + 128
max_out = (128 * number_actions) + number_actions

MAX_WEIGHTS = max_l1 + max_l2 + max_l3 + max_out

gene_space = [
    {'low': 1, 'high': 4},                    # 1-3 layers
    {'low': 32, 'high': 128}, {'low': 0, 'high': 4},  # layer1: 12-16 neurons
    {'low': 32, 'high': 128}, {'low': 0, 'high': 4},  # layer2: 12-16 neurons
    {'low': 32, 'high': 128}, {'low': 0, 'high': 4},  # layer3: 12-16 neurons
] + [{'low': -1.0, 'high': 1.0}] * MAX_WEIGHTS

