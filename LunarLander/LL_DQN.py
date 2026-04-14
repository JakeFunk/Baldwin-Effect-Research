import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from LL_Constants import Transition

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, genome):
        super(DQN, self).__init__()
        
        arch = decode_genome(genome[:7], n_observations, n_actions)

        self.layers = nn.ModuleList()
        self.activation_funcs = arch['activations']
        self.activation_names = arch['names']

        input_size = n_observations
        for layer_size in arch['layers']:
            self.layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        self.output_layer = nn.Linear(input_size, n_actions)

        if len(genome) > 7:
            self.inject_weights(genome[7:])

    def inject_weights(self, weight_DNA):
        with torch.no_grad():
            ptr = 0
            for param in self.parameters():
                num_param = param.numel()
                if ptr + num_param <= len(weight_DNA):
                    gene_slice = torch.tensor(weight_DNA[ptr:ptr+num_param], dtype=torch.float32)
                    param.data.copy_(gene_slice.view(param.size()))
                    ptr += num_param

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation_funcs[i](x)
        return self.output_layer(x)

def decode_genome(genome, input_size, output_size):
    activation_map = {
        0: (F.relu, 'relu'),
        1: (torch.tanh, 'tanh'),
        2: (torch.sigmoid, 'sigmoid'),
        3: (lambda x: F.leaky_relu(x, 0.01), 'leaky_relu')
    }
    arch = {'layers': [], 'activations': [], 'names':[] }
    num_layers = min(3, int(genome[0]))

    for i in range(num_layers):
        layer_size = int(genome[1 + i*2])
        activation_idx = int(genome[2 + i*2]) % 4
        
        func, name = activation_map[activation_idx]
        
        arch['layers'].append(layer_size)
        arch['activations'].append(func)
        arch['names'].append(name)

    return arch
