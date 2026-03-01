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
    """
    Class:       DQN (Deep Q-Network)
    Purpose:     Neural network model to approximate Q-values for RL.
    Parameters:
                 - n_observations: Number of input features (state dimensions).
                 - n_actions: Number of possible actions.
                 - genome: Genome array encoding architecture, activations, and optional weights.
    Methods:
                 - inject_weights(weight_DNA): Assigns weights directly from genome.
                 - forward(x): Performs forward pass to produce Q-values.
    Details:
                 - Network architecture and activations decoded from genome.
                 - Supports variable number of layers and activation functions.
                 - Optional weight injection allows exact genome-determined parameters.
    """
    def __init__(self, n_observations, n_actions, genome):
        super(DQN, self).__init__()
        
        arch = decode_genome(genome[:9], n_observations, n_actions)

        self.layers = nn.ModuleList()
        self.activation_funcs = arch['activations']

        input_size = n_observations
        for layer_size in arch['layers']:
            self.layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        self.output_layer = nn.Linear(input_size, n_actions)

        if len(genome) > 9:
            self.inject_weights(genome[9:])

    def inject_weights(self, weight_DNA):
        """
        Function:    inject_weights
        Purpose:     Initialize network parameters from genome-provided weights.
        Parameters:
                     - weight_DNA: List of weight values from genome.
        Returns:     None
        Details:
                     - Iterates over all model parameters and reshapes slices of weight_DNA to match.
                     - Uses torch.no_grad() to avoid affecting gradient computation.
        """
        with torch.no_grad():
            ptr = 0
            for param in self.parameters():
                num_param = param.numel()
                if ptr + num_param <= len(weight_DNA):
                    gene_slice = torch.tensor(weight_DNA[ptr:ptr+num_param], dtype=torch.float32)
                    param.data.copy_(gene_slice.view(param.size()))
                    ptr += num_param

    def forward(self, x):
        """
        Function:    forward
        Purpose:     Compute Q-values for a given input state.
        Parameters:
                     - x: Input state tensor or array.
        Returns:     Tensor of Q-values, one per action.
        Details:
                     - Converts input to torch.Tensor if needed.
                     - Applies each layer and its corresponding activation function.
                     - Final layer outputs raw Q-values for all actions.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation_funcs[i](x)
        return self.output_layer(x)

def decode_genome(genome, input_size, output_size):
    """
    Function:    decode_genome
    Purpose:     Translate genome array into network architecture and activations.
    Parameters:
                 - genome: List of integers defining layers and activations.
                 - input_size: Number of input features.
                 - output_size: Number of output actions (not used directly here).
    Returns:     Dictionary with keys:
                   - 'layers': list of layer sizes
                   - 'activations': list of activation functions
    Details:
                 - First genome element determines number of hidden layers (max 4).
                 - Each layer defined by size and activation index.
                 - Activation index mapped via activation_map: ReLU, Tanh, Sigmoid, Leaky ReLU.
                 - Enables flexible, genome-driven network architectures for evolutionary RL.
    """
    activation_map = {
        0: F.relu, 1: torch.tanh, 2: torch.sigmoid, 3: lambda x: F.leaky_relu(x, 0.01)
    }
    arch = {'layers': [], 'activations': []}
    num_layers = min(4, int(genome[0]))

    for i in range(num_layers):
        layer_size = int(genome[1 + i*2])
        activation_idx = int(genome[2 + i*2]) % 4
        arch['layers'].append(layer_size)
        arch['activations'].append(activation_map[activation_idx])

    return arch
