import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from NeuralNetwork.DeepQNetwork import DQN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int=int(1e5),
        learning_rate=5e-4,
        batch_size=128,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        self.network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.memory = ReplayMemory(buffer_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.t_step = 0
        self.update_every = 4

    def get_action(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_t)
        return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        # convert to tensors
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = None if done else torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_t = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # store transition
        self.memory.push(state_t, action_t, next_state_t, reward_t)

        # learn every few steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size*2:
            self.learn()

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # mask for non-final states
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool
        )

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Q(s,a)
        q_sa = self.network(state_batch).gather(1, action_batch).squeeze(1)

        # max_a' Q_target(s',a')
        next_q = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q[non_final_mask] = self.target_network(non_final_next_states).max(1).values

        # target = r + gamma * next_q
        target = reward_batch + self.gamma * next_q

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)