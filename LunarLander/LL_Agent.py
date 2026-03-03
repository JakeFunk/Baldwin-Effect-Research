import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from LL_DQN import ReplayMemory, DQN
from LL_Constants import MAX_WEIGHTS, learning_rate, minibatch_size, discount_factor, replay_buffer_size, Transition, interpolation_parameter


class Agent():
    def __init__(self, state_size, action_size, genome=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        if genome is None:
            genome = [2, 64, 0, 64, 0, 64, 0] + [0.0] * MAX_WEIGHTS


        self.network = DQN(state_size, action_size, genome).to(self.device)
        self.target_network = DQN(state_size, action_size, genome).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.memory = ReplayMemory(replay_buffer_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.batch_size = minibatch_size
        self.gamma = discount_factor

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.t_step = 0
        self.update_every = 4

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_t)
        return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = None if done else torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_t = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        self.memory.push(state_t, action_t, next_state_t, reward_t)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool
        )

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        q_sa = self.network(state_batch).gather(1, action_batch).squeeze(1)

        next_q = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q[non_final_mask] = self.target_network(non_final_next_states).max(1).values

        target = reward_batch + self.gamma * next_q

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()

        self.soft_update_target(tau=interpolation_parameter)

    def soft_update_target(self, tau=0.001):
        for target_param, param in zip(self.target_network.parameters(),
                                         self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
