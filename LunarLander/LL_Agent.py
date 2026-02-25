import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from LL_DQN import ReplayMemory, DQN
from LL_Constants import learning_rate, minibatch_size, discount_factor, replay_buffer_size, Transition, interpolation_parameter


class Agent():
    """
        Class:       Agent
        Purpose:     A DQN agent with optional genome-specified architecture for evolutionary optimization.
        Parameters:
                    - state_size: Dimension of the state observation space.
                    - action_size: Number of discrete actions available.
                    - genome: Optional numerical array encoding the neural network architecture.
                              If None, uses a default architecture.
        Details:
                    - Creates both policy and target networks using genome-specified or default architecture.
                    - Implements epsilon-greedy exploration with decay.
                    - Uses experience replay and target network for stable learning.
                    - Supports both evolutionary (with genome) and standard DQN training.
    """
    def __init__(self, state_size, action_size, genome=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        if genome is None:
            genome = [3, 64, 0, 64, 0, 32, 0, 16, 0] + [0.0] * 3588

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
        """
            Function:    get_action
            Purpose:     Select an action using epsilon-greedy policy.
            Parameters:
                        - state: Current environment state.
            Returns:     Action index (integer).
            Details:
                        - With probability epsilon, selects random action (exploration).
                        - Otherwise, selects action with highest Q-value (exploitation).
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_t)
        return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        """
            Function:    step
            Purpose:     Store experience and trigger learning updates.
            Parameters:
                        - state: Current state.
                        - action: Action taken.
                        - reward: Reward received.
                        - next_state: Resulting state.
                        - done: Whether episode terminated.
            Details:
                        - Stores transition in replay memory.
                        - Triggers learning every 'update_every' steps if enough samples available.
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = None if done else torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_t = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        self.memory.push(state_t, action_t, next_state_t, reward_t)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        """
            Function:    learn
            Purpose:     Update network weights using sampled experiences.
            Details:
                        - Samples random batch from replay memory.
                        - Computes Q-values for current states.
                        - Computes target Q-values using target network.
                        - Updates policy network via gradient descent on TD error.
                        - Soft updates target network.
        """
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
        """
            Function:    soft_update_target
            Purpose:     Gradually update target network weights toward policy network.
            Parameters:
                        - tau: Interpolation parameter (0 = no update, 1 = full copy).
            Details:
                        - Implements soft update: θ_target = τ*θ_policy + (1-τ)*θ_target
                        - Stabilizes learning by preventing abrupt target changes.
        """
        for target_param, param in zip(self.target_network.parameters(),
                                         self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        """
            Function:    decay_epsilon
            Purpose:     Reduce exploration rate over time.
            Details:
                        - Multiplies epsilon by decay factor.
                        - Ensures epsilon doesn't fall below minimum threshold.
                        - Implements gradual shift from exploration to exploitation.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
