import gymnasium as gym
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import numpy as np

env = gym.make("LunarLander-v3", render_mode = "human")
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 0.05



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

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        self.network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
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

    def soft_update_target(self, tau=interpolation_parameter):
        for target_param, param in zip(self.target_network.parameters(),
                                         self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    agent = Agent(state_size, number_actions)
    train_env = gym.make("LunarLander-v3")
    render_env = gym.make("LunarLander-v3", render_mode="human")

    num_episodes = 1000

    csv_file_step = open("ideal_agent_behavior.csv", "w", newline="")
    step_writer = csv.writer(csv_file_step)
    step_writer.writerow(["episode", "step", "state", "action", "reward", "cumulative_reward"])

    csv_file_episode = open("ideal_agent_episode_stats.csv", "w", newline="")
    episode_writer = csv.writer(csv_file_episode)
    episode_writer.writerow([
        "episode",
        "total_reward",
        "action_prob_0",
        "action_prob_1",
        "action_prob_2",
        "action_prob_3"
    ])

    for episode in range(num_episodes):
        render = (episode % 100) < 5
        env = render_env if render else train_env

        state, _ = env.reset()
        done = False
        total_reward = 0
        step_num = 0
        actions_taken = []  # Track actions for this episode

        while not done:
            action = agent.get_action(state)
            actions_taken.append(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if not render:
                agent.step(state, action, reward, next_state, done)

            total_reward += reward

            step_writer.writerow([
                episode,
                step_num,
                state.tolist(),
                action,
                reward,
                total_reward
            ])

            state = next_state
            step_num += 1

        # Calculate action probabilities for this episode
        action_counts = np.bincount(actions_taken, minlength=4)
        action_probs = action_counts / action_counts.sum()

        # Write episode stats
        episode_writer.writerow([
            episode,
            total_reward,
            action_probs[0],
            action_probs[1],
            action_probs[2],
            action_probs[3]
        ])

        if not render:
            agent.decay_epsilon()

        if episode % 10 == 0:
            print(f"Episode {episode} Reward {total_reward:.1f} Render={render}")

    csv_file_step.close()
    csv_file_episode.close()
    train_env.close()
    render_env.close()

    torch.save(agent.network.state_dict(), "LL_Ideal_Agent.pth")
    print(f"Model has been saved as LL_Ideal_Agent")
