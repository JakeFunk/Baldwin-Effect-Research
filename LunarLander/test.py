import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Define the SAME network
# =========================
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


# =========================
# Setup environment
# =========================
RENDER = False  # set True if you want to watch occasionally

def make_env(render=False):
    if render:
        return gym.make("LunarLander-v3", render_mode="human")
    else:
        return gym.make("LunarLander-v3")


env = make_env(RENDER)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load trained model
# =========================
model = DQN(state_size, action_size).to(device)
model.load_state_dict(torch.load("LL_Ideal_Agent.pth", map_location=device))
model.eval()

print("✅ Model loaded successfully!")


# =========================
# Run evaluation
# =========================
num_episodes = 1000
rewards = []

for episode in range(num_episodes):

    # Optional: render occasionally
    if episode % 100 == 0:
        env.close()
        env = make_env(render=True)
    else:
        env.close()
        env = make_env(render=False)

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_t)

        action = q_values.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

    rewards.append(total_reward)

    print(f"Episode {episode}: Reward = {total_reward:.1f}")


# =========================
# Final stats
# =========================
env.close()

avg_reward = sum(rewards) / len(rewards)
max_reward = max(rewards)
min_reward = min(rewards)

print("\n===== RESULTS =====")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Max Reward: {max_reward:.2f}")
print(f"Min Reward: {min_reward:.2f}")
