import minigrid
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from Agents.Agents import Agent, ReplayMemory

def make_env_with_flattening(env_id, render_mode=None):
    env = gym.make(env_id, render_mode)
    env = FlatObsWrapper(env)
    
    return env

def training_loop(agent: Agent, env_name, num_episodes = 1000, target_update_every = 10):    
    print("Training on: ", env_name)
    env = make_env_with_flattening(env_name)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs.astype(np.float32)

        done = False
        total_reward = 0.0
        avg_over_update = 0.0

        while not done:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            next_state = next_obs.astype(np.float32)

            agent.step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            avg_over_update += reward

        agent.decay_epsilon()

        if episode % target_update_every == 0:
            agent.update_target()
            print(f"Episode {episode:4d} | Reward {total_reward:6.2f} | Epsilon {agent.epsilon:.3f}")

def curriculum_learning():
    temp_env = make_env_with_flattening("MiniGrid-DoorKey-6x6-v0")
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.n
    temp_env.close()

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        epsilon_decay=0.999
    )    
       
    training_loop(
        agent=agent,
        env_name="MiniGrid-Fetch-6x6-N2-v0",
        num_episodes=1000,
        target_update_every=10
    )
    agent.epsilon = 1.0
    
    training_loop(
        agent=agent,
        env_name="MiniGrid-DoorKey-6x6-v0",
        num_episodes=2500,
        target_update_every=10
    )
    
if __name__ == '__main__':
    curriculum_learning()