import minigrid
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from Agents.Agents import Agent
  
def main() -> None:
    env = gym.make("MiniGrid-DoorKey-6x6-v0")
    env = FlatObsWrapper(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        epsilon_decay=0.999
    )
    
    num_episodes = 1000
    target_update_every = 10
       
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs.astype(np.float32)

        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            next_state = next_obs.astype(np.float32)

            agent.step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if episode % target_update_every == 0:
            agent.update_target()
            print(f"Episode {episode:4d} | Reward {total_reward:6.2f} | Epsilon {agent.epsilon:.3f}")
    
    env.close()
    
if __name__ == '__main__':
    main()