# Baldwin-Effect-Research

This project investigates the Baldwin Effect in a simulated evolutionary setting by combining reinforcement learning with a genetic algorithm. Agents act as organisms that learn during their lifetime and evolve across generations, allowing us to study whether learned behaviours can become incorporated into inherited policies.

The experiment is conducted using the Lunar Lander environment from OpenAI’s Gymnasium library, a custom environment from using MiniGrid, and a custom Snake game. Each agent learns a control policy through reinforcement learning, while evolutionary selection operates on agent parameters to produce successive generations.

## Project Goals

- Simulate Baldwin Effect in a controlled computational environment
- Examine if learned behaviours by early generations influence the initial policy of later generations

## Environments

### Lunar Lander

**Type:** LunerLander-v3

**State Space:** 8-dimensional vector consisting of the lander’s x, y coordinates, linear velocities along both axes, orientation angle, angular velocity, and two binary indicators for ground contact of each landing leg.

**Action Space:** 4 discrete actions being: No-operation, fire left engine, fire right engine, and fire main engine.

**Reward Structure:** The agent receives a reward, and the episode return is the sum of all step rewards. Rewards favor proximity to the landing pad, slow descent, stable orientation, and ground contact, while engine use and tilt are penalized. Episodes end with +100 for a safe landing or −100 for a crash, and a score of 200 or higher is considered a solution.

### MiniGrid Lava Gap

**Type:** MiniGrid-LavaGapS7-v0

**State Space:** The environment is wrapped in the FlatObsWrapper and produces a partially observable egocentric view as a 7×7×3 tensor. Each cell encodes a 3-dimensional tuple:

- Object type e.g., empty, lava, wall, goal
- Object color (not applicable in this environment)
- State (not applicable in this environment)

The observation is oriented relative to the agent’s facing direction, meaning rotations rotate the observation accordingly. Additionally, the observation is transposed using Stable-Baselines3’s VecTransposeImage so that it can be used directly with SB3’s CNN-based PPO policy (CnnPolicy), which expects channel-first format (channels, height, width).

**Action Space:** The action space is restricted by the ValidActionsWrapper to allow only meaningful actions for this environment:

- Turn Left
- Turn Right
- Move Forward

**Reward Structure:** The environment uses a sparse reward system. The agent receives a reward only upon reaching the goal tile, which is scaled according to the number of steps taken: 1 - 0.9 * (step_count / max_steps). If the agent fails to reach the goal within the maximum number of steps or steps into lava, it receives a reward of zero. Intermediate actions do not yield any reward, and episodes terminate when the agent reaches the goal, steps in lava, or exceeds the maximum step limit.

### Ethan add your env details

## Methodology

1. Agents will learn policy during their lifetime
2. Agents performance is evaluated based on cumulative reward
3. A genetic algorithm will combine, select, and mutate top performing agents
4. Initial and final policies are compared across generations

## Technologies Used

- Python 3.13.11
- Gymnasium
- MiniGrid
- Pytorch
