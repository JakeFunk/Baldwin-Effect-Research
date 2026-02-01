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

### Matt add your env details

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
