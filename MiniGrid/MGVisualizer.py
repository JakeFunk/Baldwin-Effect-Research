import json
import pandas as pd
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Lava, Goal

class ReplayEnv(MiniGridEnv):
    def __init__(
        self,
        generation,
        individual,
        env_flat,
        size=7,
        agent_start_pos=(2,2),
        agent_start_dir=0,
        max_steps=None,
        **kwargs
    ):
        self.generation = generation
        self.individual = individual
        self.env_flat = env_flat
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        # Place the agent at its starting position
        self.agent_pos = np.array(self.agent_start_pos)
        self.agent_dir = 0
        
        self.step_count = 0
        self.max_steps = max_steps or 256
        self.terminated = False
        self.truncated = False
        self.carrying = None

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )
    
    @staticmethod
    def _gen_mission():
        return "Replaying Environment"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Reshape the CSV data to follow the MiniGrid format
        env_array = np.array(self.env_flat[: width * height]).reshape((height, width))

        for y in range(height):
            for x in range(width):
                obj_id = env_array[y, x]

                if obj_id == 1:
                    self.grid.set(x, y, Wall())
                elif obj_id == 2:
                    self.grid.set(x, y, Lava())
                elif obj_id == 5:
                    self.grid.set(x, y, Goal())
                else:
                    self.grid.set(x, y, None)

        # Place the agent at it's starting position
        self.agent_pos = np.array(self.agent_start_pos)
        self.agent_dir = self.agent_start_dir

        self.mission = f"Generation {self.generation} - {self.individual}"

def main():
    df = pd.read_csv("minigrid_stats.csv")
    for _, row in df.iterrows():
        # Obtain the necessary data from the row
        env_flat = json.loads(row["agreement_env_flattened"])
        agent_actions = json.loads(row["agreement_agent_actions"])
        expert_actions = json.loads(row["agreement_expert_actions"])
        generation = row["generation"]
        individual = row["individual"]
        
        # Generate the same environment the agreement was assessed on
        env = ReplayEnv(generation, individual, env_flat, render_mode="human")
        for a_actions, e_actions in zip(agent_actions, expert_actions):
            #TODO Not implemented yet
            pass
    
if __name__ == '__main__':
    main() 