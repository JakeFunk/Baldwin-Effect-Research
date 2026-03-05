import json
import csv
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
        agent_start_pos=(1,1),
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
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = f"Generation {self.generation} - {self.individual}"
        
    def serialize_grid(self):
        width, height = self.width, self.height
        grid_array = np.zeros((height, width), dtype=int)

        for y in range(height):
            for x in range(width):
                cell = self.grid.get(x, y)
                if cell is None:
                    grid_array[y, x] = 0
                elif isinstance(cell, Wall):
                    grid_array[y, x] = 1
                elif isinstance(cell, Lava):
                    grid_array[y, x] = 2
                elif isinstance(cell, Goal):
                    grid_array[y, x] = 5
                else:
                    grid_array[y, x] = 9

        x, y = self.agent_pos

        return {
            "grid": grid_array.tolist(),
            "agent_pos": (int(x), int(y)),
            "agent_dir": self.agent_dir,
            "generation": self.generation,
            "individual": self.individual,
            "step_count": self.step_count,
        }

def main():
    csv_file = open("minigrid_disagreements.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "generation",
            "individual",
            "agent_actions",
            "expert_actions",
            "disagreement_step",
            "agent_proposed",
            "expert_performed",
            "env_at_disagreement"
        ]
    )
    
    df = pd.read_csv("Run 5/minigrid_stats.csv")
    for _, row in df.iterrows():
        # Obtain the necessary data from the row
        env_flat = json.loads(row["agreement_env_flattened"])
        agent_actions = json.loads(row["agreement_agent_actions"])
        expert_actions = json.loads(row["agreement_expert_actions"])
        generation = row["generation"]
        individual = row["individual"]
        
        # Generate the same environment the agreement was assessed on
        env = ReplayEnv(generation, individual, env_flat)
        env.reset()
        
        for i, (a_action, e_action) in enumerate(zip(agent_actions, expert_actions)):
            if a_action != e_action:
                env_snapshot = env.serialize_grid()
                csv_writer.writerow(
                    [
                        generation,
                        individual,
                        json.dumps(agent_actions),
                        json.dumps(expert_actions),
                        i,
                        a_action,
                        e_action,
                        json.dumps(env_snapshot)
                    ]
                )

            env.step(e_action)
            
    
if __name__ == '__main__':
    main() 