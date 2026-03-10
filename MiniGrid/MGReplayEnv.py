import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Lava, Goal

class ReplayEnv(MiniGridEnv):
    """
    A MiniGrid environment that reconstructs a grid from a flattened representation.
    Primarily based on MiniGrid's documentation: https://minigrid.farama.org/content/create_env_tutorial/
    
    :param env_flat: Flattened grid representation of the environment (1D list of integers).
    :type env_flat: list[int]
    :param size: Width and height of the square grid.
    :type size: int
    :param agent_start_pos: Starting position of the agent (x, y).
    :type agent_start_pos: tuple[int, int]
    :param agent_start_dir: Starting direction of the agent (0: right, 1: down, 2: left, 3: up).
    :type agent_start_dir: int
    :param max_steps: Maximum number of steps per episode. Defaults to 256 if None.
    :type max_steps: int, optional
    """
    def __init__(
        self,
        env_flat,
        size=7,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        max_steps=None,
        **kwargs
    ):
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

        self.mission = "Replaying Environment"