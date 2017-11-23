from gym.envs.registration import register
from gym_aigame.envs.ai_game_env import *

class EmptyEnv(AIGameEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super(EmptyEnv, self).__init__(gridSize=size, maxSteps=2 * size)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super(EmptyEnv6x6, self).__init__(size=6)

register(
    id='AIGame-Empty-8x8-v0',
    entry_point='gym_aigame.envs:EmptyEnv',
    reward_threshold=1000.0
)

register(
    id='AIGame-Empty-6x6-v0',
    entry_point='gym_aigame.envs:EmptyEnv6x6',
    reward_threshold=1000.0
)

class WallHoleEnv(AIGameEnv):
    """
    Environment with a hollowed wall, sparse reward
    """

    def __init__(self, size=8):
        super(WallHoleEnv, self).__init__(gridSize=size, maxSteps=2 * size)

    def _seed(self, seed=None):
        super(WallHoleEnv, self)._seed(seed)

        gridSz = self.gridSize

        # Create a vertical splitting wall
        splitIdx = self.np_random.randint(2, gridSz-3)
        for i in range(0, gridSz):
            self.setGrid(splitIdx, i, Wall())

        # Place a hole in the wall
        doorIdx = self.np_random.randint(1, gridSz-2)
        self.setGrid(splitIdx, doorIdx, None)

        # Store a copy of the grid so we can restore it on reset
        self.seedGrid = deepcopy(self.grid)

register(
    id='AIGame-Wall-Hole-8x8-v0',
    entry_point='gym_aigame.envs:WallHoleEnv',
    reward_threshold=1000.0
)

class DoorKeyEnv(AIGameEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super(DoorKeyEnv, self).__init__(gridSize=size, maxSteps=4 * size)

    def _seed(self, seed=None):
        super(DoorKeyEnv, self)._seed(seed)

        gridSz = self.gridSize

        # Create a vertical splitting wall
        splitIdx = self.np_random.randint(2, gridSz-3)
        for i in range(0, gridSz):
            self.setGrid(splitIdx, i, Wall())

        # Place a door in the wall
        doorIdx = self.np_random.randint(1, gridSz-2)
        self.setGrid(splitIdx, doorIdx, Door('yellow'))

        # Place a key on the left side
        keyIdx = self.np_random.randint(1, gridSz-2)
        self.setGrid(1, keyIdx, Key('yellow'))

        # Store a copy of the grid so we can restore it on reset
        self.seedGrid = deepcopy(self.grid)

register(
    id='AIGame-Door-Key-8x8-v0',
    entry_point='gym_aigame.envs:WallHoleEnv',
    reward_threshold=1000.0
)
