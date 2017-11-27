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

class DoorKeyEnv(AIGameEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super(DoorKeyEnv, self).__init__(gridSize=size, maxSteps=4 * size)

    def _genGrid(self, width, height):
        grid = super(DoorKeyEnv, self)._genGrid(width, height)
        assert width == height
        gridSz = width

        # Create a vertical splitting wall
        splitIdx = self.np_random.randint(2, gridSz-3)
        for i in range(0, gridSz):
            grid.set(splitIdx, i, Wall())

        # Place a door in the wall
        doorIdx = self.np_random.randint(1, gridSz-2)
        grid.set(splitIdx, doorIdx, Door('yellow'))

        # Place a key on the left side
        #keyIdx = self.np_random.randint(1 + gridSz // 2, gridSz-2)
        keyIdx = gridSz-2
        grid.set(1, keyIdx, Key('yellow'))

        return grid

class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super(DoorKeyEnv16x16, self).__init__(size=16)

register(
    id='AIGame-Door-Key-8x8-v0',
    entry_point='gym_aigame.envs:DoorKeyEnv',
    reward_threshold=1000.0
)

register(
    id='AIGame-Door-Key-16x16-v0',
    entry_point='gym_aigame.envs:DoorKeyEnv16x16',
    reward_threshold=1000.0
)
