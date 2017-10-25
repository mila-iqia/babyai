import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from gym_aigame.envs.rendering import *

# Size in pixels of a cell in the human view
CELL_PIXELS = 32

# Size of the image given as an observation to the agent
IMG_ARRAY_SIZE = (160, 160, 3)

COLORS = {
    'red'   : (255, 0, 0),
    'green' : (0, 255, 0),
    'blue'  : (0, 0, 255),
    'purple': (112, 39, 195),
    'yellow': (255, 255, 0),
    'grey'  : (100, 100, 100)
}

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert color in COLORS, color
        self.type = type
        self.color = color
        self.contains = None

    def canOverlap(self):
        """Can the agent overlap with this?"""
        return False

    def canPickup(self):
        """Can the agent pick this up?"""
        return False

    def canContain(self):
        """Can this contain another object?"""
        return False

    def render(self, r):
        assert False

    def _setColor(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

class Goal(WorldObj):
    def __init__(self):
        super(Goal, self).__init__('goal', 'green')

    def render(self, r):
        self._setColor(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Wall(WorldObj):
    def __init__(self):
        super(Wall, self).__init__('wall', 'grey')

    def render(self, r):
        self._setColor(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def canPickup(self):
        return True

    def render(self, r):
        self._setColor(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('ball', color)

    def canPickup(self):
        return True

    def render(self, r):
        self._setColor(r)

        # Vertical quad
        r.drawPolygon([
            (16, 10),
            (20, 10),
            (20, 28),
            (16, 28)
        ])

        # Teeth
        r.drawPolygon([
            (12, 19),
            (16, 19),
            (16, 21),
            (12, 21)
        ])
        r.drawPolygon([
            (12, 26),
            (16, 26),
            (16, 28),
            (12, 28)
        ])

        r.drawCircle(18, 9, 6)
        r.setLineColor(0, 0, 0)
        r.setColor(0, 0, 0)
        r.drawCircle(18, 9, 2)

class AIGameEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Possible actions
    NUM_ACTIONS = 5
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2
    ACTION_BACK = 3
    ACTION_PICKUP = 4

    def __init__(self, gridSize=20):
        assert (gridSize >= 4)

        # For visual rendering
        self.renderer = None

        self.action_space = spaces.Discrete(AIGameEnv.NUM_ACTIONS)

        # The observations are RGB images
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape = IMG_ARRAY_SIZE
        )

        self.reward_range = (-1, 1000)

        # Environment configuration
        self.gridSize = gridSize
        self.maxSteps = 100
        self.startPos = (1, 1)

        # Initialize the state
        self.seed()
        self.reset()

    def _reset(self):
        # Place the agent in the starting position
        self.agentPos = self.startPos

        # Agent direction, initially pointing right (+x axis)
        self.agentDir = 0

        # Step count since episode start
        self.stepCount = 0

        # Last step the environment was rendered
        self.lastRender = None

        # Return first observation
        self.render()
        obs = self.renderer.getArray(IMG_ARRAY_SIZE)
        return obs

    def _seed(self, seed=None):
        """
        The seed function sets the random elements of the environment,
        and initializes the world.
        """

        gridSz = self.gridSize

        # Initialize the grid
        self.grid = [None] * gridSz * gridSz

        # Place walls around the edges
        for i in range(0, gridSz):
            self.setGrid(i, 0, Wall())
            self.setGrid(i, gridSz - 1, Wall())
            self.setGrid(0, i, Wall())
            self.setGrid(gridSz - 1, i, Wall())

        self.setGrid(4, 8, Ball('blue'))
        self.setGrid(12, 3, Key('yellow'))

        # Place a goal in the bottom-left corner
        self.setGrid(gridSz - 2, gridSz - 2, Goal())

        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def setGrid(self, i, j, v):
        assert i >= 0 and i < self.gridSize
        assert j >= 0 and j < self.gridSize
        self.grid[j * self.gridSize + i] = v

    def getGrid(self, i, j):
        assert i >= 0 and i < self.gridSize
        assert j >= 0 and j < self.gridSize
        return self.grid[j * self.gridSize + i]

    def getDirVec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        # Pointing right
        if self.agentDir == 0:
            return (1, 0)
        # Down (positive Y)
        elif self.agentDir == 1:
            return (0, 1)
        # Pointing left
        elif self.agentDir == 2:
            return (-1, 0)
        # Up (negative Y)
        elif self.agentDir == 3:
            return (0, -1)

        assert (False)

    def _step(self, action):
        self.stepCount += 1

        reward = 0
        done = False

        # Rotate left
        if action == AIGameEnv.ACTION_LEFT:
            self.agentDir -= 1
            if self.agentDir < 0:
                self.agentDir += 4

        # Rotate right
        elif action == AIGameEnv.ACTION_RIGHT:
            self.agentDir = (self.agentDir + 1) % 4

        # Move forward
        elif action == AIGameEnv.ACTION_FORWARD:
            u, v = self.getDirVec()
            newPos = (self.agentPos[0] + u, self.agentPos[1] + v)

            targetCell = self.getGrid(newPos[0], newPos[1])

            if targetCell == None or targetCell.canOverlap():
                self.agentPos = newPos
            elif targetCell.type == 'goal':
                done = True
                reward = 1000

        # Move backward
        elif action == AIGameEnv.ACTION_BACK:
            u, v = self.getDirVec()
            u *= -1
            v *= -1
            newPos = (self.agentPos[0] + u, self.agentPos[1] + v)
            if self.getGrid(newPos[0], newPos[1]) == None:
                self.agentPos = newPos

        # Pick up an item
        elif action == AIGameEnv.ACTION_PICKUP:
            # TODO
            u, v = self.getDirVec()
            itemPos = (self.agentPos[0] + u, self.agentPos[1] + v)
            pass

        else:
            assert False, "unknown action"

        if self.stepCount >= self.maxSteps:
            done = True

        # Render the environment to produce an observation
        self.render()
        obs = self.renderer.getArray(IMG_ARRAY_SIZE)

        return obs, reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            #if self.renderer:
            #    self.renderer.close()
            return

        width = self.gridSize * CELL_PIXELS
        height = self.gridSize * CELL_PIXELS

        if self.renderer is None:
            self.renderer = Renderer(width, height)

        # Avoid rendering the same environment state twice
        if self.lastRender == self.stepCount:
            return
        self.lastRender = self.stepCount

        r = self.renderer
        r.beginFrame()

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(1, self.gridSize):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, width, y)
        for colIdx in range(1, self.gridSize):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, height)

        # Draw the agent
        assert (self.agentDir >= 0 and self.agentDir < 4)
        r.push()
        r.translate(
            CELL_PIXELS * (self.agentPos[0] + 0.5),
            CELL_PIXELS * (self.agentPos[1] + 0.5)
        )
        r.rotate(self.agentDir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            ( 12,  0),
            (-12, -10)
        ])
        r.pop()

        # Render the grid
        for j in range(0, self.gridSize):
            for i in range(0, self.gridSize):
                cell = self.getGrid(i, j)
                if cell == None:
                    continue
                r.push()
                r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                cell.render(r)
                r.pop()

        r.endFrame()

        if mode == 'rgb_array':
            return r.getNumpyArray()

        return r
