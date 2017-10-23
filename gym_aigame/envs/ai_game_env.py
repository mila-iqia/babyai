import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from gym_aigame.envs.rendering import *

CELL_PIXELS = 32

IMG_ARRAY_SIZE = (100,100,3)

class AIGameEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    def __init__(self, gridSize=8):
        assert (gridSize >= 4)

        # For visual rendering
        self.renderer = None

        self.action_space = spaces.Discrete(4)

        sizePixels = gridSize * CELL_PIXELS
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape= (sizePixels, sizePixels, 3)
        )

        self.reward_range = (-1, 1000)

        # Environment configuration
        self.gridSize = gridSize
        self.maxSteps = 100
        self.startPos = (1, 1)

        # Initialize the grid
        self.grid = [None] * gridSize * gridSize

        # TODO: May want to move this to seed function?
        # Could store original grid for reset
        # Place walls around the edges
        for i in range(0, gridSize):
            self.setGrid(i, 0, 'WALL')
            self.setGrid(i, gridSize - 1, 'WALL')
            self.setGrid(0, i, 'WALL')
            self.setGrid(gridSize - 1, i, 'WALL')

        self.setGrid(gridSize - 2, gridSize - 2, 'GOAL')

        # Initialize the state
        self.reset()
        self.seed()

    def _reset(self):
        # Agent position
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
        The seed function sets the random elements of the environment.
        """

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
        if action == 0:
            self.agentDir -= 1
            if self.agentDir < 0:
                self.agentDir += 4

        # Rotate right
        elif action == 1:
            self.agentDir = (self.agentDir + 1) % 4

        # Forward
        elif action == 2:
            u, v = self.getDirVec()
            newPos = (self.agentPos[0] + u, self.agentPos[1] + v)
            if self.getGrid(newPos[0], newPos[1]) == None:
                self.agentPos = newPos

        # Back
        elif action == 3:
            u, v = self.getDirVec()
            u *= -1
            v *= -1
            newPos = (self.agentPos[0] + u, self.agentPos[1] + v)
            if self.getGrid(newPos[0], newPos[1]) == None:
                self.agentPos = newPos

        if self.stepCount >= self.maxSteps:
            done = True

        # Render the environment to produce an observation
        self.render()
        obs = self.renderer.getArray(IMG_ARRAY_SIZE)

        return obs, reward, done, {}

    def _render(self, mode='human', close=False):
        #if close:
        #    if self.renderer:
        #        self.renderer.close()
        #    return

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
                if cell == 'WALL':
                    r.setLineColor(100, 100, 100)
                    r.setColor(100, 100, 100)
                    r.drawPolygon([
                        ((i+0) * CELL_PIXELS, (j+1) * CELL_PIXELS),
                        ((i+1) * CELL_PIXELS, (j+1) * CELL_PIXELS),
                        ((i+1) * CELL_PIXELS, (j+0) * CELL_PIXELS),
                        ((i+0) * CELL_PIXELS, (j+0) * CELL_PIXELS)
                    ])
                elif cell == 'GOAL':
                    r.setLineColor(0, 255, 0)
                    r.setColor(0, 255, 0)
                    r.drawCircle(CELL_PIXELS * (i+0.5), CELL_PIXELS * (j+0.5), 10)
                else:
                    assert cell == None

        r.endFrame()

        # TODO: rgb_array, return numpy array
        return r
