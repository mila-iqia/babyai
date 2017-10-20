import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from gym_aigame.envs.rendering import *

CELL_PIXELS=32

class AIGameEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, gridSize=8):
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
        self.startPos = (0, 0)

        # Initialize the state
        self.reset()
        self.seed()

    def _reset(self):
        # Agent position
        self.agentPos = self.startPos

        # Agent direction, initially pointing up
        self.agentDir = 0

        # Step count since episode start
        self.stepCount = 0

        # TODO: need to render screen
        # Return first observation
        return np.array([])

    def _seed(self, seed=None):
        """
        The seed function sets the random elements of the environment.
        """

        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def _step(self, action):
        self.stepCount += 1

        reward = 0
        done = False

        # Rotate left
        if action == 0:
            pass

        # Rotate right
        elif action == 1:
            pass

        # Forward
        elif action == 2:
            pass

        # back
        elif action == 3:
            pass

        # FIXME: need to render screen
        obs = np.array([])

        if self.stepCount >= self.maxSteps:
            done = True

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
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([

            (0, 0),
            (10, 20),
            (20, 0)

        ])



        r.endFrame()

        # TODO: rgb_array, return numpy array
        return r
