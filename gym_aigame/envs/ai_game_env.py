import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np

def drawQuad(cX, cY, w):
    verts = (
        cX - W, cY - w,
        cX - W, cY + w,
        cX + W, cY + w,
        cX + W, cY - w,
    )

    pyglet.graphics.draw(
        4,
        pyglet.gl.GL_LINE_LOOP,
        ('v2i', verts)
    )

def drawCircle(cX, cY, r, n):
    verts = []
    for i in range(0, n):
        a = (i / n) * 2 * math.pi
        x = cX + r * math.cos(a)
        y = cY + r * math.sin(a)
        verts.append(x)
        verts.append(y)

    pyglet.graphics.draw(
        n,
        pyglet.gl.GL_LINE_LOOP,
        ('v2f', verts)
    )

def drawArrow(cX, cY, dir, r):
    """
    Draw an equilateral triangle pointing in a given direction
    dir: direction angle in radians
    """

    pyglet.gl.glPushMatrix()

    pyglet.gl.glTranslatef(cX, cY, 0)
    angleDegs = 360 * dir / (2 * math.pi)
    pyglet.gl.glRotatef(angleDegs, 0, 0, 1)

    verts = (
        -r,  r * 0.75,
         r,  0,
        -r, -r * 0.75
    )
    pyglet.graphics.draw(
        3,
        pyglet.gl.GL_LINE_LOOP,
        ('v2f', verts)
    )

    pyglet.gl.glPopMatrix()

def dst2D(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

# Rendering window size
WINDOW_SIZE = 512

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
        # For rendering
        self.window = None

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
        #    if self.window:
        #        self.window.close()
        #    return

        #if self.window is None:
        #    self.window = pyglet.window.Window(width=WINDOW_SIZE, height=WINDOW_SIZE)

        pass
