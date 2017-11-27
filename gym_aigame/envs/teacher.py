import pickle
import gym
from gym import Wrapper

class Teacher(Wrapper):
    def __init__(self, env):
        super(Teacher, self).__init__(env)

    def _close(self):
        super(Teacher, self)._close()

    def getStepsRemaining(self):
        return self.env.getStepsRemaining()

    def _reset(self, **kwargs):
        """
        Called at the start of an episode
        """

        obs = self.env.reset(**kwargs)

        return obs

    def _step(self, action):
        """
        Called at every action
        """

        obs, reward, done, info = self.env.step(action)

        # TODO
        if self.env.agentPos[0] < 5:
            info['advice'] = "go right"





        return obs, reward, done, info
