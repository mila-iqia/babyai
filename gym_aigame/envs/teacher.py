import pickle
import gym
from gym import Wrapper

class Teacher(Wrapper):
    def __init__(self, env):
        super(Teacher, self).__init__(env)

    def _close(self):
        super(Teacher, self)._close()

    def _reset(self, **kwargs):
        """
        Called at the start of an episode
        """

        obs = self.env.reset(**kwargs)

        return {
            "image": obs,
            "advice": "open the first door"
        }

    def _step(self, action):
        """
        Called at every action
        """

        obs, reward, done, info = self.env.step(action)

        if self.env.agentPos[0] < 5:
            advice = "go right"
        else:
            advice = "get to the goal!"

        obs = {
            "image": obs,
            "advice": advice
        }

        return obs, reward, done, info
