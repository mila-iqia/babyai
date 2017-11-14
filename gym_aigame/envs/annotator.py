import pickle
import gym
from gym import Wrapper

class Annotator(Wrapper):
    def __init__(self, env):
        super(Annotator, self).__init__(env)

        # Dictionary keyed by agent position and direction
        self.annotations = {}

        # Set of previously visited states
        # This is so we don't give the same advice/reward twice
        self.visited = set()

        # Last agent position (before the last action taken)
        self.lastPos = None

        # Last action taken
        self.lastAction = None

        try:
            self.annotations = pickle.load(open("annotations.p", "rb"))
            print('Loaded annotations')
        except:
            print('No annotations found')

    def _close(self):
        super(Annotator, self)._close()
        print('Saving annotations')

        pickle.dump(self.annotations, open("annotations.p", "wb"))

    def getStepsRemaining(self):
        return self.env.getStepsRemaining()

    def setReward(self, val):
        """Sets the reward for the last state and action taken"""

        assert self.lastPos is not None
        assert self.lastAction is not None, "can't set reward, no action taken yet"

        if self.lastPos not in self.annotations:
            self.annotations[self.lastPos] = { 'rewards': {}, 'advice': '' }

        ann = self.annotations[self.lastPos]
        ann['rewards'][self.lastAction] = val

        #print(self.lastPos)
        #print(self.lastAction)

    def setAdvice(self, text):
        """Set the advice to be provided to the agent in this state"""

        curPos = (self.env.agentPos, self.env.agentDir)

        if curPos not in self.annotations:
            self.annotations[curPos] = { 'rewards': {}, 'advice': '' }

        self.annotations[curPos]['advice'] = text

    def _reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.visited = set()

        self.lastPos = (self.env.agentPos, self.env.agentDir)
        self.lastAction = None

        return obs

    def _step(self, action):

        self.lastPos = (self.env.agentPos, self.env.agentDir)
        self.lastAction = action

        obs, reward, done, info = self.env.step(action)

        if self.lastPos in self.annotations:
            ann = self.annotations[self.lastPos]

            posAction = (self.lastPos, action)

            # Override the reward if one is specified for this action
            if action in ann['rewards'] and posAction not in self.visited:
                reward = ann['rewards'][action]
                #print('overriding reward: %s' % reward)

                # Mark this position and action combination as visited
                self.visited.add(posAction)

            # Provide advice observation
            info['advice'] = ann['advice']

        return obs, reward, done, info
