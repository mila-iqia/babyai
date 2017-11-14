import pickle
import gym
from gym import Wrapper

class Annotator(Wrapper):
    def __init__(self, env, saveOnClose=False):
        super(Annotator, self).__init__(env)

        # Dictionary keyed by agent position and direction
        self.annotations = {}

        # Set of previously visited states
        # This is so we don't give the same advice/reward twice
        self.visited = set()

        # Last agent state (before the last action taken)
        self.lastState = None

        # Last action taken
        self.lastAction = None

        self.saveOnClose = saveOnClose

        try:
            self.annotations = pickle.load(open("annotations.p", "rb"))
            print('Loaded annotations')
        except:
            print('No annotations found')

    def _close(self):
        super(Annotator, self)._close()

        if self.saveOnClose:
            print('Saving annotations')
            pickle.dump(self.annotations, open("annotations.p", "wb"))

    def getStepsRemaining(self):
        return self.env.getStepsRemaining()

    def setReward(self, val):
        """Sets the reward for the last state and action taken"""

        assert self.lastState is not None
        assert self.lastAction is not None, "can't set reward, no action taken yet"

        if self.lastState not in self.annotations:
            self.annotations[self.lastState] = { 'rewards': {}, 'advice': '' }

        ann = self.annotations[self.lastState]
        ann['rewards'][self.lastAction] = val

        print(self.lastState)
        print(self.lastAction)

    def setAdvice(self, text):
        """Set the advice to be provided to the agent in this state"""

        curState = (self.env.agentPos, self.env.agentDir)

        if curState not in self.annotations:
            self.annotations[curState] = { 'rewards': {}, 'advice': '' }

        self.annotations[curState]['advice'] = text

    def getCurState(self):

        carrying = None
        if self.env.carrying:
            carrying = (self.env.carrying.type, self.env.carrying.color)

        return (self.env.agentPos, self.env.agentDir, carrying)

    def _reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.visited = set()

        self.lastState = self.getCurState()
        self.lastAction = None

        return obs

    def _step(self, action):

        self.lastState = self.getCurState()
        self.lastAction = action

        obs, reward, done, info = self.env.step(action)

        # If there are annotations for the previous state
        if self.lastState in self.annotations:
            ann = self.annotations[self.lastState]

            stateAction = (self.lastState, action)

            # Override the reward if one is specified for this action
            if action in ann['rewards'] and stateAction not in self.visited:
                reward = ann['rewards'][action]
                #print('overriding reward: %s' % reward)

                # Mark this position and action combination as visited
                self.visited.add(stateAction)

            # Provide advice observation
            info['advice'] = ann['advice']

        return obs, reward, done, info
