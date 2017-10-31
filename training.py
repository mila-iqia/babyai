from collections import namedtuple, deque
from ActionGenerator import *
from torch.autograd import Variable
import UsefulComputations as cp



# Minibatch size
BATCH_SIZE = 64

# Experience store capacity
EXP_STORE_CAP = 10000

# Interval at which to train the network
TRAIN_INTERVAL = 100

# Number of steps performed
stepCount = 0

State = namedtuple('State', ['image', 'mission', 'advice'])

Trans = namedtuple('Trans', ['state', 'action', 'nextState', 'reward'])

# Experience store
expStore = deque()

gen = ActionGenerator()


def selectAction(state):
    """
    Select the next action to be performed in an episode
    @state tuple containing (image, mission, advice)
    """

    print('selectAction')

    #preprocess the sentences
    mission=Variable(gen.dico.seq2matrix(state.mission))
    advice=Variable(gen.dico.seq2matrix(state.advice))
    img=Variable(cp.preProcessImage(state.image))
    #compute the action
    distribution = gen(img, mission, advice)
    action = int(torch.max(distribution, 1)[-1].data[0])
    #print(distribution)
    #print(action)

    # Return the index of the action to perform
    return action


def storeTrans(state, action, nextState, reward):
    """Store info associated with a state transition"""
    global stepCount

    t = Trans(state, action, nextState, reward)

    expStore.append(t)

    if len(expStore) > EXP_STORE_CAP:
        expStore.popleft()

    stepCount += 1

    if len(expStore) >= BATCH_SIZE and stepCount % TRAIN_INTERVAL == 0:
        train()


def train():
    # TODO: sample a minibatch from expStore







    pass
