from collections import namedtuple, deque
from torch.autograd import Variable
import torch
import UsefulComputations
import ActionGenerator


State = namedtuple('State', ['image', 'mission', 'advice'])

Trans = namedtuple('Trans', ['state', 'action', 'nextState', 'reward'])

model=ActionGenerator.ActionGenerator()


def selectAction(obs):
    """
    Select the next action to be performed in an episode
    @state tuple containing (image, mission, advice)
    """

    # TODO
    print('selectAction: implement me!')


    image = obs['image']
    mission = obs['mission']
    advice = obs['advice']
    
    

    #print('image: %s' % str(image.shape))
    #print('mission: %s' % mission)
    print('advice: %s' % advice)

    import random
    return random.randint(0, 3)


import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
print(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))