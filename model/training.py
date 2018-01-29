from collections import namedtuple, deque
from torch.autograd import Variable
import torch

def selectAction(obs):
    """
    Select the next action to be performed in an episode
    @state tuple containing (image, mission, advice)
    """

    # TODO
    print('selectAction: implement me!')

    image = obs['image']
    mission = obs['mission']

    print('image: %s' % str(image.shape))
    print('mission: %s' % mission)

    import random
    return random.randint(0, 3)
