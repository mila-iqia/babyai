import gym
from .verifier import *
from .levelgen import *


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(seed=seed)


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25
        )


# Register the levels in this file
register_levels(globals())
