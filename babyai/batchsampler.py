import numpy as np
import copy

class BatchSampler(object):
    """
    Class used to sample a batch of demonstrations from demonstrations of multiple
    environments based on a distribution.
    Used for Teacher Student Curriculum setting in imitation learning.
    """

    def __init__(self, demos, batch_size, seed, no_mem=False):
        self.num_task = len(demos)
        self.dist_task = np.ones(self.num_task) / self.num_task * 1.0
        self.demos = demos
        self.batch_size = batch_size
        self.no_mem = no_mem
        self.rng = np.random.RandomState(seed)

        self.total_demos = 0
        self.num_used_demos = 0
        self.current_demos = [None] * self.num_task
        self.current_ids = [None] * self.num_task
        for tid in range(self.num_task):
            self.total_demos += self.reset(tid)

        self.tracking_total_demos = self.total_demos

    def setDist(self, dist_task):
        self.dist_task = dist_task

    def reset(self, tid):
        demo = copy.deepcopy(self.demos[tid])
        np.random.shuffle(demo)
        self.current_demos[tid] = demo
        self.current_ids[tid] = 0

        return len(demo)

    def sample(self):

        batch = []
        for i in range(self.batch_size):
            tid = self.rng.choice(range(len(self.dist_task)), p=self.dist_task)
            cid = self.current_ids[tid]
            if cid >= len(self.current_demos[tid]):
                self.reset(tid)
                cid = self.current_ids[tid]

            batch += [self.current_demos[tid][cid]]
            self.current_ids[tid] += 1

        if self.no_mem:
            batch = np.array(batch)

        self.num_used_demos += self.batch_size
        should_evaluate = self.num_used_demos >= self.tracking_total_demos
        if should_evaluate:
            self.tracking_total_demos += self.total_demos
        return batch, should_evaluate