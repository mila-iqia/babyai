from abc import ABC, abstractmethod
import numpy

class DistComputer(ABC):
    """A distribution computer.

    It receives returns for some environments, updates the return history
    given by each environment and computes a distribution over
    environments given these histories of return."""

    def __init__(self, return_hists):
        self.return_hists = return_hists

        self.step = 0

    @abstractmethod
    def __call__(self, returns):
        self.step += 1
        for env_id, returnn in returns.items():
            self.return_hists[env_id].append(self.step, returnn)

class LpDistComputer(DistComputer):
    """A distribution computer based on learning progress.

    It associates an attention a_i to each environment i that is equal
    to the learning progress of this environment, i.e. a_i = lp_i."""

    def __init__(self, return_hists, compute_lp, create_dist):
        super().__init__(return_hists)

        self.return_hists = return_hists
        self.compute_lp = compute_lp
        self.create_dist = create_dist

    def __call__(self, returns):
        super().__call__(returns)

        self.lps = self.compute_lp()
        self.attentions = numpy.absolute(self.lps)
        dist = self.create_dist(self.attentions)

        return dist