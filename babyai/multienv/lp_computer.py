from abc import ABC, abstractmethod
import numpy

class LpComputer(ABC):
    """A learning progress computer.

    It computes the learning progress on each environment
    from the return history."""

    def __init__(self, return_hists):
        self.return_hists = return_hists

        self.num_envs = len(self.return_hists)
        self.lps = numpy.zeros((self.num_envs))

    def __call__(self):
        for env_id in range(self.num_envs):
            self._compute_lp(env_id)
        return self.lps

    @abstractmethod
    def _compute_lp(self, env_id):
        pass

class TSLpComputer(LpComputer):
    """A learning progress computer for Teacher-Student
    ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))
    learning progress computers.

    It computes an exponential moving average of the immediate
    learning progress."""

    def __init__(self, num_envs, α):
        super().__init__(num_envs)

        self.α = α

    @abstractmethod
    def _compute_immediate_lp(self, env_id):
        pass

    def _compute_lp(self, env_id):
        lp = self._compute_immediate_lp(env_id)
        if lp is not None:
            self.lps[env_id] = self.α * lp + (1 - self.α) * self.lps[env_id]

class OnlineLpComputer(TSLpComputer):
    """The online learning progress computer from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def _compute_immediate_lp(self, env_id):
        steps, returns = self.return_hists[env_id][-2:]
        if len(returns) >= 2:
            return numpy.polyfit(steps, returns, 1)[0]

class WindowLpComputer(TSLpComputer):
    """The window learning progress computer from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def __init__(self, num_envs, α, K):
        super().__init__(num_envs, α)

        self.K = K

    def _compute_immediate_lp(self, env_id):
        steps, returns = self.return_hists[env_id][-self.K:]
        if len(steps) >= 2:
            return numpy.polyfit(steps, returns, 1)[0]

class LinregLpComputer(LpComputer):
    """A learning progress computer using the immediate learning progress.

    It is similar to WindowLpComputer except that the learning progress
    is the immediate learning progress instead of an exponential moving
    average of it."""

    def __init__(self, num_envs, K):
        super().__init__(num_envs)

        self.K = K

    def _compute_lp(self, env_id):
        steps, returns = self.return_hists[env_id][-self.K:]
        if len(steps) >= 2:
            self.lps[env_id] = numpy.polyfit(steps, returns, 1)[0]