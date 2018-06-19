from abc import ABC, abstractmethod
import numpy

class LpComputer(ABC):
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.timestep = 0
        self.timesteps = [[] for _ in range(self.num_envs)]
        self.returns = [[] for _ in range(self.num_envs)]
        self.lps = numpy.zeros((self.num_envs))

    def __call__(self, returns):
        self.timestep += 1
        for env_id, returnn in returns.items():
            self.timesteps[env_id].append(self.timestep)
            self.returns[env_id].append(returnn)
            self._compute_lp(env_id)
        return self.lps

    @abstractmethod
    def _compute_lp(self, env_id):
        pass        

class TSLpComputer(LpComputer):
    def __init__(self, num_envs, α):
        super().__init__(num_envs)

        self.α = α

    @abstractmethod
    def _compute_direct_lp(self, env_id):
        pass

    def _compute_lp(self, env_id):
        lp = self._compute_direct_lp(env_id)
        if lp is not None:
            self.lps[env_id] = self.α * lp + (1 - self.α) * self.lps[env_id]

class OnlineLpComputer(TSLpComputer):
    def _compute_direct_lp(self, env_id):
        returns = self.returns[env_id]
        if len(returns) >= 2:
            return returns[-1] - returns[-2]

class WindowLpComputer(TSLpComputer):
    def __init__(self, num_envs, α, K):
        super().__init__(num_envs, α)

        self.K = K

    def _compute_direct_lp(self, env_id):
        timesteps = self.timesteps[env_id][-self.K:]
        returns = self.returns[env_id][-self.K:]
        if len(timesteps) >= 2:
            return numpy.polyfit(timesteps, returns, 1)[0]

class LinregLpComputer(LpComputer):
    def __init__(self, num_envs, K):
        super().__init__(num_envs)

        self.K = K

    def _compute_lp(self, env_id):
        timesteps = self.timesteps[env_id][-self.K:]
        returns = self.returns[env_id][-self.K:]
        if len(timesteps) >= 2:
            self.lps[env_id] = numpy.polyfit(timesteps, returns, 1)[0]