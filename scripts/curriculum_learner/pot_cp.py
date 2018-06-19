from abc import ABC, abstractmethod
import numpy

class PotComputer(ABC):
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.returns = [[] for _ in range(self.num_envs)]
        self.pots = numpy.zeros((self.num_envs))
    
    def __call__(self, returns):
        for env_id, returnn in returns.items():
            self.returns[env_id].append(returnn)
            self._compute_pot(env_id)
        return self.pots
    
    @abstractmethod
    def _compute_pot(self, env_id):
        pass

class VariablePotComputer(PotComputer):
    def __init__(self, num_envs, K, returns=None, max_returns=None):
        super().__init__(num_envs)

        returns = [float("+inf")]*self.num_envs if returns is None else returns
        self.max_returns = [float("-inf")]*self.num_envs if max_returns is None else max_returns
        self.K = K

        for env_id in range(len(self.pots)):
            returnn = returns[env_id]
            max_return = self.max_returns[env_id]
            self.pots[env_id] = max(max_return - returnn, 0)

    def _compute_pot(self, env_id):
        returns = self.returns[env_id][-self.K:]
        returnn = numpy.mean(returns)
        max_return = max(self.max_returns[env_id], returnn)
        self.pots[env_id] = max_return - returnn
        if len(returns) >= self.K:
            self.max_returns[env_id] = max_return