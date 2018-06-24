from abc import ABC, abstractmethod
import numpy

class DistCreator(ABC):
    """A distribution creator.

    It creates a distribution [p_1, ..., p_N] from values
    [v_1, ..., v_N]."""

    @abstractmethod
    def __call__(self, values):
        pass

class GreedyAmaxDistCreator(DistCreator):
    """A greedy argmax-based distribution creator.

    If values = [v_1, ..., v_N], then it creates the distribution
    [p_1, ..., p_N] where:
    - p_i = 1 - ε/N if v_i is the greatest value,
    - p_i = ε/N otherwise."""

    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        value_id = numpy.random.choice(numpy.flatnonzero(values == values.max()))
        dist = self.ε*numpy.ones(len(values))/len(values)
        dist[value_id] += 1-self.ε
        return dist

class PropDistCreator(DistCreator):
    """A proportionality-based distribution creator.

    If values = [v_1, ..., v_N], then it creates the distribution
    [p_1, ..., p_N] where p_i = v_i / (v_1 + ... + v_N)."""

    ρ = 1e-8

    def __call__(self, values):
        assert numpy.all(values >= 0), "All values must be positive."

        values = values + self.ρ
        return values/numpy.sum(values)

class GreedyPropDistCreator(PropDistCreator):
    """A greedy proportionality-based distribution creator.

    If values = [v_1, ..., v_N] and q is the distribution created by
    PropDistCreator from values, then it creates the distribution
    p = (1-ε)*q + ε*u where u is the uniform distribution."""

    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        dist = super().__call__(values)
        uniform = numpy.ones(len(values))/len(values)
        return (1-self.ε)*dist + self.ε*uniform

class BoltzmannDistCreator(DistCreator):
    """A Boltzmann-based distribution creator.

    If values = [v_1, ..., v_N], then it creates the distribution
    [p_1, ..., p_N] where p_i = exp(v_i/τ) / Σ exp(v_j/τ)."""

    def __init__(self, τ):
        self.τ = τ

    def __call__(self, values):
        temperatured_values = numpy.exp(values/self.τ)
        return temperatured_values / numpy.sum(temperatured_values)