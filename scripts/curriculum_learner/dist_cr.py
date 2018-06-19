from abc import ABC, abstractmethod
import numpy

class DistCreator(ABC):
    @abstractmethod
    def __call__(self, values):
        pass

class GreedyAmaxDistCreator(DistCreator):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        value_id = numpy.random.choice(numpy.flatnonzero(values == values.max()))
        dist = self.ε*numpy.ones(len(values))/len(values)
        dist[value_id] += 1-self.ε
        return dist

class PropDistCreator(DistCreator):
    ρ = 1e-8

    def __call__(self, values):
        assert numpy.all(values >= 0), "All values must be positive."

        values = values + self.ρ
        return values/numpy.sum(values)

class GreedyPropDistCreator(PropDistCreator):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        dist = super().__call__(values)
        uniform = numpy.ones(len(values))/len(values)
        return (1-self.ε)*dist + self.ε*uniform

class ClippedPropDistCreator(PropDistCreator):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        dist = super().__call__(values)
        n = len(dist)
        γ = numpy.amin(dist)
        if γ < self.ε/n:
            dist = (self.ε/n - 1/n)/(γ - 1/n)*(dist - 1/n) + 1/n
        return dist

class BoltzmannDistCreator(DistCreator):
    def __init__(self, τ):
        self.τ = τ
    
    def __call__(self, values):
        temperatured_values = numpy.exp(values/self.τ)
        return temperatured_values / numpy.sum(temperatured_values)