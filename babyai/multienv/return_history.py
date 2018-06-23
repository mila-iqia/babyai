from abc import ABC, abstractmethod
import numpy

class ReturnHistory:
    """The return history.

    It tracks the return given by an environment over time."""

    def __init__(self):
        self.steps = []
        self.returns = []

    def append(self, step, returnn):
        self.steps.append(step)
        self.returns.append(returnn)

    def __getitem__(self, index):
        return self.steps[index], self.returns[index]