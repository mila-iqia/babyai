import torch
import utils


class Demonstrator:
    def __init__(self, demos):
        self.demos = demos
        self.current_instance = 0
        self.current_step = 0

    def move_to_next_instance(self):
        self.current_instance += 1
        self.current_step = 0
        if self.current_instance >= len(self.demos):
            return False
        return True

    def move_to_next_step(self):
        self.current_step += 1

    def get_observation(self):
        return self.demos[self.current_instance][self.current_step][0]

    def get_action(self):
        return self.demos[self.current_instance][self.current_step][1]