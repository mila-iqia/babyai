#!/usr/bin/env python3

from __future__ import division, print_function

import gym
import gym_aigame

from PyQt5.QtWidgets import QApplication, QWidget

def main():

    env = gym.make('AI-Game-v0')
    env.reset()

    #env.render()




if __name__ == "__main__":
    main()
