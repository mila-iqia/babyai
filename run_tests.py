#!/usr/bin/env python3

"""
Run basic BabyAI level tests
Note: there are other automated tests in .circleci/config.yml
"""

import babyai
from babyai import levels

# NOTE: please make sure that tests are always deterministic

print('Testing levels, mission generation')
levels.test()
