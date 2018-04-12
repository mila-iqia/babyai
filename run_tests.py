#!/usr/bin/env python3

import levels
import levels.instr_gen
import levels.verifier
import levels.levels
import agents

# NOTE: please make sure that tests are always deterministic

print('Testing instruction generation')
levels.instr_gen.test()

# TODO: verifier tests
# could potentially use a tiny environment for this
# something with an object in a fixed location, and short known action
# sequence to succeed

print('Testing levels, mission generation')
levels.levels.test()
