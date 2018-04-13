#!/usr/bin/env python3

import levels
import levels.instr_gen
import levels.verifier
import levels.levels
import agents

# NOTE: please make sure that tests are always deterministic

print('Testing levels, mission generation')
levels.levels.test()
