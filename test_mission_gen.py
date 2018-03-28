#!/usr/bin/env python3

import random

from levels.instr_gen import gen_instr_seq, gen_surface
from levels.env_gen import gen_env
from levels.instrs import *
from levels.verifier import InstrSeqVerifier

def test():
    seed = random.randint(0, 0xFFFF)

    instr = [
        Instr(action="open", object=Object(type="door", color="red", loc=None, state="locked")),
        Instr(action="pickup", object=Object(type="key", color="green", loc=None, state=None)),
        Instr(action="open", object=Object(type="door", color="blue", loc=None, state=None)),
        Instr(action="drop", object=None)
    ]

    print(instr)
    print(gen_surface(instr))

    env = gen_env(instr, seed)
    verifier = InstrSeqVerifier(env, instr)

    def keyDownCb(keyName):
        if keyName == 'ESCAPE':
            env.gridRender.window.close()

        action = 0
        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward
        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop
        else:
            return

        _, _, _, _ = env.step(action)
        done = verifier.step()
        print(done)

    renderer = env.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    import time

    while True:
        time.sleep(0.01)
        env.render('human')
        if env.gridRender.window.closed:
            break

test()
