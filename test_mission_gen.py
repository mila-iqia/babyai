#!/usr/bin/env python3

from levels.instr_gen import gen_instr, surface
from levels.env_gen import gen_env
from levels.instrs import *

def test():

    import random

    seed = random.randint(0, 0xFFFF)

    """
    instrs = [
        Instr(action="pick", object=Object(color="red", loc=RelLoc('front'), type="key")),
        #Instr(action="pick", object=Object(color="blue", loc=RelLoc('front'), type="key")),

        Instr(action="pick", object=Object(color=None, loc=RelLoc('left'), type="ball")),

        #Instr(action="drop", object=Object(color="red", loc=None, type="key")),
        #Instr(action="goto", object=Object(color="blue", loc=AbsLoc('north'), type="ball")),
        #Instr(action="goto", object=Object(color="green", loc=AbsLoc('east'), type="box")),
        #Instr(action="open", object=Object(color="yellow", loc=AbsLoc('north'), type="door")),
    ]
    """

    instr = gen_instr()

    print(instr)
    print(surface(instr))

    env = gen_env([instr], seed)

    def keyDownCb(keyName):
        if keyName == 'ESCAPE':
            env.gridRender.window.close()
    renderer = env.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        if env.gridRender.window.closed:
            break

test()
