#!/usr/bin/env python3

from levels.instr_gen import gen_instr, gen_instr_surface
from levels.env_gen import gen_env
from levels.instrs import *
from levels.verifier import InstrVerifier

def test():

    import random

    seed = random.randint(0, 0xFFFF)

    """
    instrs = [
        Instr([AInstr(action="pick", object=Object(color="red", loc=RelLoc('front'), type="key"))]),
        #Instr([AInstr(action="pick", object=Object(color="blue", loc=RelLoc('front'), type="key"))]),

        Instr([AInstr(action="pick", object=Object(color=None, loc=RelLoc('left'), type="ball"))]),

        #Instr([AInstr(action="drop", object=Object(color="red", loc=None, type="key"))]),
        #Instr([AInstr(action="goto", object=Object(color="blue", loc=AbsLoc('north'), type="ball"))]),
        #Instr([AInstr(action="goto", object=Object(color="green", loc=AbsLoc('east'), type="box"))]),
        #Instr([AInstr(action="open", object=Object(color="yellow", loc=AbsLoc('north'), type="door"))]),
    ]
    """

    # while True:
    #     instr = gen_instr()
    #     if instr.ainstrs[0].object.type == 'door':
    #         break
    # instr = Instr([
    #     AInstr(action='open', object=Object(type='door', color="red", loc=None, state=None)),
    #     AInstr(action='open', object=Object(type='door', color="blue", loc=None, state=None)),
    # ])
    instr = Instr([
        AInstr(action='open', object=Object(type='door', color=None, loc=None, state=None)),
    ])

    print(instr)
    print(gen_instr_surface(instr, seed))

    env = gen_env(instr, seed)
    verifier = InstrVerifier(env, instr)

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
        elif keyName == 'CTRL':
            action = env.actions.wait
        else:
            return

        done = verifier.isTerminalAction(action)
        print(done)
        _, _, _, _ = env.step(action)

    renderer = env.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    import time    

    while True:
        time.sleep(0.01)
        env.render('human')
        if env.gridRender.window.closed:
            break

test()
