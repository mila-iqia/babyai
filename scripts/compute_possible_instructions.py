#!/usr/bin/env python3

"""
Compute the number of possible instructions in the BabyAI grammar.
"""

from gym_minigrid.minigrid import COLOR_NAMES

def count_Sent():
    return (
        count_Sent1() +
        # Sent1, then Sent1
        count_Sent1() * count_Sent1() +
        # Sent1 after you Sent1
        count_Sent1() * count_Sent1()
    )

def count_Sent1():
    return (
        count_Clause() +
        # Clause and Clause
        count_Clause() * count_Clause()
    )

def count_Clause():
    return (
        # go to
        count_Descr() +
        # pick up
        count_DescrNotDoor() +
        # open
        count_DescrDoor() +
        # put next
        count_DescrNotDoor() * count_Descr()
    )

def count_DescrDoor():
    # (the|a) Color door Location
    return 2 * count_Color() * count_LocSpec()
def count_DescrBall():
    return count_DescrDoor()
def count_DescrBox():
    return count_DescrDoor()
def count_DescrKey():
    return count_DescrDoor()
def count_Descr():
    return count_DescrDoor() + count_DescrBall() + count_DescrBox() + count_DescrKey()
def count_DescrNotDoor():
    return count_DescrBall() + count_DescrBox() + count_DescrKey()

def count_Color():
    # Empty string or color
    return len([None] + COLOR_NAMES)

def count_LocSpec():
    # Empty string or location
    return len([None, 'left', 'right', 'front', 'behind'])

print('DescrKey: ', count_DescrKey())
print('Descr: ', count_Descr())
print('DescrNotDoor: ', count_DescrNotDoor())
print('Clause: ', count_Clause())
print('Sent1: ', count_Sent1())
print('Sent: ', count_Sent())
print('Sent: {:.3g}'.format(count_Sent()))
