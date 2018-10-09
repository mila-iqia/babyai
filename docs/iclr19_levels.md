# ICLR19 Levels

The levels described in this file were created for the ICLR19 submission.
These form a curriculum that is subdivided according to specific competencies.

## GoToObj

Go to an object, inside a single room with no doors, no distractors.

<p align="center"><img src="/media/GoToObj.png" width="180"></p>

## GoToRedBall

Go to the red ball, single room, with obstacles.
The obstacles/distractors are all the same, to eliminate
perceptual complexity.

<p align="center"><img src="/media/GoToRedBall.png" width="180"></p>

## GoToRedBallGrey

Go to the red ball, single room, with obstacles.
The obstacles/distractors are all grey boxes, to eliminate
perceptual complexity. No unblocking required.

<p align="center"><img src="/media/GoToRedBallGrey.png" width="180"></p>

## GoToLocal

Go to an object, inside a single room with no doors, no distractors.

<p align="center"><img src="/media/GoToLocal.png" width="180"></p>

## PutNextLocal

Put an object next to another object, inside a single room
with no doors, no distractors.

<p align="center"><img src="/media/PutNextLocal.png" width="180"></p>

## PickUpLoc

Pick up an object which may be described using its location. This is a
single room environment.

Competencies: PickUp, Loc. No unblocking.

<p align="center"><img src="/media/PickupLoc.png" width="180"></p>

## GoToObjMaze

Go to an object, the object may be in another room. No distractors.

<p align="center"><img src="/media/GoToObjMaze.png" width="400"></p>

## GoTo

Go to an object, the object may be in another room. Many distractors.

<p align="center"><img src="/media/GoTo.png" width="400"></p>

## Pickup

Pick up an object, the object may be in another room.

<p align="center"><img src="/media/Pickup.png" width="400"></p>

## UnblockPickup

Pick up an object, the object may be in another room. The path may
be blocked by one or more obstructors.

<p align="center"><img src="/media/UnblockPickup.png" width="400"></p>

## Open

Open a door, which may be in another room.

<p align="center"><img src="/media/Open.png" width="400"></p>

## Unlock

Maze environment where the agent has to retrieve a key to open a locked door.

Competencies: Maze, Open, Unlock. No unblocking.

<p align="center"><img src="/media/Unlock.png" width="400"></p>

## PutNext

Put an object next to another object. Either of these may be in another room.

<p align="center"><img src="/media/PutNext.png" width="400"></p>

## Synth

Union of all instructions from PutNext, Open, Goto and PickUp. The agent
may need to move objects around. The agent may have to unlock the door,
but only if it is explicitly referred by the instruction.

Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open

<p align="center"><img src="/media/Synth.png" width="400"></p>

## SynthLoc

Like Synth, but a significant share of object descriptions involves
location language like in PickUpLoc. No implicit unlocking.
Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc

<p align="center"><img src="/media/SynthLoc.png" width="400"></p>

## GoToSeq

Sequencing of go-to-object commands.

Competencies: Maze, GoTo, Seq. No locked room. No locations. No unblocking.

<p align="center"><img src="/media/GoToSeq.png" width="400"></p>

## SynthSeq

Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.

Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq. No implicit unlocking.

<p align="center"><img src="/media/SynthSeq.png" width="400"></p>

## GoToImpUnlock

Go to an object, which may be in a locked room. No unblocking.

Competencies: Maze, GoTo, ImpUnlock

<p align="center"><img src="/media/GoToImpUnlock.png" width="400"></p>

## BossLevel

Command can be any sentence drawn from the Baby Language grammar. Union of
all competencies. This level is a superset of all other levels.

<p align="center"><img src="/media/BossLevel.png" width="400"></p>
