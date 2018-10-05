# ICLR19 Levels

The levels described in this file were created for the ICLR19 submission.
These form a curriculum that is subdivided according to specific competencies.

## GoToObj

Go to an object, inside a single room with no doors, no distractors

## GoToRedBall

Go to the red ball, single room, with obstacles.
The obstacles/distractors are all the same, to eliminate
perceptual complexity.

## GoToRedBallGrey

Go to the red ball, single room, with obstacles.
The obstacles/distractors are all grey boxes, to eliminate
perceptual complexity. No unblocking required.

## GoToLocal

Go to an object, inside a single room with no doors, no distractors.

## PutNextLocal

Put an object next to another object, inside a single room
with no doors, no distractors.

## PickUpLoc

Pick up an object which may be described using its location. This is a
single room environment.

Competencies: PickUp, Loc. No unblocking.

## GoToObjMaze

Go to an object, the object may be in another room. No distractors.

## GoTo

Go to an object, the object may be in another room. Many distractors.

## Pickup

Pick up an object, the object may be in another room.

## PickupUnblock

Pick up an object, the object may be in another room. The path may
be blocked by one or more obstructors.

## Open

Open a door, which may be in another room.

## Unlock

Maze environment where the agent has to retrieve a key to open a locked door.

Competencies: Maze, Open, Unlock. No unblocking.

## PutNext

Put an object next to another object. Either of these may be in another room.

## Synth

Union of all instructions from PutNext, Open, Goto and PickUp. The agent
may need to move objects around. The agent may have to unlock the door,
but only if it is explicitly referred by the instruction.

Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open

## SynthLoc

Like Synth, but a significant share of object descriptions involves
location language like in PickUpLoc. No implicit unlocking.
Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc

## GoToSeq

Sequencing of go-to-object commands.

Competencies: Maze, GoTo, Seq. No locked room. No locations. No unblocking.

## SynthSeq

Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.

Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq. No implicit unlocking.

## GoToImpUnlock

Go to an object, which may be in a locked room. No unblocking.

Competencies: Maze, GoTo, ImpUnlock

## BossLevel

Command can be any sentence drawn from the Baby Language grammar. Union of
all competencies. This level is a superset of all other levels.
