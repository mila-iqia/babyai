# Bonus Levels

The levels described in this file were created prior to the ICLR19 publication.
We've chosen to keep these because they may be useful for curriculum learning
or for specific research projects.

Please note that these levels are not as widely tested as the ICLR19 levels.
If you run into problems, please open an issue on this repository.

In naming the levels we adhere to the following convention:
- `N2`, `N3`, `N4` refers to the number of objects in the room/environment
- `S2`, `S3`, `S4` refers to the size of the room/environment
- in `Debug` levels the episode is terminated once the agent does something unnecessary or fatally bad, for example
    - picks up an object which it is not supposed to pick up (unnecessary)
    - open the door that it is supposed to open _after_ another one (fatal)
- in `Carrying` levels the agent starts carrying the object of interest
- in `Dist` levels distractor objects are placed to confuse the agent

## OpenRedDoor

- Environment: The agent is placed in a room with a door.
- instruction: open the red door
- Evaluate: image understanding
- Level id: `BabyAI-OpenRedDoor-v0`

<p align="center"><img src="/media/OpenRedDoor.png" width="250"></p>

## OpenDoor

- Environment: The agent is placed in a room with 4 different doors. The environment is done when the instruction is executed in the regular mode or when a door is opened in the `debug` mode.
- instruction: open a door of:
    - a given color or location in `OpenDoor`
    - a given color in `OpenDoorColor`
    - a given location in `OpenDoorLoc`
- Evaluate: image & text understanding, memory in `OpenDoor` and `OpenDoorLoc`
- Level id:
    - `BabyAI-OpenDoor-v0`
    - `BabyAI-OpenDoorDebug-v0`
    - `BabyAI-OpenDoorColor-v0`
    - `BabyAI-OpenDoorColorDebug-v0`
    - `BabyAI-OpenDoorLoc-v0`
    - `BabyAI-OpenDoorLocDebug-v0`

<p align="center"><img src="/media/OpenDoor.png" width="250"></p>

## GoToDoor

- Environment: The agent is placed in a room with 4 different doors.
- Instruction: Go to a door of a given of a given color.
- Evaluate: image & text understanding
- Level id: `BabyAI-GoToDoor-v0`

## GoToObjDoor

- Environment: The agent is placed in a room with 4 different doors and 5 different objects.
- Instruction: Go to an object or a door of a given type and color
- Evaluate: image & text understanding
- Level id: `BabyAI-GoToObjDoor-v0`

<p align="center"><img src="/media/GoToObjDoor.png" width="250"></p>

## ActionObjDoor

- Environment: The agent is placed in a room with 4 different doors and 5 different objects.
- Instruction: [Pick up an object] or [go to an object or door] or [open a door]
- Evaluate: image & text understanding
- Level id: `BabyAI-ActionObjDoor-v0`

<p align="center"><img src="/media/ActionObjDoor.png" width="250"></p>

## UnlockPickup

- Environment: The agent is placed in a room with a key and a locked door. The door opens onto a room with a box. Rooms have either no distractors in `UnlockPickup` or 4 distractors in `UnlockPickupDist`.
- instruction: pick up an object of a given type and color
- Evaluate: image understanding, memory in `UnlockPickupDist`
- Level id: `BabyAI-UnlockPickup-v0`, `BabyAI-UnlockPickupDist-v0`

<p align="center">
    <img src="/media/UnlockPickup.png" width="250">
    <img src="/media/UnlockPickupDist.png" width="250">
</p>

## BlockedUnlockPickup

- Environment: The agent is placed in a room with a key and a locked door. The door is blocked by a ball. The door opens onto a room with a box.
- instruction: pick up the box
- Evaluate: image understanding
- Level id: `BabyAI-BlockedUnlockPickup-v0`

<p align="center"><img src="/media/BlockedUnlockPickup.png" width="250"></p>

## UnlockToUnlock

- Environment: The agent is placed in a room with a key of color A and two doors of color A and B. The door of color A opens onto a room with a key of color B. The door of color B opens onto a room with a ball.
- instruction: pick up the ball
- Evaluate: image understanding
- Level id: `BabyAI-UnlockToUnlock-v0`

<p align="center"><img src="/media/UnlockToUnlock.png" width="250"></p>

## KeyInBox

- Environment: The agent is placed in a room with a box containing a key and a locked door.
- instruction: open the door
- Evaluate: image understanding
- Level id: `BabyAI-KeyInBox-v0`

<p align="center"><img src="/media/KeyInBox.png" width="250"></p>

## PickupDist

- Environment: The agent is placed in a room with 5 objects. The environment is done when the instruction is executed in the regular mode or when any object is picked in the `debug` mode.
- instruction: pick up an object of a given type and color
- Evaluate: image & text understanding
- Level id:
    - `BabyAI-PickupDist-v0`
    - `BabyAI-PickupDistDebug-v0`

<p align="center"><img src="/media/PickupDist.png" width="250"></p>

## PickupAbove

- Environment: The agent is placed in the middle room. An object is placed in the top-middle room.
- instruction: pick up an object of a given type and color
- Evaluate: image & text understanding, memory
- Level id: `BabyAI-PickupAbove-v0`

<p align="center"><img src="/media/PickupAbove.png" width="250"></p>

## OpenRedBlueDoors

- Environment: The agent is placed in a room with a red door and a blue door facing each other. The environment is done when the instruction is executed in the regular mode or when the blue door is opened in the `debug` mode.
- instruction: open the red door then open the blue door
- Evaluate: image understanding, memory
- Level id:
    - `BabyAI-OpenRedBlueDoors-v0`
    - `BabyAI-OpenRedBlueDoorsDebug-v0`

<p align="center"><img src="/media/OpenRedBlueDoors.png" width="250"></p>

## OpenTwoDoors

- Environment: The agent is placed in a room with a red door and a blue door facing each other. The environment is done when the instruction is executed in the regular mode or when the second door is opened in the `debug` mode.
- instruction: open the door of color X then open the door of color Y
- Evaluate: image & text understanding, memory
- Level id:
    - `BabyAI-OpenTwoDoors-v0`
    - `BabyAI-OpenTwoDoorsDebug-v0`

<p align="center"><img src="/media/OpenTwoDoors.png" width="250"></p>

## FindObj

- Environment: The agent is placed in the middle room. An object is placed in one of the rooms. Rooms have a size of 5 in `FindObjS5`, 6 in `FindObjS6` or 7 in `FindObjS7`.
- instruction: pick up an object of a given type and color
- Evaluate: image understanding, memory
- Level id:
    - `BabyAI-FindObjS5-v0`
    - `BabyAI-FindObjS6-v0`
    - `BabyAI-FindObjS7-v0`

<p align="center">
    <img src="/media/FindObjS5.png" width="250">
    <img src="/media/FindObjS6.png" width="250">
    <img src="/media/FindObjS7.png" width="250">
</p>

## FourObjs

- Environment: The agent is placed in the middle room. 4 different objects are placed in the adjacent rooms. Rooms have a size of 5 in `FourObjsS5`, 6 in `FourObjsS6` or 7 in `FourObjsS7`.
- instruction: pick up an object of a given type and location
- Evaluate: image understanding, memory
- Level id:
    - `BabyAI-FourObjsS5-v0`
    - `BabyAI-FourObjsS6-v0`
    - `BabyAI-FourObjsS7-v0`

<p align="center">
    <img src="/media/FourObjsS5.png" width="250">
    <img src="/media/FourObjsS6.png" width="250">
    <img src="/media/FourObjsS7.png" width="250">
</p>

## KeyCorridor

- Environment: The agent is placed in the middle of the corridor. One of the rooms is locked and contains a ball. Another room contains a key for opening the previous one. The level is split into a curriculum starting with one row of 3x3 rooms, going up to 3 rows of 6x6 rooms.
- instruction: pick up an object of a given type
- Evaluate: image understanding, memory
- Level ids:
  - `BabyAI-KeyCorridorS3R1-v0`
  - `BabyAI-KeyCorridorS3R2-v0`
  - `BabyAI-KeyCorridorS3R3-v0`
  - `BabyAI-KeyCorridorS4R3-v0`
  - `BabyAI-KeyCorridorS5R3-v0`
  - `BabyAI-KeyCorridorS6R3-v0`

<p align="center">
    <img src="/media/KeyCorridorS3R1.png" width="250">
    <img src="/media/KeyCorridorS3R2.png" width="250">
    <img src="/media/KeyCorridorS3R3.png" width="250">
    <img src="/media/KeyCorridorS4R3.png" width="250">
    <img src="/media/KeyCorridorS5R3.png" width="250">
    <img src="/media/KeyCorridorS6R3.png" width="250">
</p>

## 1Room

- Environment: The agent is placed in a room with a ball. The level is split into a curriculum with rooms of size 8, 12, 16 or 20.
- instruction: pick up the ball
- Evaluate: image understanding, memory
- Level ids:
  - `BabyAI-1RoomS8-v0`
  - `BabyAI-1RoomS12-v0`
  - `BabyAI-1RoomS16-v0`
  - `BabyAI-1RoomS20-v0`

<p align="center">
    <img src="/media/1RoomS8.png" width="250">
    <img src="/media/1RoomS12.png" width="250">
    <img src="/media/1RoomS16.png" width="250">
    <img src="/media/1RoomS20.png" width="250">
</p>

## OpenDoorsOrder

- Environment: There are two or four doors in a room. The agent has to open
 one or two of the doors in a given order.
- Instruction:
  - open the X door
  - open the X door and then open the Y door
  - open the X door after you open the Y door
- Level ids:
  - `BabyAI-OpenDoorsOrderN2-v0`
  - `BabyAI-OpenDoorsOrderN4-v0`
  - `BabyAI-OpenDoorsOrderN2Debug-v0`
  - `BabyAI-OpenDoorsOrderN4Debug-v0`

## PutNext

- Environment: Single room with multiple objects. One of the objects must be moved next to another specific object.
- instruction: put the X next to the Y
- Level ids:
  - `BabyAI-PutNextS4N1-v0`
  - `BabyAI-PutNextS5N1-v0`
  - `BabyAI-PutNextS6N2-v0`
  - `BabyAI-PutNextS6N3-v0`
  - `BabyAI-PutNextS7N4-v0`
  - `BabyAI-PutNextS6N2Carrying-v0`
  - `BabyAI-PutNextS6N3Carrying-v0`
  - `BabyAI-PutNextS7N4Carrying-v0`

## MoveTwoAcross

- Environment: Two objects must be moved so that they are next to two other objects. This task is structured to have a very large number of possible instructions.
- instruction: put the A next to the B and the C next to the D
- Level ids:
  - `BabyAI-MoveTwoAcrossS5N2-v0`
  - `BabyAI-MoveTwoAcrossS8N9-v0`
