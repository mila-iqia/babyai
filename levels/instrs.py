from collections import namedtuple

# Note: we may want to have an implicit parameter to distinguish between
# locations that are near or far. This way, the difficulty level would be
# implicitly represented in instructions.

# Actions: goto, open, pick, drop
Instr = namedtuple('Instr', ['action', 'object'])

Object = namedtuple('Object', ['type', 'color', 'loc', 'state'])

# Relative locations: left, right, front, behind
RelLoc = namedtuple('RelLoc', ['loc'])

# Absolute locations: north, south, east, west
AbsLoc = namedtuple('RelLoc', ['loc'])
