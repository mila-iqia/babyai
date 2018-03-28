from collections import namedtuple

# Note: we may want to have an implicit parameter to distinguish between
# locations that are near or far. This way, the difficulty level would be
# implicitly represented in instructions.

# action: goto, open, pick, drop
Instr = namedtuple('Instr', ['action', 'object'])

# loc: RelLoc, AbsLoc
# state: locked
Object = namedtuple('Object', ['type', 'color', 'loc', 'state'])

# Relative locations
# loc: left, right, front, behind
RelLoc = namedtuple('RelLoc', ['loc'])

# Absolute locations
# loc: north, south, east, west
AbsLoc = namedtuple('AbsLoc', ['loc'])
