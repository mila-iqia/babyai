from collections import namedtuple

# Note: we may want to have an implicit parameter to distinguish between
# locations that are near or far. This way, the difficulty level would be
# implicitly represented in instructions.

# action: goto, open, pick, drop
Instr = namedtuple('Instr', ['action', 'object'])

# loc: loc_rel (ie. left, right, front, behind)
# state: locked
Object = namedtuple('Object', ['type', 'color', 'loc', 'state'])