from collections import namedtuple

Instr = namedtuple('Instr', ['action', 'object'])
Object = namedtuple('Object', ['kind', 'color', 'loc'])

RecLoc = namedtuple('RelLoc', ['loc'])
AbsLoc = namedtuple('RelLoc', ['loc'])
