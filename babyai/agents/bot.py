import numpy
import time

from gym_minigrid.minigrid import *
from babyai.levels.verifier import *

class Bot:
    """
    Heuristic-based level solver
    """

    def __init__(self, mission):
        # Mission to be solved
        self.mission = mission

        grid_size = mission.grid_size

        # Grid containing what has been mapped out
        self.grid = Grid(grid_size, grid_size)

        # Visibility mask. True for explored/seen, false for unexplored.
        self.vis_mask = np.zeros(shape=(grid_size, grid_size), dtype=np.bool)

        # Stack of tasks/subtasks to complete (tuples)
        self.stack = []

        # Process/parse the instructions
        self.process_instr(mission.instrs)

        #for subgoal, datum in self.stack:
        #    print(subgoal)
        #    if datum:
        #        print(datum.surface(self.mission))

    def process_instr(self, instr):
        """
        Translate instructions into an internal form the agent can execute
        """

        if isinstance(instr, GoToInstr):
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(('Open', instr.desc))
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, PickupInstr):
            self.stack.append(('Pickup', instr.desc))
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(('Drop', None))
            self.stack.append(('GoToAdjPos', instr.desc_fixed))
            self.stack.append(('Pickup', None))
            self.stack.append(('GoToObj', instr.desc_move))
            return

        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr):
            self.process_instr(instr.instr_b)
            self.process_instr(instr.instr_a)
            return

        if isinstance(instr, AfterInstr):
            self.process_instr(instr.instr_a)
            self.process_instr(instr.instr_b)
            return

        assert False, "unknown instruction type"

    def step(self):
        """
        Take an observation and produce an action as output
        """

        # Process the current observation
        self.process_obs()

        # Iterate until we have an action to perform
        while True:
            action = self._iterate()
            if action is not None:
                return action

    def _iterate(self):
        """
        Perform one iteration of the internal control loop
        Returns either an action to perform or None
        """

        pos = self.mission.agent_pos
        dir_vec = self.mission.dir_vec
        right_vec = self.mission.right_vec
        fwd_pos = pos + dir_vec

        actions = self.mission.actions
        carrying = self.mission.carrying

        assert len(self.stack) > 0, "stack empty"

        # Get the topmost instruction on the stack
        subgoal, datum = self.stack[-1]

        # Open a door
        if subgoal == 'Open':
            fwd_cell = self.mission.grid.get(*fwd_pos)
            assert fwd_cell
            assert fwd_cell.type == "door" or fwd_cell.type == "locked_door"

            # If the door is locked, go find the key and then return
            if fwd_cell.type == 'locked_door':
                if not carrying or carrying.type != 'key' or carrying.color != fwd_cell.color:
                    key_desc = ObjDesc('key', fwd_cell.color)
                    key_desc.find_matching_objs(self.mission)
                    self.stack.append(('GoNextTo', tuple(fwd_pos)))
                    self.stack.append(('Pickup', key_desc))
                    self.stack.append(('GoToObj', key_desc))
                    return None

            self.stack.pop()
            return actions.toggle

        # Pick up an object
        if subgoal == 'Pickup':
            self.stack.pop()
            return actions.pickup

        # Drop an object
        if subgoal == 'Drop':
            self.stack.pop()
            return actions.drop

        # Go to an object
        if subgoal == 'GoToObj':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)

            # Go to the location of this object
            if obj_pos:
                # If we are right in front of the object,
                # go back to the previous subgoal
                if np.array_equal(obj_pos, fwd_pos):
                    self.stack.pop()
                    return None

                path, _ = self.shortest_path(lambda pos, cell: pos == obj_pos)

                if path:
                    # New subgoal: go next to the object
                    self.stack.append(('GoNextTo', obj_pos))
                    return None

            # Explore the world
            self.stack.append(('Explore', None))
            return None

        # Go to a given location
        if subgoal == 'GoNextTo':
            # If we are facing the target cell, subgoal completed
            if np.array_equal(datum, fwd_pos):
                self.stack.pop()
                return None

            assert pos is not datum

            path, _ = self.shortest_path(lambda pos, cell: pos == datum)
            next_cell = path[0]

            # Turn towards the direction we need to go
            if np.array_equal(next_cell, fwd_pos):
                return actions.forward
            if np.array_equal(next_cell - pos, right_vec):
                return actions.right
            return actions.left

        # Go to next to a position adjacent to an object
        if subgoal == 'GoToAdjPos':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)

            if not obj_pos:
                self.stack.append(('Explore', None))
                return None

            # Find the closest position adjacent to the object
            #adj_pos = self.closest_adj_pos(obj_pos)
            path, adj_pos = self.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
            )

            if not adj_pos:
                self.stack.append(('Explore', None))
                return None

            # FIXME: h4xx
            # Randomly navigate away from this position
            if np.array_equal(pos, adj_pos):
                if np.random.randint(0, 2) == 0:
                    return actions.left
                else:
                    return actions.forward

            self.stack.pop()
            self.stack.append(('GoNextTo', adj_pos))
            return None

        # Explore the world, uncover new unseen cells
        if subgoal == 'Explore':
            # TODO: handle unopened doors

            # Find the closest unseen position
            _, unseen_pos = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos]
            )

            if unseen_pos:
                self.stack.pop()
                self.stack.append(('GoNextTo', unseen_pos))
                return None

            # Find the closest unlocked unopened door
            def unopened_unlocked_door(pos, cell):
                if not cell:
                    return False
                if cell.type != 'door':
                    return False
                return not cell.is_open

            # Find the closest unopened door
            def unopened_door(pos, cell):
                if not cell:
                    return False
                if cell.type != 'door' and cell.type != 'locked_door':
                    return False
                return not cell.is_open

            # Try to find an unlocked door first
            # We do this because otherwise, opening an unlocked door as
            # a subgoal may try to open the same subdoor for exploration,
            # resulting in an infinite loop
            _, door_pos = self.shortest_path(unopened_unlocked_door)
            if not door_pos:
                _, door_pos = self.shortest_path(unopened_door)

            # Open the door
            if door_pos:
                door = self.mission.grid.get(*door_pos)
                self.stack.pop()
                self.stack.append(('Open', None))
                self.stack.append(('GoNextTo', door_pos))
                return None

            print(self.stack)
            assert False, "nothing left to explore"

        assert False, 'invalid subgoal "%s"' % subgoal

    def process_obs(self):
        """
        Parse the contents of an observation/image and update our state
        """

        grid, vis_mask = self.mission.gen_obs_grid()

        pos = self.mission.agent_pos
        f_vec = self.mission.dir_vec
        r_vec = self.mission.right_vec

        # Compute the absolute coordinates of the top-left corner
        # of the agent's view area
        top_left = pos + f_vec * (AGENT_VIEW_SIZE-1) - r_vec * (AGENT_VIEW_SIZE // 2)

        # Mark everything in front of us as visible
        for vis_j in range(0, AGENT_VIEW_SIZE):
            for vis_i in range(0, AGENT_VIEW_SIZE):

                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.vis_mask.shape[0]:
                    continue
                if abs_j < 0 or abs_j >= self.vis_mask.shape[1]:
                    continue

                self.vis_mask[abs_i, abs_j] = True

    def find_obj_pos(self, obj_desc):
        """
        Find the position of a visible object matching a given description
        """

        assert len(obj_desc.obj_set) > 0

        for i in range(len(obj_desc.obj_set)):
            obj = obj_desc.obj_set[i]
            pos = obj_desc.obj_poss[i]

            if self.vis_mask[pos]:
                return pos

        return None

    def shortest_path(
        self,
        accept_fn
    ):
        """
        Perform a Breadth-First Search (BFS) starting from the agent position,
        in order to find the closest cell or shortest path to a cell
        satisfying a given condition.
        """

        grid = self.mission.grid

        # Set of visited positions
        visited = set()

        # Queue of states to visit (BFS)
        # Includes (i,j) positions along with path to given position
        queue = []

        # Start visiting from the agent's position
        queue.append((*self.mission.agent_pos, []))

        # Until we are done
        while len(queue) > 0:
            i, j, path = queue[0]
            queue = queue[1:]

            if i < 0 or i >= grid.width or j < 0 or j >= grid.height:
                continue

            if (i, j) in visited:
                continue

            # Mark this position as visited
            visited.add((i, j))

            cell = grid.get(i, j)

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), cell):
                return path, (i,j)

            # If this cell was not visually observed, don't visit neighbors
            if not self.vis_mask[i, j]:
                continue

            # Can't go through closed doors, don't visit neighbors
            if cell:
                if cell.type != 'door' and cell.type != 'locked_door':
                    continue
                if not cell.is_open:
                    continue

            # Visit each neighbor cell
            for k, l in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (i+k, j+l)
                queue.append((*next_pos, path + [next_pos]))

        # Path not found
        return None, None
