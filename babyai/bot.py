import numpy
import time
import math

from gym_minigrid.minigrid import *
from babyai.levels.verifier import *


class Bot:
    """
    Heuristic-based level solver
    """

    def __init__(self, mission, forget=False, timeout=10000):
        # Mission to be solved
        self.mission = mission

        grid_size = mission.grid_size

        # Grid containing what has been mapped out
        self.grid = Grid(grid_size, grid_size)

        # Visibility mask. True for explored/seen, false for unexplored.
        self.vis_mask = np.zeros(shape=(grid_size, grid_size), dtype=np.bool)

        # Number of environment steps
        self.step_count = 0

        # Forget the visibility mask after each subgoal is completed
        self.forget = forget

        # Number of compute iterations performed
        self.itr_count = 0

        # Maximum number of compute iterations to perform
        self.timeout = timeout

        # Stack of tasks/subtasks to complete (tuples)
        self.stack = []

        # Process/parse the instructions
        self.process_instr(mission.instrs)

        #for subgoal, datum in self.stack:
        #    print(subgoal)
        #    if datum:
        #        print(datum.surface(self.mission))
    def update_objs_poss(self, instr=None):
        if instr is None:
            instr = self.mission.instrs
        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr) or isinstance(instr, AfterInstr):
            self.update_objs_poss(instr.instr_a)
            self.update_objs_poss(instr.instr_b)
        else:
            instr.update_objs_poss()

    def process_instr(self, instr):
        """
        Translate instructions into an internal form the agent can execute
        """

        if isinstance(instr, GoToInstr):
            self.stack.append(('Forget', None))
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(('Forget', None))
            self.stack.append(('Open', instr.desc))
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, PickupInstr):
            # We pick up and immediately drop so
            # that we may carry other objects
            self.stack.append(('Forget', None))
            self.stack.append(('UpdateObjsPoss', None))
            self.stack.append(('Drop', None))
            self.stack.append(('Pickup', instr.desc))
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(('Forget', None))
            self.stack.append(('UpdateObjsPoss', None))
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

        self.step_count += 1

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

        if self.itr_count >= self.timeout:
            raise TimeoutError('bot timed out')

        self.itr_count += 1

        pos = self.mission.agent_pos
        dir_vec = self.mission.dir_vec
        right_vec = self.mission.right_vec
        fwd_pos = pos + dir_vec

        actions = self.mission.actions
        carrying = self.mission.carrying

        if len(self.stack) == 0:
            return actions.done

        # Get the topmost instruction on the stack
        subgoal, datum = self.stack[-1]

        #print(subgoal, datum)
        #print('pos:', pos)

        # Forget after a subgoal is completed
        if subgoal == 'Forget':
            if self.forget:
                self.vis_mask = np.zeros(shape=self.vis_mask.shape, dtype=np.bool)
                self.process_obs()
            self.stack.pop()
            return None

        # Open a door
        if subgoal == 'Open':
            fwd_cell = self.mission.grid.get(*fwd_pos)
            assert fwd_cell
            assert fwd_cell.type == "door" or fwd_cell.type == "locked_door"

            # If the door is locked, go find the key and then return
            if fwd_cell.type == 'locked_door':
                if not carrying or carrying.type != 'key' or carrying.color != fwd_cell.color:
                    # Find the key
                    key_desc = ObjDesc('key', fwd_cell.color)
                    key_desc.find_matching_objs(self.mission)

                    # If we're already carrying something
                    if carrying:
                        self.stack.pop()

                        # Find a location to drop what we're already carrying
                        drop_pos_cur = self.find_drop_pos()

                        # Take back the object being carried
                        self.stack.append(('Pickup', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))

                        # Go back to the door and open it
                        self.stack.append(('UpdateObjsPoss', None))
                        self.stack.append(('Open', None))
                        self.stack.append(('GoNextTo', tuple(fwd_pos)))

                        # Go to the key and pick it up
                        self.stack.append(('Pickup', key_desc))
                        self.stack.append(('GoToObj', key_desc))

                        # Drop the object being carried
                        self.stack.append(('UpdateObjsPoss', None))
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))
                    else:
                        self.stack.pop()
                        self.stack.append(('UpdateObjsPoss', None))
                        self.stack.append(('Open', None))
                        self.stack.append(('GoNextTo', tuple(fwd_pos)))
                        self.stack.append(('Pickup', key_desc))
                        self.stack.append(('GoToObj', key_desc))

                    # Don't perform any action for this iteration
                    return None

            # If the door is already open, close it so we can open it again
            if fwd_cell.type == 'door' and fwd_cell.is_open:
                return actions.toggle

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

                path, _ = self.shortest_path(
                    lambda pos, cell: pos == obj_pos
                )

                if not path:
                    next_search_dist = 999
                    while next_search_dist is not None:
                        suggested_path, _, next_search_dist = self.shortest_path(
                            lambda pos, cell: pos == obj_pos,
                            ignore_blockers=True,
                            blocker_fn=lambda pos: self.blocker_fn(pos, obj_pos, next_search_dist),
                            distance_fn=lambda pos: self.distance(pos, obj_pos)
                        )
                        if suggested_path:
                            path = [elem for elem in suggested_path]

                if path:
                    # New subgoal: go next to the object
                    self.stack.append(('GoNextTo', obj_pos))
                    return None

            # Explore the world
            self.stack.append(('Explore', None))
            return None

        # Go to a given location
        if subgoal == 'GoNextTo':
            assert pos is not datum

            # If we are facing the target cell, subgoal completed
            if np.array_equal(datum, fwd_pos):
                self.stack.pop()
                return None

            # Try to find a path
            path, _ = self.shortest_path(
                lambda pos, cell: np.array_equal(pos, datum)
            )

            # If we failed to find a path, try again while ignoring blockers
            if not path:
                next_search_dist = 999
                while next_search_dist is not None:
                    suggested_path, _, next_search_dist = self.shortest_path(
                        lambda pos, cell: np.array_equal(pos, datum),
                        ignore_blockers=True,
                        blocker_fn=lambda pos: self.blocker_fn(pos, datum, next_search_dist),
                        distance_fn=lambda pos: self.distance(pos, datum)
                    )
                    if suggested_path:
                        path = [elem for elem in suggested_path]

            # No path found, explore the world
            if not path:
                # Explore the world
                #print(pos, datum)
                #print('Exploring0')
                self.stack.append(('Explore', None))
                return None

            next_cell = path[0]

            # If the destination is ahead of us
            if np.array_equal(next_cell, fwd_pos):
                fwd_cell = self.mission.grid.get(*fwd_pos)

                # If there is a blocking object in front of us
                if fwd_cell and not fwd_cell.type.endswith('door'):
                    if carrying:
                        drop_pos_cur = self.find_drop_pos()
                        drop_pos_block = self.find_drop_pos(drop_pos_cur)

                        # Take back the object being carried
                        self.stack.append(('Pickup', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))

                        # Pick up the blocking object and drop it
                        self.stack.append(('UpdateObjsPoss', None))
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_block))
                        self.stack.append(('Pickup', None))
                        self.stack.append(('GoNextTo', fwd_pos))

                        # Drop the object being carried
                        self.stack.append(('UpdateObjsPoss', None))
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))

                    else:
                        drop_pos_block = self.find_drop_pos()
                        self.stack.append(('UpdateObjsPoss', None))
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_block))
                        self.stack.append(('Pickup', None))
                        self.stack.append(('GoNextTo', fwd_pos))

                    return None

                return actions.forward

            # Turn towards the direction we need to go
            if np.array_equal(next_cell - pos, right_vec):
                return actions.right
            return actions.left

        if subgoal == 'UpdateObjsPoss':
            self.stack.pop()
            self.update_objs_poss()
            return None

        # Go to next to a position adjacent to an object
        if subgoal == 'GoToAdjPos':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)

            if not obj_pos:
                self.stack.append(('Explore', None))
                return None

            # Find the closest position adjacent to the object
            path, adj_pos = self.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
            )

            if not adj_pos:
                path, adj_pos = self.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                    ignore_blockers=True)

            if not adj_pos:
                self.stack.append(('Explore', None))
                return None

            # FIXME: h4xx
            # If we are on the target position,
            # Randomly navigate away from this position
            if np.array_equal(pos, adj_pos):
                #return actions.left
                if np.random.randint(0, 2) == 0:
                    return actions.left
                else:
                    return actions.forward

            if np.array_equal(adj_pos, fwd_pos):
                self.stack.pop()
                return None

            self.stack.append(('GoNextTo', adj_pos))
            return None

        # Explore the world, uncover new unseen cells
        if subgoal == 'Explore':
            # Find the closest unseen position
            _, unseen_pos = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos]
            )

            if not unseen_pos:
                _, unseen_pos = self.shortest_path(
                    lambda pos, cell: not self.vis_mask[pos],
                    ignore_blockers=True
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
            # We do this because otherwise, opening a locked door as
            # a subgoal may try to open the same door for exploration,
            # resulting in an infinite loop
            _, door_pos = self.shortest_path(unopened_unlocked_door)
            if not door_pos:
                _, door_pos = self.shortest_path(unopened_unlocked_door, ignore_blockers=True)
            if not door_pos:
                _, door_pos = self.shortest_path(unopened_door)
            if not door_pos:
                _, door_pos = self.shortest_path(unopened_door, ignore_blockers=True)

            # Open the door
            if door_pos:
                door = self.mission.grid.get(*door_pos)
                self.stack.pop()
                self.stack.append(('Open', None))
                self.stack.append(('GoNextTo', door_pos))
                return None

            # Find the closest unseen position, ignoring blocking objects
            path, unseen_pos = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos],
                ignore_blockers=True
            )

            if unseen_pos:
                self.stack.pop()
                self.stack.append(('GoNextTo', unseen_pos))
                return None

            #print(self.stack)
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

    def distance(self, pos, target):
        return np.abs(target[0] - pos[0]) + np.abs(target[1] - pos[1])

    def blocker_fn(self, pos, target, n_steps):
        # Useful to define a function that blockers need to satisfy when looking for shortest path
        # target and pos are tuples or arrays of 2 elements
        return  self.distance(pos, target) <= n_steps

    def shortest_path(
        self,
        accept_fn,
        ignore_blockers=False,
        blocker_fn=None,
        distance_fn=None
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
        queue.append((*self.mission.agent_pos, *self.mission.dir_vec, []))

        distance_of_first_blocking_obj_to_target = None

        # Until we are done
        while len(queue) > 0:
            i, j, di, dj, path = queue[0]

            queue = queue[1:]

            if i < 0 or i >= grid.width or j < 0 or j >= grid.height:
                continue

            if (i, j) in visited:
                continue
            #print(i, j)
            # Mark this position as visited
            visited.add((i, j))

            cell = grid.get(i, j)

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), cell):
                if distance_fn is None:
                    return path, (i, j)
                else:
                    return path, (i, j), distance_of_first_blocking_obj_to_target - 1

            # If this cell was not visually observed, don't visit neighbors
            if not self.vis_mask[i, j]:
                continue

            # If there is something in this cell
            if cell:
                # If this is a wall, don't visit neighbors
                if cell.type == 'wall':
                    continue
                # If this is a door
                elif cell.type == 'door' or cell.type == 'locked_door':
                    # If the door is closed, don't visit neighbors
                    if not cell.is_open:
                        continue
                else:
                    if blocker_fn is None:
                        def blocker_fn(pos):
                            return True
                    # This is a blocking object, don't visit neighbors
                    if not ignore_blockers or not blocker_fn((i, j)):
                        continue
                    else:
                        if distance_fn is not None and distance_of_first_blocking_obj_to_target is None:
                            distance_of_first_blocking_obj_to_target = distance_fn((i, j))
                            #print('first blocker path contains a blocker of distance {} to target'.format(distance_of_first_blocking_obj_to_target))

            # Visit each neighbor cell
            # TODO: I want this to be "for positions that are one action away and that would make the state change (e.g. if position changes or if carrying changes) instead of one cell away. If there are no one action away positions that change the state, check "2 action away things" then "3 action away things", that is the maximum we should tolerate (e.g. left left forward/pickup/drop or right right forward/pickup/drop)
            for k, l in [(di, dj), (dj, di), (- dj, - di), (- di, - dj)]:
                next_pos = (i+k, j+l)
                next_dir_vec = (k, l)
                queue.append((*next_pos, *next_dir_vec, path + [next_pos]))

        # Path not found
        if distance_fn is None:
            return None, None
        else:
            return None, None, None

    def find_drop_pos(self, except_pos=None):
        """
        Find a position where an object can be dropped,
        ideally without blocking anything
        """

        grid = self.mission.grid

        def match_noadj(pos, cell):
            i, j = pos

            if np.array_equal(pos, self.mission.agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            # If the cell or a neighbor was unseen or is occupied, reject
            for k, l in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                nb_pos = (i+k, j+l)
                if not self.vis_mask[nb_pos] or grid.get(*nb_pos):
                    return False

            return True

        def match_empty(pos, cell):
            i, j = pos

            if np.array_equal(pos, self.mission.agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[pos] or grid.get(*pos):
                return False

            return True

        _, drop_pos = self.shortest_path(match_noadj)

        if not drop_pos:
            _, drop_pos = self.shortest_path(match_empty)

        if not drop_pos:
            _, drop_pos = self.shortest_path(match_noadj, ignore_blockers=True)

        if not drop_pos:
            _, drop_pos = self.shortest_path(match_empty, ignore_blockers=True)

        return drop_pos


class BotRewardWrapper(gym.Wrapper):
    """
    Wrapper that rewards the agent for taking the same action as the
    bot would take
    """

    def step(self, action):
        try:
            expert_action = self.expert.step()
        except:
            expert_action = None

        obs, reward, done, info = self.env.step(action)

        reward *= 1000

        if action == expert_action:
            #reward += 1 / self.unwrapped.max_steps
            reward += 1

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.expert = Bot(self.env)
        return obs


class BotActionInfoWrapper(gym.Wrapper):
    def step(self, action):
        bot_action = self.expert.step()

        obs, reward, done, info = self.env.step(action)

        info['bot_action'] = bot_action

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.expert = Bot(self.env)
        return obs