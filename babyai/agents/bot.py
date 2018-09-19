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
        # BotAdvisor agent doesn't work as a ForgetBot
        assert not (forget and isinstance(self, BotAdvisor)), "Forgetful BotAdvisor not implemented"
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
            self.stack.append(('Drop', None))
            self.stack.append(('Pickup', instr.desc))
            self.stack.append(('GoToObj', instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(('Forget', None))
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
                        self.stack.append(('Open', None))
                        self.stack.append(('GoNextTo', tuple(fwd_pos)))

                        # Go to the key and pick it up
                        self.stack.append(('Pickup', key_desc))
                        self.stack.append(('GoToObj', key_desc))

                        # Drop the object being carried
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))
                    else:
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
            print('datum0{} obj_pos0 {}'.format(datum, obj_pos))

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
                    path, _ = self.shortest_path(
                        lambda pos, cell: pos == obj_pos,
                        ignore_blockers=True
                    )

                if path:
                    # New subgoal: go next to the object
                    self.stack.append(('GoNextTo', obj_pos))
                    return None

            # Explore the world
            self.stack.append(('Explore', None))
            return None

        # Go to a given location
        if subgoal == 'GoNextTo':
            assert tuple(pos) != datum

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
                path, _ = self.shortest_path(
                    lambda pos, cell: np.array_equal(pos, datum),
                    ignore_blockers=True
                )

            # No path found, explore the world
            if not path:
                # Explore the world
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
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_block))
                        self.stack.append(('Pickup', None))
                        self.stack.append(('GoNextTo', fwd_pos))

                        # Drop the object being carried
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))

                        return None
                    else:
                        drop_pos = self.find_drop_pos()
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos))
                        return actions.pickup

                return actions.forward

            # Turn towards the direction we need to go
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
            path, adj_pos = self.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
            )

            if not adj_pos:
                path, adj_pos = self.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                    ignore_blockers=True
                )

            if not adj_pos:
                self.stack.append(('Explore', None))
                return None

            # FIXME: h4xx
            # If we are on the target position,
            # Randomly navigate away from this position
            if np.array_equal(pos, adj_pos):
                return actions.left
                if np.random.randint(0, 2) == 0:
                    return actions.left
                else:
                    return actions.forward

            self.stack.pop()
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

    def shortest_path(
        self,
        accept_fn,
        ignore_blockers=False
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
                    # This is a blocking object, don't visit neighbors
                    if not ignore_blockers:
                        continue

            # Visit each neighbor cell
            for k, l in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (i+k, j+l)
                queue.append((*next_pos, path + [next_pos]))

        # Path not found
        return None, None

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


class BotAdvisor(Bot):
    def get_action(self):
        """
        Produce the optimal action at the current state
        """

        # Process the current observation
        self.process_obs()
        subgoal, datum = self.stack[-1]
        action = self.get_action_from_subgoal(subgoal, datum)

        return action

    def get_action_from_subgoal(self, subgoal, datum=None, alternative=(None, None)):
        # alternative is a tuple (subgoal, datum) that should be used if the actual subgoal
        # is satisfied already and requires no action taking
        # It is useful for GoToObj and GoNextTo subgoals
        while subgoal is None:
            self.stack.pop()
            subgoal, datum = self.stack[-1]

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

        if subgoal == 'Forget':
            # BotAdvisor cannot be forgetful
            self.stack.pop()
            return self.get_action_from_subgoal(*self.stack[-1]) if self.stack else None

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

                        # Find a location to drop what we're already carrying
                        drop_pos_cur = self.find_drop_pos()
                        return self.get_action_from_subgoal('GoNextTo', drop_pos_cur, ('Drop', None))
                    else:
                        # Go To the key
                        return self.get_action_from_subgoal('GoToObj', key_desc, ('Pickup', key_desc))

            # If the door is already open, close it so we can open it again
            if fwd_cell.type == 'door' and fwd_cell.is_open:
                return actions.toggle

            return actions.toggle

        # Pick up an object
        if subgoal == 'Pickup':
            return actions.pickup

        # Drop an object
        if subgoal == 'Drop':
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
                        return self.get_action_from_subgoal(*alternative)

                path, _ = self.shortest_path(
                    lambda pos, cell: pos == obj_pos
                )

                if not path:
                    path, _ = self.shortest_path(
                        lambda pos, cell: pos == obj_pos,
                        ignore_blockers=True
                    )

                if path:
                    # New subgoal: go next to the object
                    return self.get_action_from_subgoal('GoNextTo', obj_pos, alternative)

            # Explore the world
            return self.get_action_from_subgoal('Explore', None)

        # Go to a given location
        if subgoal == 'GoNextTo':
            assert pos is not datum

            # If we are facing the target cell, subgoal completed
            if np.array_equal(datum, fwd_pos):
                return self.get_action_from_subgoal(*alternative)

            # Try to find a path
            path, _ = self.shortest_path(
                lambda pos, cell: np.array_equal(pos, datum)
            )

            # If we failed to find a path, try again while ignoring blockers
            if not path:
                path, _ = self.shortest_path(
                    lambda pos, cell: np.array_equal(pos, datum),
                    ignore_blockers=True
                )

            # No path found, explore the world
            if not path:
                return self.get_action_from_subgoal('Explore', None)

            next_cell = path[0]

            # If the destination is ahead of us
            if np.array_equal(next_cell, fwd_pos):
                fwd_cell = self.mission.grid.get(*fwd_pos)

                # If there is a blocking object in front of us
                if fwd_cell and not fwd_cell.type.endswith('door'):
                    if carrying:
                        drop_pos_cur = self.find_drop_pos()
                        # Drop the object being carried
                        return self.get_action_from_subgoal('GoNextTo', drop_pos_cur, ('Drop', None))

                    else:
                        return actions.pickup

                return actions.forward

            # Turn towards the direction we need to go
            if np.array_equal(next_cell - pos, right_vec):
                return actions.right
            return actions.left

        # Go to next to a position adjacent to an object
        if subgoal == 'GoToAdjPos':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)

            if not obj_pos:
                return self.get_action_from_subgoal('Explore', None)

            # Find the closest position adjacent to the object
            path, adj_pos = self.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
            )

            if not adj_pos:
                path, adj_pos = self.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                    ignore_blockers=True
                )

            if not adj_pos:
                return self.get_action_from_subgoal('Explore', None)

            # FIXME: h4xx ??
            # If we are on the target position,
            # Randomly navigate away from this position
            if np.array_equal(pos, adj_pos):
                return actions.left
                if np.random.randint(0, 2) == 0:
                    return actions.left
                else:
                    return actions.forward
            return self.get_action_from_subgoal('GoNextTo', adj_pos)

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
                return self.get_action_from_subgoal('GoNextTo', unseen_pos)

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
                return self.get_action_from_subgoal('GoNextTo', door_pos, ('Open', None))

            # Find the closest unseen position, ignoring blocking objects
            path, unseen_pos = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos],
                ignore_blockers=True
            )

            if unseen_pos:
                return self.get_action_from_subgoal('GoNextTo', unseen_pos)

            # print(self.stack)
            assert False, "nothing left to explore"

        assert False, 'invalid subgoal "%s"' % subgoal

    def simulate_step(self, action):
        pos = self.mission.agent_pos
        agent_dir = self.mission.agent_dir
        dir_vec = DIR_TO_VEC[agent_dir]
        right_vec = self.mission.right_vec
        fwd_pos = pos + dir_vec
        # Get the contents of the cell in front of the agent
        fwd_cell = self.mission.grid.get(*fwd_pos)

        actions = self.mission.actions
        carrying = self.mission.carrying

        # Rotate left
        if action == actions.left:
            agent_dir -= 1
            if agent_dir < 0:
                agent_dir += 4

        # Rotate right
        elif action == actions.right:
            agent_dir = (agent_dir + 1) % 4

        # Move forward
        elif action == actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                pos = fwd_pos

        # Pick up an object
        elif action == actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if carrying is None:
                    carrying = fwd_cell

        # Drop an object
        elif action == actions.drop:
            if not fwd_cell and carrying:
                carrying = None

        # Toggle/activate an object
        elif action == actions.toggle:
            pass

        # Done action (not used by default)
        elif action == actions.done:
            pass

        else:
            assert False, "unknown action"

        dir_vec = DIR_TO_VEC[agent_dir]
        right_vec = np.array((-dir_vec[1], dir_vec[0]))
        fwd_pos = pos + dir_vec

        return pos, dir_vec, right_vec, fwd_pos, carrying

    def take_action(self, action):
        """
        Update agent's internal state. Should always be called after get_action() and before env.step()
        """

        self.step_count += 1

        finished_updating = False
        while not finished_updating:
            finished_updating = self.take_action_iterate(action)

    def take_action_iterate(self, action):
        # TODO: make this work with done action ?
        if len(self.stack) == 0:
            # Is this right?
            return True

        # Get the topmost instruction on the stack
        subgoal, datum = self.stack[-1]

        #print(subgoal, datum)
        #print('pos:', pos)

        while subgoal == 'Forget':
            self.stack.pop()
            assert len(self.stack) != 0
            subgoal, datum = self.stack[-1]

        if self.itr_count >= self.timeout:
            raise TimeoutError('bot timed out')

        self.itr_count += 1
        print(self.itr_count)

        pos = self.mission.agent_pos
        dir_vec = self.mission.dir_vec
        print(dir_vec)
        right_vec = self.mission.right_vec
        fwd_pos = pos + dir_vec

        actions = self.mission.actions
        carrying = self.mission.carrying

        new_pos, new_dir_vec, new_right_vec, new_fwd_pos, new_carrying = self.simulate_step(action)

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
                        self.stack.append(('Open', None))
                        self.stack.append(('GoNextTo', tuple(fwd_pos)))

                        # Go to the key and pick it up
                        self.stack.append(('Pickup', key_desc))
                        self.stack.append(('GoToObj', key_desc))

                        # Drop the object being carried
                        self.stack.append(('Drop', None))
                        self.stack.append(('GoNextTo', drop_pos_cur))
                    else:
                        self.stack.append(('GoNextTo', tuple(fwd_pos)))
                        self.stack.append(('Pickup', key_desc))
                        self.stack.append(('GoToObj', key_desc))

                    return False

            # If the door is already open, close it so we can open it again
            if fwd_cell.type == 'door' and fwd_cell.is_open:
                if action in (actions.left, actions.right, actions.forward):
                    # Go back to the door to open it
                    self.stack.append(('GoNextTo', tuple(fwd_pos)))
                if action == actions.drop and carrying:
                    self.stack.append(('Pickup', None))
                return True

            # If nothing is returned so far, it means that the door is openable with a simple toggle

            if action == actions.toggle:
                self.stack.pop()
            if action in (actions.left, actions.right):
                # Go back to the door to open it
                self.stack.append(('GoNextTo', tuple(fwd_pos)))
            # You can't do anything else
            return True

        # Pick up an object
        if subgoal == 'Pickup':
            if action == actions.pickup:
                self.stack.pop()
            # these are the only actions that can change your state (you are carrying nothing, and there is something pickup-able in front of you !)
            elif action in (actions.left, actions.right):
                # Go back to where you were to pickup what was in front of you
                self.stack.append(('GoNextTo', tuple(fwd_pos)))
            elif action == actions.toggle and self.mission.grid.get(*fwd_pos).type == 'box':
                # basically the foolish agent opens the box that he had to pickup, the box doesn't exist anymore (MAGIC !)
                # just loop until timeout and abandon. Bot can't help the agent after such a mistake
                # TODO: give the agent another chance by looking for a similar box still unopened
                return False
            return True

        # Drop an object
        if subgoal == 'Drop':
            if action == actions.drop:
                self.stack.pop()
            # these are the only actions that can change your state (you are already carrying something, and there is nothing in front of you !)
            elif action in (actions.left, actions.right, actions.forward):
                # Go back to where you were to drop what you got
                self.stack.append(('GoNextTo', tuple(fwd_pos)))
            return True

        def drop_or_pickup_or_open_something_while_exploring():
            print(carrying, new_carrying)
            # A bit too conservative here:
            # If you pick up an object when you shouldn't have, you should drop it back in the same position
            # If you drop an object when you shouldn't have, you should pick that one up, and not one similar to it
            if action == actions.drop and carrying != new_carrying:
                # get that thing back
                new_fwd_cell = carrying
                assert new_fwd_cell.type in ('key', 'box', 'ball')
                # Hopefully the bot would pickup THIS object and not something similar to it
                self.stack.append(('Pickup', None))
            elif action == actions.pickup and carrying != new_carrying:
                # drop that thing where you found it
                fwd_cell = self.mission.grid.get(*fwd_pos)
                assert fwd_cell.type in ('key', 'box', 'ball')
                self.stack.append(('Drop', None))

            elif action == actions.toggle:
                fwd_cell = self.mission.grid.get(*fwd_pos)
                if fwd_cell and fwd_cell.type == 'door' and fwd_cell.is_open:
                    # i.e. the agent decided to close the door
                    # need to open it
                    self.stack.append(('Open', None))
            return True


        # Go to an object
        if subgoal == 'GoToObj':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)
            print('datum{} obj_pos {}'.format(datum, obj_pos))

            # Go to the location of this object
            if obj_pos:
                # If we are right in front of the object,
                # go back to the previous subgoal
                if np.array_equal(obj_pos, fwd_pos):
                    self.stack.pop()
                    return False
                if action in (actions.forward, actions.left, actions.right) and np.array_equal(obj_pos, new_fwd_pos):
                    # stack will be popped at next step
                    return True
                elif action in (actions.drop, actions.pickup, actions.toggle):
                    return drop_or_pickup_or_open_something_while_exploring()
                else:
                    path, _ = self.shortest_path(
                        lambda pos, cell: pos == obj_pos
                    )

                    if not path:
                        path, _ = self.shortest_path(
                            lambda pos, cell: pos == obj_pos,
                            ignore_blockers=True
                        )

                    if path:
                        # New subgoal: go next to the object
                        self.stack.append(('GoNextTo', obj_pos))
                        return True
            # Explore the world
            self.stack.append(('Explore', None))
            return False

        # Go to a given location
        if subgoal == 'GoNextTo':
            print(subgoal)
            if tuple(pos) == datum:
                return True

            # If we are facing the target cell, subgoal completed
            if np.array_equal(datum, fwd_pos):

                self.stack.pop()
                return False
            if action in (actions.forward, actions.left, actions.right) and np.array_equal(datum, new_fwd_pos):
                # stack will be popped at next step anyway
                return True
            elif action in (actions.drop, actions.pickup, actions.toggle):
                return drop_or_pickup_or_open_something_while_exploring()
            else:
                # Try to find a path
                path, _ = self.shortest_path(
                    lambda pos, cell: np.array_equal(pos, datum)
                )

                # If we failed to find a path, try again while ignoring blockers
                if not path:
                    path, _ = self.shortest_path(
                        lambda pos, cell: np.array_equal(pos, datum),
                        ignore_blockers=True
                    )

                # No path found, explore the world
                if not path:
                    # Explore the world
                    self.stack.append(('Explore', None))
                    return True

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
                            self.stack.append(('Drop', None))
                            self.stack.append(('GoNextTo', drop_pos_block))
                            self.stack.append(('Pickup', None))
                            self.stack.append(('GoNextTo', fwd_pos))

                            # Drop the object being carried
                            self.stack.append(('Drop', None))
                            self.stack.append(('GoNextTo', drop_pos_cur))

                        else:
                            drop_pos = self.find_drop_pos()
                            self.stack.append(('Drop', None))
                            self.stack.append(('GoNextTo', drop_pos))
                    return True
                return True

        # Go to next to a position adjacent to an object
        if subgoal == 'GoToAdjPos':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)

            if action in (actions.drop, actions.pickup, actions.toggle):
                return drop_or_pickup_or_open_something_while_exploring()
            else:
                if not obj_pos:
                    self.stack.append(('Explore', None))
                    return True

                # Find the closest position adjacent to the object
                path, adj_pos = self.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
                )

                if not adj_pos:
                    path, adj_pos = self.shortest_path(
                        lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                        ignore_blockers=True
                    )

                if not adj_pos:
                    self.stack.append(('Explore', None))
                    return False

                # FIXME: h4xx ??
                # If we are on the target position,
                # Randomly navigate away from this position
                if action in (actions.forward, actions.left, actions.right):
                    self.stack.pop()
                    self.stack.append(('GoNextTo', adj_pos))
                    return False
                return True

        # Explore the world, uncover new unseen cells
        if subgoal == 'Explore':
            # Find the closest unseen position
            _, unseen_pos = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos]
            )
            if action in (actions.drop, actions.pickup, actions.toggle):
                return drop_or_pickup_or_open_something_while_exploring()

            if not unseen_pos:
                _, unseen_pos = self.shortest_path(
                    lambda pos, cell: not self.vis_mask[pos],
                    ignore_blockers=True
                )

            if unseen_pos:
                self.stack.pop()
                self.stack.append(('GoNextTo', unseen_pos))
                return False

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
                return False

            # Find the closest unseen position, ignoring blocking objects
            path, unseen_pos = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos],
                ignore_blockers=True
            )

            if unseen_pos:
                self.stack.pop()
                self.stack.append(('GoNextTo', unseen_pos))
                return False

            #print(self.stack)
            assert False, "nothing left to explore"

        assert False, 'invalid subgoal "%s"' % subgoal

    # The following function is for testing purposes only
    def step(self):
        action = self.get_action()
        self.take_action(action)