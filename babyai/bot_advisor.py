from babyai.levels.verifier import *
from babyai.bot import Bot


class DisappearedBoxError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class BotAdvisor(Bot):
    def get_action(self):
        """
        Produce the optimal action at the current state
        """
       # print(self.stack)
        # Process the current observation
        self.process_obs()
        if len(self.stack) == 0:
            return self.mission.actions.done
        subgoal, datum = self.stack[-1]
        action = self.get_action_from_subgoal(subgoal, datum)
       # print(action)
        return action

    def get_action_from_subgoal(self, subgoal, datum=None, alternative=(None, None)):
        # alternative is a tuple (subgoal, datum) that should be used if the actual subgoal
        # is satisfied already and requires no action taking
        # It is useful for GoToObj and GoNextTo subgoals
        #print('step {} get subgoal {} {}'.format(self.step_count, subgoal, datum))
        #print("{}: {}".format(self.step_count, subgoal))
        if len(self.stack) == 0:
            return self.mission.actions.done

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
        fwd_cell = self.mission.grid.get(*fwd_pos)

        actions = self.mission.actions
        carrying = self.mission.carrying

        if subgoal == 'Forget':
            # BotAdvisor cannot be forgetful
            self.stack.pop()
            return self.get_action_from_subgoal(*self.stack[-1]) if self.stack else actions.done

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
                       # print('yooo')
                        return self.get_action_from_subgoal('GoNextTo', drop_pos_cur, ('Drop', None))
                    else:
                        # Go To the key
                        return self.get_action_from_subgoal('GoToObj', key_desc, ('Pickup', key_desc))

            # If the door is already open, close it so we can open it again
            if fwd_cell.type == 'door' and fwd_cell.is_open:
                return actions.toggle
           # print('open1')
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

                path, _, _ = self.shortest_path(
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
                    return self.get_action_from_subgoal('GoNextTo', obj_pos, alternative)

            # Explore the world
            #print('exp')
            return self.get_action_from_subgoal('Explore', None)

        # Go to a given location
        if subgoal == 'GoNextTo':
            no_reexplore = False
            if isinstance(datum, dict):
                if 'no_reexplore' in datum.keys():
                    no_reexplore = datum['no_reexplore']
                datum = datum['dest']
            #print(datum)
            if tuple(pos) == tuple(datum):
                # move away from it
                if fwd_cell is None:
                    return actions.forward
                if self.mission.grid.get(*(pos + right_vec)) is None:
                    return actions.right
                if self.mission.grid.get(*(pos - right_vec)) is None:
                    return actions.left
                return actions.left
                # rand = np.random.randint(0, 3)
                # if rand == 0:
                #     return actions.left
                # elif rand == 1:
                #     return actions.forward
                # else:
                #     return actions.right

            # If we are facing the target cell, subgoal completed
            if np.array_equal(datum, fwd_pos):
                return self.get_action_from_subgoal(*alternative)

            # Try to find a path
            path, _, _ = self.shortest_path(
                lambda pos, cell: np.array_equal(pos, datum)
            )
            blocked_path = False

            # Before checking if there is a path with blockers, maybe we should explore a bit to see if a non blocker path exists

            if not no_reexplore and not path:
                try:
                   # print('gcv')
                    return self.get_action_from_subgoal('Explore', 'no_reexplore')
                except (DisappearedBoxError, AssertionError) as e:
                    pass

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
                       # print('found it')
                blocked_path = True

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
                       # print('yoooo {}'.format(drop_pos_cur))
                        return self.get_action_from_subgoal('GoNextTo', drop_pos_cur, ('Drop', None))
                    else:
                        return actions.pickup
                #print("that's why you're going forward, path {}".format(path))
                return actions.forward

            #print(path, next_cell)
            # Turn towards the direction we need to go
            else:
                def closest_wall_or_door_given_dir(position, direction):
                    distance = 1
                    while True:
                        position_to_try = position + distance * direction
                        # If the current position is outside the field of view, stop everything and return the previous one
                        if not self.mission.in_view(*position_to_try):
                            return distance - 1
                        cell = self.mission.grid.get(*position_to_try)
                        if cell and (cell.type.endswith('door') or cell.type == 'wall'):
                            return distance
                        distance += 1

                if np.array_equal(next_cell - pos, right_vec):
                    return actions.right
                elif np.array_equal(next_cell - pos, - right_vec):
                    return actions.left
                # well then the cell is behind us, instead of choosing left or right randomly, let's do something that might be useful
                # Because when we're GoingNextTo for the purpose of exploring, things might change while on the way to the position we're going to
                # let's pick this right or left wisely
                # The simplest thing we should do is: pick the one that doesn't lead you to face a non empty cell
                # We can do better of course
                # if not self.mission.grid.get(*(pos + right_vec)):
                #    return actions.right
                # One better thing would be to go to the direction where the closest wall/door is the furthese
                distance_right = closest_wall_or_door_given_dir(pos, right_vec)
                distance_left = closest_wall_or_door_given_dir(pos, - right_vec)
                if distance_left > distance_right:
                    return actions.left
                return actions.right

        if subgoal == 'UpdateStackIfNecessary':
            assert len(self.stack) >= 1
            obj, old_obj_pos = datum
            obj_pos = self.find_obj_pos(obj)
            # Check if Obj has been moved
            if obj_pos != old_obj_pos:
                return self.get_action_from_subgoal('GoToAdjPos', obj)

            return self.get_action_from_subgoal(*self.stack[-2])

        # Go to next to a position adjacent to an object
        if subgoal == 'GoToAdjPos':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)

            if not obj_pos:
                return self.get_action_from_subgoal('Explore', None)

            # Find the closest position adjacent to the object
            path, adj_pos, _ = self.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
            )

            if not adj_pos:
                path, adj_pos, _ = self.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                    ignore_blockers=True
                )

            if not adj_pos:
                return self.get_action_from_subgoal('Explore', None)

            # FIXME: h4xx ??
            # If we are on the target position,
            # Randomly navigate away from this position
            if np.array_equal(pos, adj_pos):
                if not fwd_cell:
                    # Empty cell ahead, go there
                    return actions.forward
                # TODO: maybe introduce a "closest_wall_or_door_given_dir" function to decide between right and left
                return actions.left
            return self.get_action_from_subgoal('GoNextTo', adj_pos)

        # Explore the world, uncover new unseen cells
        if subgoal == 'Explore':
            #print(self.vis_mask)
            # Find the closest unseen position
            _, unseen_pos, _ = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos]
            )

            if not unseen_pos:
                _, unseen_pos, _ = self.shortest_path(
                    lambda pos, cell: not self.vis_mask[pos],
                    ignore_blockers=True
                )

            if unseen_pos:

                #print(unseen_pos)

                return self.get_action_from_subgoal('GoNextTo', {'dest': unseen_pos,
                                                                 'no_reexplore': datum == 'no_reexplore'}
                                                    )

            # don't open a door before checking if the target is accessible via a blocked path
            if datum == 'no_reexplore':
                raise DisappearedBoxError('not really a disappeared box error but yeah, let\'s change later')

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
            _, door_pos, _ = self.shortest_path(unopened_unlocked_door)
            if not door_pos:
                _, door_pos, _ = self.shortest_path(unopened_unlocked_door, ignore_blockers=True)
            if not door_pos:
                _, door_pos, _ = self.shortest_path(unopened_door)
            if not door_pos:
                _, door_pos, _ = self.shortest_path(unopened_door, ignore_blockers=True)

            # Open the door
            if door_pos:
                door = self.mission.grid.get(*door_pos)
               # print('fsdfsd {}'.format(door_pos))
                return self.get_action_from_subgoal('GoNextTo', {'dest': door_pos,
                                                                 'no_reexplore': True}
                                                    , ('Open', None))

            # Find the closest unseen position, ignoring blocking objects
            path, unseen_pos, _ = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos],
                ignore_blockers=True
            )

            if unseen_pos:
                return self.get_action_from_subgoal('GoNextTo', unseen_pos)

            ### print(self.stack)
            assert False, "0nothing left to explore"

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
       # print('before {}: {}, direction {}'.format(self.step_count, self.stack, self.mission.dir_vec))

        finished_updating = False
        while not finished_updating:
            finished_updating = self.take_action_iterate(action)

       # print('after {}: {}, direction {}'.format(self.step_count, self.stack, self.mission.dir_vec))

    def take_action_iterate(self, action):
       # print('{}: {}, direction {}'.format(self.step_count, self.stack, self.mission.dir_vec))
        # TODO: make this work with done action ?
        pos = self.mission.agent_pos
        dir_vec = self.mission.dir_vec
        #print(dir_vec)
        right_vec = self.mission.right_vec
        fwd_pos = pos + dir_vec

        actions = self.mission.actions
        carrying = self.mission.carrying

        if len(self.stack) == 0:
            # Is this right?
            if action != actions.done:
                self.stack.append(('Forget', None))
                self.stack.append(('GoNextTo', fwd_pos))
            return True

        # Get the topmost instruction on the stack
        subgoal, datum = self.stack[-1]

        # AS OF 25/09/2018, this subgoal doesn't exist ! Just thinking about the future if we have a mission with such a subgoal !
        if subgoal != 'OpenBox':
            if action == actions.toggle and self.mission.grid.get(*fwd_pos) is not None and self.mission.grid.get(*fwd_pos).type == 'box':
                # basically the foolish agent opens the box, the box doesn't exist anymore (MAGIC !)
                # throw exception and consider the instance unsolvable
                # TODO: give the agent another chance by looking for a similar box still unopened, or not care if the box is irrelevant (need to update stack for example if that box was moved and bot advises to put it back to its position after it has been opened by mistake)
                # return False
                raise DisappearedBoxError('A box was opened, too bad !')

        while subgoal == 'Forget':
            self.stack.pop()
            assert len(self.stack) != 0
            subgoal, datum = self.stack[-1]

        if self.itr_count >= self.timeout:
            raise TimeoutError('bot timed out')

        self.itr_count += 1
        #print(self.itr_count)

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
                        self.stack.pop()
                        self.stack.append(('Open', None))
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
            return True

        # Drop an object
        if subgoal == 'Drop':
            if action == actions.drop:
                self.stack.pop()
            # these are the only actions that can change your state (you are already carrying something, and there is nothing in front of you !)
            elif action in (actions.left, actions.right, actions.forward):
                #print('here')
                # Go back to where you were to drop what you got
                self.stack.append(('GoNextTo', tuple(fwd_pos)))
                #print(self.stack)
            return True

        def drop_or_pickup_or_open_something_while_exploring():
            #print(carrying, new_carrying)
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



        # Go to an object
        if subgoal == 'GoToObj':
            # Do we know where any one of these objects are?
            obj_pos = self.find_obj_pos(datum)
            #print('1{}'.format(obj_pos))
            #print('datum{} obj_pos {}'.format(datum, obj_pos))

            # Go to the location of this object
            if obj_pos:
                #print(0)
                # If we are right in front of the object,
                # go back to the previous subgoal
                if np.array_equal(obj_pos, fwd_pos):
                    #print(1)
                    self.stack.pop()
                    return False
                if action in (actions.forward, actions.left, actions.right) and np.array_equal(obj_pos, new_fwd_pos):
                    # stack will be popped at next step
                    #print(2)
                    return True
                fwd_cell = self.mission.grid.get(*fwd_pos)
                # If there is a blocking object in front of us
                if fwd_cell and not fwd_cell.type.endswith('door') and action == actions.pickup:
                    #print('DIS')
                    self.stack.append(('GoNextTo', obj_pos))
                    return False
                elif action in (actions.drop, actions.pickup, actions.toggle):
                    #print(4)
                    drop_or_pickup_or_open_something_while_exploring()
                    return True
                else:
                    #print(5)
                    path, _, _ = self.shortest_path(
                        lambda pos, cell: pos == obj_pos
                    )

                    if not path:
                        #print(6)
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
                        #print(7)
                        # New subgoal: go next to the object
                        self.stack.append(('GoNextTo', obj_pos))
                        return True
            #print('{} HEEERE'.format(self.step_count))
            # Explore the world
            #print('takeexp')
            self.stack.append(('Explore', None))
            return False

        # Go to a given location
        if subgoal == 'GoNextTo':
            why = None  # Reason why whe're going next to something
            blocker = False
            if isinstance(datum, dict):
                if 'why' in datum.keys():
                    why = datum['why']
                if 'blocker' in datum.keys():
                    blocker = datum['blocker']
                datum = datum['dest']
            #print(subgoal)
            try:
                if tuple(pos) == tuple(datum):
                    if action in (actions.drop, actions.pickup, actions.toggle):
                        drop_or_pickup_or_open_something_while_exploring()
                    return True
                if tuple(new_pos) == tuple(datum):
                    return True
            except ValueError:
                print(tuple(pos), datum)

            # If we are facing the target cell, subgoal completed
            if np.array_equal(datum, fwd_pos):

                self.stack.pop()
                #print("popped this")
                return False
            if action in (actions.forward, actions.left, actions.right) and np.array_equal(datum, new_fwd_pos):
                # stack will be popped at next step anyway
                return True
            else:
                # Try to find a path
                path, _, _ = self.shortest_path(
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
                    self.stack.append(('Explore', None))
                    return True

                next_cell = path[0]

                # If the destination is ahead of us
                if np.array_equal(next_cell, fwd_pos):
                    fwd_cell = self.mission.grid.get(*fwd_pos)

                    # If there is a blocking object in front of us
                    if fwd_cell and not fwd_cell.type.endswith('door'):
                       # print('that\'s what im doin')

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
                            return True
                        else:
                            drop_pos = self.find_drop_pos()
                            self.stack.append(('Drop', None))
                            self.stack.append(('GoNextTo', drop_pos))
                            if action == actions.pickup:
                                return True
                            else:
                                self.stack.append(('Pickup', None))
                                self.stack.append(('GoNextTo', fwd_pos))
                                return True
                    #if action == actions.pikcup:
                    # TODO: what if I drop a blocker not in drop_pos ? Can I make drop_pos a list ? Please be the case !
                    # TODO: what if I pickup another blocker (and that's good)
                # If there is nothing blocking us and we drop/pickup/toggle something for no reason
                if action in (actions.drop, actions.pickup, actions.toggle):
                    drop_or_pickup_or_open_something_while_exploring()
                if why == 'exploring':
                    if not blocker:
                       # print('?HERE !')
                        self.stack.pop()
                    else:
                       # print('!HERE ?')
                        if self.vis_mask[datum]:
                           # print('!HERE2 ?')
                            self.stack.pop()
                return True

        if subgoal == 'UpdateObjsPoss':
            #print("Updating Objs Poss")
            self.stack.pop()
            self.update_objs_poss()
            return False

        # Go to next to a position adjacent to an object
        if subgoal == 'GoToAdjPos':
            #print("I'm here")
            # Do we know where any one of these objects are?
            #print(datum)
            obj_pos = self.find_obj_pos(datum)
            #print(obj_pos)

            if action in (actions.drop, actions.pickup, actions.toggle):
                drop_or_pickup_or_open_something_while_exploring()
                return True
            else:
                if not obj_pos:
                    self.stack.append(('Explore', None))
                    return False

                # Find the closest position adjacent to the object
                path, adj_pos, _ = self.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
                )

                if not adj_pos:
                    path, adj_pos, _ = self.shortest_path(
                        lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                        ignore_blockers=True
                    )

                if not adj_pos:
                    self.stack.append(('Explore', None))
                    return True

                # FIXME: h4xx ??
                # If we are on the target position,
                # Randomly navigate away from this position
                if action in (actions.forward, actions.left, actions.right):
                    if np.array_equal(pos, adj_pos):
                        return True
                    else:
                        self.stack.pop()
                        self.stack.append(('UpdateStackIfNecessary', (datum, obj_pos)))
                        self.stack.append(('GoNextTo', adj_pos))
                    return False
                return True

        if subgoal == 'UpdateStackIfNecessary':
            '''
            The only reason this should be used (for now) would be when we need
            to unblock the way by moving an object that we should have gone
            adjacent to, and the GoToAdjPos had been removed and replaced by
            a GoNextTo subgoal
            Note that in this scenario, UpdateObjsPoss should have already been called.
            datum is a tuple containing the object we should go adjacent to and its previous position
            '''
            self.stack.pop()
            obj, old_obj_pos = datum
            obj_pos = self.find_obj_pos(obj)
            # Check if Obj has been moved
            if obj_pos != old_obj_pos:
                self.stack.append(('GoToAdjPos', obj))
            return False

        # Explore the world, uncover new unseen cells
        if subgoal == 'Explore':
            # Find the closest unseen position
            _, unseen_pos, _ = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos]
            )
            #print('unseen {}'.format(unseen_pos))

            if unseen_pos:
                #print('did this')
                self.stack.pop()
                self.stack.append(('GoNextTo', {'dest': unseen_pos,
                                                'why': 'exploring',
                                                'blocker': False}))
                return False

            if not unseen_pos:
                _, unseen_pos, _ = self.shortest_path(
                    lambda pos, cell: not self.vis_mask[pos],
                    ignore_blockers=True
                )

            if unseen_pos:
                #print('did this')
                self.stack.pop()
                self.stack.append(('GoNextTo', {'dest': unseen_pos,
                                                'why': 'exploring',
                                                'blocker': True}))
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
            _, door_pos, _ = self.shortest_path(unopened_unlocked_door)
            if not door_pos:
                _, door_pos, _ = self.shortest_path(unopened_unlocked_door, ignore_blockers=True)
            if not door_pos:
                _, door_pos, _ = self.shortest_path(unopened_door)
            if not door_pos:
                _, door_pos, _ = self.shortest_path(unopened_door, ignore_blockers=True)
            #print(0)
            # Open the door
            if door_pos:
                #print(1)
                door = self.mission.grid.get(*door_pos)
                self.stack.pop()
                self.stack.append(('Open', None))
                self.stack.append(('GoNextTo', door_pos))
                return False

            # Find the closest unseen position, ignoring blocking objects
            path, unseen_pos, _ = self.shortest_path(
                lambda pos, cell: not self.vis_mask[pos],
                ignore_blockers=True
            )

            if unseen_pos:
                self.stack.pop()
                self.stack.append(('GoNextTo', {'dest': unseen_pos,
                                                'why': 'exploring',
                                                'blocker': True}))
                return False

            #print(self.stack)
            assert False, "1nothing left to explore"

        assert False, 'invalid subgoal "%s"' % subgoal

    # The following function is for testing purposes only
    def step(self):
        action = self.get_action()
        self.take_action(action)