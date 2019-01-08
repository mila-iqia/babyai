from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)


class DisappearedBoxError(Exception):
    """
    Error that's thrown when a box is opened.
    We make the assumption that the bot cannot accomplish the mission when it happens.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Subgoal:
    """The base class for all possible Bot subgoals"""

    def __init__(self, bot=None, datum=None):
        """
        Initializes a Subgoal object
        bot is the Bot that is trying to perform this subgoal
        datum is the primary information necessary to accomplish the subgoal
        These subgoals have extra class variables defined in theyr own __init__ method: GoNextTo
        the rest are secondary information that are used to make the bot more efficient
        """
        self.bot = bot
        self.datum = datum

        self.update_agent_attributes()

        self.actions = self.bot.mission.actions

    def update_agent_attributes(self):
        """Should be called at each step to update some elements about the agent and the environment"""
        self.pos = self.bot.mission.agent_pos
        self.dir_vec = self.bot.mission.dir_vec
        self.right_vec = self.bot.mission.right_vec
        self.fwd_pos = self.pos + self.dir_vec
        self.fwd_cell = self.bot.mission.grid.get(*self.fwd_pos)
        self.carrying = self.bot.mission.carrying

    def __repr__(self):
        """Mainly for debugging purposes"""
        representation = '('
        representation += type(self).__name__
        if self.datum is not None:
            representation += ': {}'.format(self.datum)
        representation += ')'

        return representation

    def get_action(self):
        """
        Function that gives the optimal action given the current subgoal. It may call the `get_action` function of other
        subgoals.
        Returns an action.
        """
        pass

    def subgoal_accomplished(self):
        self.bot.stack.pop()
        if len(self.bot.stack) >= 1:
            subgoal = self.bot.stack[-1]
            subgoal.update_agent_attributes()
            return subgoal.get_action()
        return self.bot.mission.actions.done

    def simulate_step(self, action):
        """
        Function that simulates an action by creating variables descrbing the new potential state of the agent
        """
        self.new_pos = self.pos
        agent_dir = self.bot.mission.agent_dir
        self.new_carrying = self.carrying

        # Rotate left
        if action == self.actions.left:
            agent_dir -= 1
            if agent_dir < 0:
                agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            agent_dir = (agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if self.fwd_cell is None or self.fwd_cell.can_overlap():
                self.new_pos = self.fwd_pos

        # Pick up an object
        elif action == self.actions.pickup:
            if self.fwd_cell is not None and self.fwd_cell.can_pickup():
                if self.carrying is None:
                    self.new_carrying = self.fwd_cell

        # Drop an object
        elif action == self.actions.drop:
            if self.fwd_cell is None and self.carrying is not None:
                self.new_carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            pass

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        self.new_dir_vec = DIR_TO_VEC[agent_dir]
        self.new_right_vec = np.array((-self.new_dir_vec[1], self.new_dir_vec[0]))
        self.new_fwd_pos = self.new_pos + self.new_dir_vec

    def foolish_action_while_exploring(self, action):
        '''
        A bit too conservative here:
        If you pick up an object when you shouldn't have, you should drop it back in the same position
        If you drop an object when you shouldn't have, you should pick that one up, and not one similar to it
        If you a close a door when you shouldn't have, you should open it back
        TODO: relax this
        '''

        if action == self.actions.drop and self.carrying != self.new_carrying:
            # get that thing back
            new_fwd_cell = self.carrying
            assert new_fwd_cell.type in ('key', 'box', 'ball')
            self.bot.stack.append(PickupSubgoal(self.bot))

        elif action == self.actions.pickup and self.carrying != self.new_carrying:
            # drop that thing where you found it
            fwd_cell = self.bot.mission.grid.get(*self.fwd_pos)
            assert fwd_cell.type in ('key', 'box', 'ball')
            self.bot.stack.append(DropSubgoal(self.bot))

        elif action == self.actions.toggle:
            fwd_cell = self.bot.mission.grid.get(*self.fwd_pos)
            if fwd_cell and fwd_cell.type == 'door' and fwd_cell.is_open:
                # i.e. the agent decided to close the door
                # need to open it
                self.bot.stack.append(OpenSubgoal(self.bot))

    def take_action(self, action):
        """
        Function that updates the bot's stack given the action played
        Should be overridden in all sub-classes
        TODO: There are some steps that are common in both take_action and get_action for certain subgoals, maybe the bot's speed can be improved if we do them only once - or at least we can factorize the code a bit
        """
        self.erroneous_box_opening(action)
        self.simulate_step(action)

    def erroneous_box_opening(self, action):
        """
        When the agent opens a box, we raise an error and mark the task unsolvable.
        This is a tad conservative, because maybe the box is irrelevant to the mission.
        TODO: We can relax this by checking if the opened box is crucial for the mission.
        TODO: We can relax this by checking if a similar box still exists if the box is crucial.
        """
        if action == self.actions.toggle and self.fwd_cell is not None and self.fwd_cell.type == 'box':
            raise DisappearedBoxError('A box was opened. Too Bad :(')


class OpenSubgoal(Subgoal):
    def get_action(self):
        super().get_action()
        assert self.fwd_cell is not None, 'Forward cell is empty'
        assert self.fwd_cell.type == 'door', 'Forward cell has to be a door'

        # If the door is locked, go find the key and then return
        # TODO: do we really need to be in front of the locked door to realize that we need the key for it ?
        if self.fwd_cell.type == 'door' and self.fwd_cell.is_locked:
            if not self.carrying or self.carrying.type != 'key' or self.carrying.color != self.fwd_cell.color:
                # Find the key
                key_desc = ObjDesc('key', self.fwd_cell.color)
                key_desc.find_matching_objs(self.bot.mission)

                # If we're already carrying something
                if self.carrying:

                    # Find a location to drop what we're already carrying
                    drop_pos_cur = self.bot.find_drop_pos()
                    if np.array_equal(drop_pos_cur, self.fwd_pos):
                        new_subgoal = DropSubgoal(self.bot)
                    else:
                        new_subgoal = GoNextToSubgoal(self.bot, drop_pos_cur)
                else:
                    # Go To the key
                    obj_pos = self.bot.find_obj_pos(key_desc)
                    if obj_pos is not None and np.array_equal(obj_pos, self.fwd_pos):
                        new_subgoal = PickupSubgoal(self.bot)
                    else:
                        new_subgoal = GoToObjSubgoal(self.bot, key_desc)
                return new_subgoal.get_action()

        # If the door is already open, close it so we can open it again
        if self.fwd_cell.type == 'door' and self.fwd_cell.is_open:
            return self.actions.toggle

        return self.actions.toggle

    def take_action(self, action):
        super(OpenSubgoal, self).take_action(action)

        # CASE 1: The door is locked
        # we need to fetch the key and return
        # i.e. update the stack REGARDLESS of the action
        # TODO: do we really need to be in front of the locked door to realize that we need the key for it ?
        if self.fwd_cell.type == 'door' and self.fwd_cell.is_locked:
            if not self.carrying or self.carrying.type != 'key' or self.carrying.color != self.fwd_cell.color:
                # Find the key
                key_desc = ObjDesc('key', self.fwd_cell.color)
                key_desc.find_matching_objs(self.bot.mission)

                # If we're already carrying something
                if self.carrying:
                    self.bot.stack.pop()

                    # Find a location to drop what we're already carrying
                    drop_pos_cur = self.bot.find_drop_pos()

                    # Take back the object being carried
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

                    # Go back to the door and open it
                    self.bot.stack.append(OpenSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))

                    # Go to the key and pick it up
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoToObjSubgoal(self.bot, key_desc))

                    # Drop the object being carried
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
                else:
                    self.bot.stack.pop()

                    self.bot.stack.append(OpenSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoToObjSubgoal(self.bot, key_desc))

                return False

        # CASE 2: The door is already open
        # we need to close it so we can open it again
        if self.fwd_cell.type == 'door' and self.fwd_cell.is_open:
            if action in (self.actions.left, self.actions.right, self.actions.forward):
                # Go back to the door to close it
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))
            # done/pickup/drop actions won't have any effect -> next iteration would be CASE 2 of same subgoal
            # toggle action will close the door, but we would need to go through the same subgoal again
            # -> next iteration would be CASE 3 of same subgoal
            return True

        # CASE 3: The door is openable
        if action == self.actions.toggle:
            self.bot.stack.pop()
        if action in (self.actions.left, self.actions.right):
            # Go back to the door to open it
            self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))
        # done/pickup/drop/forward actions won't have any effect -> next iteration would be CASE 3 of same subgoal
        return True


class DropSubgoal(Subgoal):
    def get_action(self):
        super().get_action()
        return self.actions.drop

    def take_action(self, action):
        super(DropSubgoal, self).take_action(action)
        if action == self.actions.drop:
            self.bot.stack.pop()
            if self.datum is not None:
                # this means that the object we just dropped was initially in self.datum
                # maybe after dropping the object, we were supposed to go to it again, and it was referred to
                # in the current stack by its old position (self.datum) -> need to loop through the stack and fix it
                for subgoal in self.bot.stack:
                    if isinstance(subgoal, GoNextToSubgoal):
                        if np.array_equal(subgoal.datum, self.datum):
                            subgoal.datum = self.fwd_pos
        elif action in (self.actions.left, self.actions.right, self.actions.forward):
            # Go back to where you were to drop what you got
            self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))
        # done/pickup actions won't have any effect -> Next step would take us to the same subgoal
        return True


class PickupSubgoal(Subgoal):
    def get_action(self):
        super().get_action()
        return self.actions.pickup

    def take_action(self, action):
        super(PickupSubgoal, self).take_action(action)
        if action == self.actions.pickup:
            self.bot.stack.pop()
        elif action in (self.actions.left, self.actions.right):
            # Go back to where you were to pickup what was in front of you
            self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))
        # done/drop/forward actions won't have any effect -> Next step would take us to the same subgoal
        return True


class GoToObjSubgoal(Subgoal):
    def get_action(self):
        super().get_action()
        # Do we know where any one of these objects are?
        obj_pos = self.bot.find_obj_pos(self.datum)

        # Go to the location of this object
        if obj_pos:
            # Stupid corner case: We have to go to a door, which is open, and we are currently in that position
            if np.array_equal(obj_pos, self.pos):
                # -> Move away from it
                if self.fwd_cell is None:
                    return self.actions.forward
                if self.bot.mission.grid.get(*(self.pos + self.right_vec)) is None:
                    return self.actions.right
                if self.bot.mission.grid.get(*(self.pos - self.right_vec)) is None:
                    return self.actions.left
                # TODO: There is a corner case : you are on the position but surrounded in 4 positions - need to open a box or pickup an object to free yourself
                # otherwise: forward, left and right are full, we assume that you can go behind
                return self.actions.left

            # If we are right in front of the object, then the subgoal is completed
            # go back to the previous subgoal
            if np.array_equal(obj_pos, self.fwd_pos):
                return self.subgoal_accomplished()

            # Look for a non-blocker path leading to the position
            path, _, _ = self.bot.shortest_path(
                lambda pos, cell: pos == obj_pos
            )

            if not path:
                # TODO: here, and every time we use this 999 number, find a cleaner/better way
                # Look for a blocker path leading to the position, with the condition that the blocker is as close to
                # the target as possible (Hence the while loop)
                next_search_dist = 999
                while next_search_dist is not None:
                    suggested_path, _, next_search_dist = self.bot.shortest_path(
                        lambda pos, cell: pos == obj_pos,
                        ignore_blockers=True,
                        blocker_fn=lambda pos: self.bot.blocker_fn(pos, obj_pos, next_search_dist),
                        distance_fn=lambda pos: self.bot.distance(pos, obj_pos)
                    )
                    if suggested_path:
                        path = [elem for elem in suggested_path]

            if path:
                # Get the action from the "GoNextTo" subgoal
                new_subgoal = GoNextToSubgoal(self.bot, obj_pos, reason='GoToObj')
                return new_subgoal.get_action()

        # No path found -> Explore the world
        return ExploreSubgoal(self.bot).get_action()

    def take_action(self, action):
        super(GoToObjSubgoal, self).take_action(action)
        # Do we know where any one of these objects are?
        obj_pos = self.bot.find_obj_pos(self.datum)

        # Stupid corner case: We have to go to a door, which is open, and we are currently in that position
        if np.array_equal(obj_pos, self.pos):
            # -> Move away from it
            if action in (self.actions.drop, self.actions.pickup, self.actions.toggle):
                # Update the stack if we did something bad, or do nothing if the action doesn't change anything
                self.foolish_action_while_exploring(action)
            # Whatever other action, the stack should stay the same, and it's the new action that should be evaluated
            # TODO: Double check what happens in this scenario with all actions
            return True

        # CASE 1: the bot has already seen the object
        if obj_pos:
            # CASE 1.1: we are already right in front of the object
            # we need go back to the previous subgoal, and take it into account to handle the current action
            if np.array_equal(obj_pos, self.fwd_pos):
                self.bot.stack.pop()
                return False

            # CASE 1.2: The action will make us in front of the object
            if np.array_equal(obj_pos, self.new_fwd_pos):
                assert action in (self.actions.forward, self.actions.left, self.actions.right), "Something seems wrong"
                # stack will be popped at next step
                return True
                # It is impossible that it's another action that yields this CASE 1.2

            # CASE 1.3: There is a blocking object in front of us, and we aren't carrying anything
            if self.fwd_cell is not None and not self.fwd_cell.type.endswith('door') and self.carrying is None:
                if action == self.actions.pickup:
                    # -> Create a GoNextTo subgoal
                    # TODO: I am not sure this is enough. Shouldn't we drop the object somewhere?
                    self.bot.stack.append(GoNextToSubgoal(self.bot, obj_pos))
                    # TODO: Double check that False should be returned here
                    return False

            # CASE 1.4: All other scenarios
            if action in (self.actions.drop, self.actions.pickup, self.actions.toggle):
                # Update the stack if we did something bad, or do nothing if the action doesn't change anything
                self.foolish_action_while_exploring(action)
                return True

            if action == self.actions.done or (self.fwd_cell is not None and action == self.actions.forward):
                # Do nothing if the action doesn't change anything
                return True

            # Otherwise, the taken action (forward, left or right) has necessarily changed the agent state
            # -> Create a GoNextTo subgoal
            else:
                path, _, _ = self.bot.shortest_path(
                    lambda pos, cell: pos == obj_pos
                )

                if not path:
                    next_search_dist = 999
                    while next_search_dist is not None:
                        suggested_path, _, next_search_dist = self.bot.shortest_path(
                            lambda pos, cell: pos == obj_pos,
                            ignore_blockers=True,
                            blocker_fn=lambda pos: self.bot.blocker_fn(pos, obj_pos, next_search_dist),
                            distance_fn=lambda pos: self.bot.distance(pos, obj_pos)
                        )
                        if suggested_path:
                            path = [elem for elem in suggested_path]

                if path:
                    # New subgoal: go next to the object
                    self.bot.stack.append(GoNextToSubgoal(self.bot, obj_pos, reason='GoToObj'))
                    return True

        # CASE 2: The bot hasn't seen the object. The action will count for the new subgoal
        self.bot.stack.append(ExploreSubgoal(self.bot))
        return False


class GoNextToSubgoal(Subgoal):
    def __init__(self, bot=None, datum=None, reason=None, no_reexplore=False, blocker=False):
        super().__init__(bot, datum)
        self.reason = reason  # Reason we are performing this subgoal. Possiblities: Explore, GoToObj
        self.no_reexplore = no_reexplore  # Flag to know if Exploring in the same room  is a good idea
        self.blocker = blocker  # Flag to know if the path considered for exploration has a blocker or not
        # TODO: I was mistakengly initiating blocker to be always False, it worked fine. Now that it's fixed, make sure it sill works, and ideally is better, otherwise revert and see again the purpose of this blocker thing

    def __repr__(self):
        """Mainly for debugging purposes"""
        representation = '('
        representation += type(self).__name__
        if self.datum is not None:
            representation += ': {}'.format(self.datum)
        if self.reason is not None:
            representation += ', reason: {}'.format(self.reason)
        if self.no_reexplore:
            representation += ', no reexplore'
        if self.blocker:
            representation += ', blocker path'
        representation += ')'

        return representation

    def get_action(self):
        super().get_action()
        # CASE 1: The position we are on is the one we should go next to
        # -> Move away from it
        if tuple(self.pos) == tuple(self.datum):
            if self.fwd_cell is None:
                return self.actions.forward
            if self.bot.mission.grid.get(*(self.pos + self.right_vec)) is None:
                return self.actions.right
            if self.bot.mission.grid.get(*(self.pos - self.right_vec)) is None:
                return self.actions.left
            # TODO: There is a corner case : you are on the position but surrounded in 4 positions
            # otherwise: forward, left and right are full, we assume that you can go behind
            return self.actions.left

        # CASE 2: we are facing the target cell, subgoal completed
        if np.array_equal(self.datum, self.fwd_pos):
            return self.subgoal_accomplished()

        # CASE 3: we are still far from the target
        # Try to find a non-blocker path
        path, _, _ = self.bot.shortest_path(
            lambda pos, cell: np.array_equal(pos, self.datum)
        )

        # CASE 3.1: No non-blocker path found, and reexploration is allowed
        # -> Explore in the same room to see if a non-blocker path exists
        if not self.no_reexplore and path is None:
            # Find the closest unseen position
            _, unseen_pos, _ = self.bot.shortest_path(
                lambda pos, cell: not self.bot.vis_mask[pos]
            )

            # TODO: I don't think unseen_pos is necessarily in the same room. Do we want that though ?
            if unseen_pos is not None:
                new_subgoal = GoNextToSubgoal(self.bot, unseen_pos, no_reexplore=True, reason='Explore')
                return new_subgoal.get_action()

        # CASE 3.2: No non-blocker path found and (reexploration is not allowed or nothing to explore)
        # -> Look for blocker paths
        if not path:
            next_search_dist = 999
            while next_search_dist is not None:
                suggested_path, _, next_search_dist = self.bot.shortest_path(
                    lambda pos, cell: np.array_equal(pos, self.datum),
                    ignore_blockers=True,
                    blocker_fn=lambda pos: self.bot.blocker_fn(pos, self.datum, next_search_dist),
                    distance_fn=lambda pos: self.bot.distance(pos, self.datum)
                )
                if suggested_path:
                    path = [elem for elem in suggested_path]

        # CASE 3.2.1: No path found
        # -> explore the world
        if not path:
            return ExploreSubgoal(self.bot).get_action()

        # CASE3.4 = CASE 3.2.2: Found a blocker path OR CASE 3.3: Found a non-blocker path
        next_cell = path[0]
        # CASE 3.4.1: the forward cell is the one we should go next to
        if np.array_equal(next_cell, self.fwd_pos):
            if self.fwd_cell is not None and not self.fwd_cell.type.endswith('door'):
                if self.carrying:
                    drop_pos_cur = self.bot.find_drop_pos()
                    # Drop the object being carried
                    # 1-Just in case we are already at the right position
                    assert not np.array_equal(drop_pos_cur, next_cell), "Drop position is forward cell, weird!"
                    # Just in case we stumble this AssertionError, we should call the DropSubgoal instead !
                    new_subgoal = GoNextToSubgoal(self.bot, drop_pos_cur)
                    return new_subgoal.get_action()
                else:
                    return self.actions.pickup

            return self.actions.forward

        # CASE 3.4.2: the forward cell is not the one we should go to
        # -> Turn towards the direction we need to go
        else:
            def closest_wall_or_door_given_dir(position, direction):
                distance = 1
                while True:
                    position_to_try = position + distance * direction
                    # If the current position is outside the field of view, stop everything and return the previous one
                    if not self.bot.mission.in_view(*position_to_try):
                        return distance - 1
                    cell = self.bot.mission.grid.get(*position_to_try)
                    if cell and (cell.type.endswith('door') or cell.type == 'wall'):
                        return distance
                    distance += 1

            if np.array_equal(next_cell - self.pos, self.right_vec):
                return self.actions.right
            elif np.array_equal(next_cell - self.pos, - self.right_vec):
                return self.actions.left
            # well then the cell is behind us, instead of choosing left or right randomly,
            # let's do something that might be useful:
            # Because when we're GoingNextTo for the purpose of exploring,
            # things might change while on the way to the position we're going to, we should
            # pick this right or left wisely.
            # The simplest thing we should do is: pick the one that doesn't lead you to face a non empty cell.
            # One better thing would be to go to the direction where the closest wall/door is the furthest
            distance_right = closest_wall_or_door_given_dir(self.pos, self.right_vec)
            distance_left = closest_wall_or_door_given_dir(self.pos, - self.right_vec)
            if distance_left > distance_right:
                return self.actions.left
            return self.actions.right

    def take_action(self, action):
        super(GoNextToSubgoal, self).take_action(action)

        # CASE 1: The position we are on is the one we should go next to
        # -> Move away from it
        if tuple(self.pos) == tuple(self.datum):
            if action in (self.actions.drop, self.actions.pickup, self.actions.toggle):
                # Update the stack if we did something bad, or do nothing if the action doesn't change anything
                self.foolish_action_while_exploring(action)
            # Whatever other action, the stack should stay the same, and it's the new action that should be evaluated
            # TODO: Double check what happens in this scenario with all actions
            return True

        # CASE 2: we are facing the target cell, subgoal completed
        if np.array_equal(self.datum, self.fwd_pos):
            self.bot.stack.pop()
            return False

        assert tuple(self.new_pos) != tuple(self.datum), "Doesn't make sense with CASE 2"

        # CASE 3: the action taken would lead us to face the target cell
        # -> Don't do anything. The stack will be popped at next step anyway
        if np.array_equal(self.datum, self.new_fwd_pos):
            assert action in (self.actions.forward, self.actions.left, self.actions.right), "Doesn't make sense"
            return True

        # CASE 4: The reason for this subgoal is because we are going to a certain object, and the played action changes
        # the field of view
        # -> remove the subgoal and make the 'GoToObj' subgoal recreate a new GoNextTo subgoal, potentially with
        # a new target, because moving might lead us to discover a similar object that's actually closer
        if self.reason == 'GoToObj':
            if action in (self.actions.forward, self.actions.left, self.actions.right):
                self.bot.stack.pop()
                return False

        # CASE 5: otherwise
        # Try to find a path
        path, _, _ = self.bot.shortest_path(
            lambda pos, cell: np.array_equal(pos, self.datum)
        )

        # CASE 5.1: If we failed to find a path, try again while ignoring blockers
        if not path:
            next_search_dist = 999
            while next_search_dist is not None:
                suggested_path, _, next_search_dist = self.bot.shortest_path(
                    lambda pos, cell: np.array_equal(pos, self.datum),
                    ignore_blockers=True,
                    blocker_fn=lambda pos: self.bot.blocker_fn(pos, self.datum, next_search_dist),
                    distance_fn=lambda pos: self.bot.distance(pos, self.datum)
                )
                if suggested_path:
                    path = [elem for elem in suggested_path]

        # CASE 5.2: No path found, explore the world
        if not path:
            # Explore the world
            self.bot.stack.append(ExploreSubgoal(self.bot))
            return True

        next_cell = path[0]

        # CASE 5.3: If the destination is ahead of us
        if np.array_equal(next_cell, self.fwd_pos):
            # If there is a blocking object in front of us
            if self.fwd_cell and not self.fwd_cell.type.endswith('door'):
                if self.carrying:
                    drop_pos_cur = self.bot.find_drop_pos()
                    drop_pos_block = self.bot.find_drop_pos(drop_pos_cur)

                    # Take back the object being carried
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

                    # Pick up the blocking object and drop it
                    self.bot.stack.append(DropSubgoal(self.bot, self.fwd_pos))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_block))
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, self.fwd_pos))

                    # Drop the object being carried
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
                    return True
                else:
                    drop_pos = self.bot.find_drop_pos()
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos))
                    if action == self.actions.pickup:
                        return True
                    else:
                        self.bot.stack.append(PickupSubgoal(self.bot))
                        self.bot.stack.append(GoNextToSubgoal(self.bot, self.fwd_pos))
                        return True

            # TODO: what if I drop a blocker not in drop_pos ? Can I make drop_pos a list ? Please be the case !
            # TODO: what if I pickup another blocker (and that's good)

        # CASE 5.4: If there is nothing blocking us and we drop/pickup/toggle something for no reason
        if action in (self.actions.drop, self.actions.pickup, self.actions.toggle):
            self.foolish_action_while_exploring(action)

        # CASE 5.5: If we are GoingNextTo something because of exploration
        if self.reason == 'Explore':
            if not self.blocker:
                self.bot.stack.pop()
            else:
                if self.bot.vis_mask[self.datum]:
                    self.bot.stack.pop()
        return True


class GoToAdjPosSubgoal(Subgoal):
    def get_action(self):
        super().get_action()
        # Look for the object
        obj_pos = self.bot.find_obj_pos(self.datum)

        # CASE 1: did not find it
        if not obj_pos:
            return ExploreSubgoal(self.bot).get_action()

        # CASE 2: found it
        # Find the closest position adjacent to the object
        path, adj_pos, _ = self.bot.shortest_path(
            lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
        )

        # CASE 2.1: did not find a non-blocker path
        if not adj_pos:
            path, adj_pos, _ = self.bot.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                ignore_blockers=True
            )

        # CASE 2.1.1: did not find any path
        if not adj_pos:
            return ExploreSubgoal(self.bot).get_action()

        # FIXME: h4xx ?? Salem: What's this FIXME @maxime ?

        # CASE 3: There is a path (blocker or not)
        # CASE 3.1: we are on the target position,
        # -> Randomly navigate away from this position
        if np.array_equal(self.pos, adj_pos):
            if not self.fwd_cell:
                # Empty cell ahead, go there
                return self.actions.forward
            # TODO: maybe introduce a "closest_wall_or_door_given_dir" function to decide between right and left
            return self.actions.left
        # CASE 3.2: We know the target position -> We have to go to it
        return GoNextToSubgoal(self.bot, adj_pos).get_action()

    def take_action(self, action):
        super(GoToAdjPosSubgoal, self).take_action(action)

        # Look for the object
        obj_pos = self.bot.find_obj_pos(self.datum)

        # TODO: is it really necessary to do all this? It works, but it seems like it's useless code. -> Double check
        # TODO: or actually maybe this is wrong, maybe the drop/pickup action is an okay thing given that GoToAdjPos might lead to a GoNextTo with blockers that need to be picked up e.g.
        if action in (self.actions.drop, self.actions.pickup, self.actions.toggle):
            self.foolish_action_while_exploring(action)
            return True

        else:
            # CASE 1: did not find it
            if not obj_pos:
                self.bot.stack.append(ExploreSubgoal(self.bot))
                return False

            # CASE 2: found it
            #
            # # CASE 2.0: we are in an okay position
            # if not self.fwd_cell and pos_next_to(obj_pos, self.fwd_pos):
            #     self.bot.stack.pop()
            #     return False
            #
            # Find the closest position adjacent to the object
            path, adj_pos, _ = self.bot.shortest_path(
                lambda pos, cell: not cell and pos_next_to(pos, obj_pos)
            )

            # CASE 2.1: did not find a non-blocker path
            if not adj_pos:
                path, adj_pos, _ = self.bot.shortest_path(
                    lambda pos, cell: not cell and pos_next_to(pos, obj_pos),
                    ignore_blockers=True
                )
            # CASE 2.1.1: did not find any path
            if not adj_pos:
                self.bot.stack.append(ExploreSubgoal(self.bot))
                return True

            # FIXME: h4xx ??Salem: What's this FIXME @maxime ?

            # CASE 3: There is a path (blocker or not)
            # CASE 3.1: we are on the target position,
            # -> Randomly navigate away from this position
            if action in (self.actions.forward, self.actions.left, self.actions.right):
                if np.array_equal(self.pos, adj_pos):
                    return True
                else:
                    self.bot.stack.pop()
                    self.bot.stack.append(UpdateStackSubgoal(self.bot, self.datum, obj_pos))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, adj_pos))
                    # self.bot.stack.append(GoNextToSubgoal(self.bot, adj_pos, reason='GoToAdjPos'))
                return False
            return True


class UpdateStackSubgoal(Subgoal):
    def __init__(self, bot, datum, obj_pos):
        super().__init__(bot, datum)
        self.obj_pos = obj_pos

    def get_action(self):
        '''
        The only reason this should be used (for now) would be when we need
        to unblock the way by moving an object that we should have gone
        adjacent to (desc_fixed), and the GoToAdjPos had been removed and replaced by
        a GoNextTo subgoal.
        TODO: This is not optimal at all. Ideally, the bot should know the new position of desc_fixed BEFORE going next to its old position. Those are unnecessary steps. e.g. Check PutNext seed 448
        '''
        assert len(self.bot.stack) >= 2
        old_obj_pos = self.obj_pos
        obj_pos = self.bot.find_obj_pos(self.datum)
        # Check if Obj has been moved
        if obj_pos != old_obj_pos:
            return GoToAdjPosSubgoal(self.bot, self.datum).get_action()

        return self.bot.stack[-2].get_action()

    def take_action(self, action):
        self.bot.stack.pop()
        old_obj_pos = self.obj_pos
        obj_pos = self.bot.find_obj_pos(self.datum)
        # Check if Obj has been moved
        if obj_pos != old_obj_pos:
            self.bot.stack.append(GoToAdjPosSubgoal(self.bot, self.datum))
        return False


class ExploreSubgoal(Subgoal):
    def get_action(self):
        super().get_action()
        # Find the closest unseen position
        _, unseen_pos, _ = self.bot.shortest_path(
            lambda pos, cell: not self.bot.vis_mask[pos]
        )

        if not unseen_pos:
            _, unseen_pos, _ = self.bot.shortest_path(
                lambda pos, cell: not self.bot.vis_mask[pos],
                ignore_blockers=True
            )

        if unseen_pos:
            return GoNextToSubgoal(self.bot, unseen_pos).get_action()

        # Find the closest unlocked unopened door
        def unopened_unlocked_door(pos, cell):
            return cell and cell.type == 'door' and not cell.is_locked and not cell.is_open

        # Find the closest unopened door
        def unopened_door(pos, cell):
            return cell and cell.type == 'door' and not cell.is_open

        # Try to find an unlocked door first
        # We do this because otherwise, opening a locked door as
        # a subgoal may try to open the same door for exploration,
        # resulting in an infinite loop
        _, door_pos, _ = self.bot.shortest_path(unopened_unlocked_door)
        if not door_pos:
            _, door_pos, _ = self.bot.shortest_path(unopened_unlocked_door, ignore_blockers=True)
        if not door_pos:
            _, door_pos, _ = self.bot.shortest_path(unopened_door)
        if not door_pos:
            _, door_pos, _ = self.bot.shortest_path(unopened_door, ignore_blockers=True)

        # Open the door
        if door_pos:
            # door = self.bot.mission.grid.get(*door_pos)
            if np.array_equal(door_pos, self.fwd_pos):
                subgoal = OpenSubgoal(self.bot)
            else:
                subgoal = GoNextToSubgoal(self.bot, door_pos, no_reexplore=True)
            return subgoal.get_action()

        # Find the closest unseen position, ignoring blocking objects
        path, unseen_pos, _ = self.bot.shortest_path(
            lambda pos, cell: not self.bot.vis_mask[pos],
            ignore_blockers=True
        )

        if unseen_pos:
            return GoNextToSubgoal(self.bot, unseen_pos).get_action()

        assert False, "0nothing left to explore"

    def take_action(self, action):
        super(ExploreSubgoal, self).take_action(action)

        # Find the closest unseen position
        _, unseen_pos, _ = self.bot.shortest_path(
            lambda pos, cell: not self.bot.vis_mask[pos]
        )

        if unseen_pos:
            self.bot.stack.pop()
            self.bot.stack.append(GoNextToSubgoal(self.bot, unseen_pos, reason='Explore', blocker=False))
            return False

        if not unseen_pos:
            _, unseen_pos, _ = self.bot.shortest_path(
                lambda pos, cell: not self.bot.vis_mask[pos],
                ignore_blockers=True
            )

        if unseen_pos:
            self.bot.stack.pop()
            self.bot.stack.append(GoNextToSubgoal(self.bot, unseen_pos, reason='Explore', blocker=True))
            return False

        # Find the closest unlocked unopened door
        def unopened_unlocked_door(pos, cell):
            if not cell:
                return False
            if cell.type != 'door':
                return False
            return not cell.is_open and not cell.is_locked

        # Find the closest unopened door
        def unopened_door(pos, cell):
            if not cell:
                return False
            if cell.type != 'door':
                return False
            return not cell.is_open

        # Try to find an unlocked door first
        # We do this because otherwise, opening a locked door as
        # a subgoal may try to open the same door for exploration,
        # resulting in an infinite loop
        _, door_pos, _ = self.bot.shortest_path(unopened_unlocked_door)
        if not door_pos:
            _, door_pos, _ = self.bot.shortest_path(unopened_unlocked_door, ignore_blockers=True)
        if not door_pos:
            _, door_pos, _ = self.bot.shortest_path(unopened_door)
        if not door_pos:
            _, door_pos, _ = self.bot.shortest_path(unopened_door, ignore_blockers=True)
        # Open the door

        if door_pos:
            # door = self.bot.mission.grid.get(*door_pos)
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot))
            self.bot.stack.append(GoNextToSubgoal(self.bot, door_pos))
            return False

        # Find the closest unseen position, ignoring blocking objects
        path, unseen_pos, _ = self.bot.shortest_path(
            lambda pos, cell: not self.bot.vis_mask[pos],
            ignore_blockers=True
        )

        if unseen_pos:
            self.bot.stack.pop()
            self.bot.stack.append(GoNextToSubgoal(self.bot, unseen_pos, reason='Explore', blocker=True))
            return False

        assert False, "1nothing left to explore"


class Bot:
    def __init__(self, mission, timeout=10000):
        # Mission to be solved
        self.mission = mission

        grid_size = mission.grid_size

        # Grid containing what has been mapped out
        self.grid = Grid(grid_size, grid_size)

        # Visibility mask. True for explored/seen, false for unexplored.
        self.vis_mask = np.zeros(shape=(grid_size, grid_size), dtype=np.bool)

        # Number of environment steps
        self.step_count = 0

        # Number of compute iterations performed
        self.itr_count = 0

        # Maximum number of compute iterations to perform
        self.timeout = timeout

        # Stack of tasks/subtasks to complete (tuples)
        self.stack = []

        # Process/parse the instructions
        self.process_instr(mission.instrs)

    def find_obj_pos(self, obj_desc, use_shortest_path=True):
        """
        Find the position of a visible object matching a given description
        Using use_shortest_path=True ensures that while on the way to the object, we won't change out minds and go to
        another similar object
        """

        assert len(obj_desc.obj_set) > 0

        best_distance_to_obj = None
        best_pos = None

        for i in range(len(obj_desc.obj_set)):
            try:
                # obj = obj_desc.obj_set[i]
                obj_pos = obj_desc.obj_poss[i]

                if self.vis_mask[obj_pos]:
                    shortest_path_to_obj = None
                    if use_shortest_path:
                        shortest_path_to_obj, _, _ = self.shortest_path(
                            lambda pos, cell: pos == obj_pos
                        )
                    if shortest_path_to_obj is not None:
                        distance_to_obj = len(shortest_path_to_obj)
                    else:
                        distance_to_obj = self.distance(obj_pos, self.mission.agent_pos)
                    if not best_distance_to_obj or distance_to_obj < best_distance_to_obj:
                        best_distance_to_obj = distance_to_obj
                        best_pos = obj_pos
            except IndexError:
                # Suppose we are tracking red keys, and we just used a red key to open a door,
                # then for the last i, accessing obj_desc.obj_poss[i] will raise an IndexError
                # -> Solution: Not care about that red key we used to open the door
                pass

        return best_pos

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
        top_left = pos + f_vec * (AGENT_VIEW_SIZE - 1) - r_vec * (AGENT_VIEW_SIZE // 2)

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

    def distance(self, pos, target):
        # Function that returns the distance between a position and a target. It is useful to define extra conditions
        # for the shortest_path function
        return np.abs(target[0] - pos[0]) + np.abs(target[1] - pos[1])

    def blocker_fn(self, pos, target, n_steps):
        # Useful to define a function that blockers need to satisfy when looking for shortest path
        # target and pos are tuples or arrays of 2 elements
        return self.distance(pos, target) <= n_steps

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

            # Mark this position as visited
            visited.add((i, j))

            cell = grid.get(i, j)

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), cell):
                next_search_dist = None
                if distance_of_first_blocking_obj_to_target is not None:
                    next_search_dist = distance_of_first_blocking_obj_to_target - 1
                return path, (i, j), next_search_dist

            # If this cell was not visually observed, don't visit neighbors
            if not self.vis_mask[i, j]:
                continue

            # If there is something in this cell
            if cell:
                # If this is a wall, don't visit neighbors
                if cell.type == 'wall':
                    continue
                # If this is a door
                elif cell.type == 'door':
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

            # Visit each neighbor cell
            # for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # TODO: this as in a for loop, that seems useles, it worked... I removed it - make sure it still works ok
            # TODO: I want this to be "for positions that are one action away and that would make the state change (e.g. if position changes or if carrying changes) instead of one cell away. If there are no one action away positions that change the state, check "2 action away things" then "3 action away things", that is the maximum we should tolerate (e.g. left left forward/pickup/drop or right right forward/pickup/drop)
            for k, l in [(di, dj), (dj, di), (- dj, - di), (- di, - dj)]:
                next_pos = (i + k, j + l)
                next_dir_vec = (k, l)
                queue.append((*next_pos, *next_dir_vec, path + [next_pos]))

        # Path not found
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
                nb_pos = (i + k, j + l)
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

        _, drop_pos, _ = self.shortest_path(match_noadj)

        if not drop_pos:
            _, drop_pos, _ = self.shortest_path(match_empty)

        if not drop_pos:
            _, drop_pos, _ = self.shortest_path(match_noadj, ignore_blockers=True)

        if not drop_pos:
            _, drop_pos, _ = self.shortest_path(match_empty, ignore_blockers=True)

        return drop_pos

    def process_instr(self, instr):
        """
        Translate instructions into an internal form the agent can execute
        """

        if isinstance(instr, GoToInstr):
            self.stack.append(GoToObjSubgoal(self, instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(OpenSubgoal(self))
            self.stack.append(GoToObjSubgoal(self, instr.desc))
            return

        if isinstance(instr, PickupInstr):
            # We pick up and immediately drop so
            # that we may carry other objects
            self.stack.append(DropSubgoal(self))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoToObjSubgoal(self, instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(DropSubgoal(self))
            self.stack.append(GoToAdjPosSubgoal(self, instr.desc_fixed))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoToObjSubgoal(self, instr.desc_move))
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

    def get_action(self):
        """
        Produce the optimal action at the current state
        """
        # Process the current observation
        self.process_obs()
        if len(self.stack) == 0:
            return self.mission.actions.done
        subgoal = self.stack[-1]
        subgoal.update_agent_attributes()
        action = subgoal.get_action()
        return action

    def take_action(self, action):
        """
        Update agent's internal state. Should always be called after get_action() and before env.step()
        """

        self.step_count += 1

        finished_updating = False
        while not finished_updating:
            empty_stack = self.empty_stack_update(action)
            if not empty_stack:
                subgoal = self.stack[-1]
                subgoal.update_agent_attributes()
                finished_updating = subgoal.take_action(action)
            else:
                finished_updating = True

    def empty_stack_update(self, action):
        pos = self.mission.agent_pos
        dir_vec = self.mission.dir_vec
        fwd_pos = pos + dir_vec

        if len(self.stack) == 0:
            if action != self.mission.actions.done:
                self.stack.append(GoNextToSubgoal(self, fwd_pos))
            return True

    def step(self):
        action = self.get_action()
        self.take_action(action)
