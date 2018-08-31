import gym
from babyai.agents.bot import Bot
from gym_minigrid.minigrid import OBJECT_TO_IDX, Grid
from .verifier import *


def wrap_env(env, aux_info):
    '''
    helper function that callss the defined wrappers depending on the what information is required
    '''
    if 'seen_state' in aux_info:
        env = SeenStateWrapper(env)
    if 'visit_proportion' in aux_info:
        env = VisitProportionWrapper(env)
    if 'see_door' in aux_info:
        env = SeeDoorWrapper(env)
    if 'see_obj' in aux_info:
        env = SeeObjWrapper(env)
    if 'in_front_of_what' in aux_info:
        env = InForntOfWhatWrapper(env)
    if 'obj_in_instr' in aux_info:
        env = ObjInInstrWrapper(env)
    if 'bot_action' in aux_info:
        env = BotActionWrapper(env)
    return env


class SeenStateWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to whether
    the new state is already visited or not
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Define a set of seen states. A state is represent by a tuple ((x, y), direction)
        self.seen_states = set()

        # Append the current state to the seen states
        # The state is defined in the reset function of the MiniGridEnv class
        self.seen_states.add((tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir))

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if (tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir) in self.seen_states:
            seen_state = True
        else:
            self.seen_states.add((tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir))
            seen_state = False

        info['seen_state'] = seen_state

        return obs, reward, done, info


class VisitProportionWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to the number of times
    the new state has been visited before, divided by the total number of steps
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Define a dict of seen states and number of times seen. A state is represent by a tuple ((x, y), direction)
        self.seen_states_dict = dict()

        # Append the current state to the seen states
        # The state is defined in the reset function of the MiniGridEnv class
        self.seen_states_dict[(tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir)] = 1

        # Instantiate a counter of total steps
        self.total_steps = 0

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.total_steps += 1
        if (tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir) in self.seen_states_dict:
            self.seen_states_dict[(tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir)] += 1
        else:
            self.seen_states_dict[(tuple(self.env.unwrapped.agent_pos), self.env.unwrapped.agent_dir)] = 1

        info['visit_proportion'] = ((self.seen_states_dict[(tuple(self.env.unwrapped.agent_pos),
                                                            self.env.unwrapped.agent_dir)]
                                     - 1) / self.total_steps)

        return obs, reward, done, info


class SeeDoorWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to whether
    the current observation contains a door, locked or not
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['see_door'] = (None, 'door') in Grid.decode(obs['image'])
        return obs, reward, done, info


class SeeObjWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to whether
    the current observation contains a key, ball, or box
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['see_obj'] = any([obj in Grid.decode(obs['image']) for obj in
                               ((None, 'key'), (None, 'ball'), (None, 'box'))
                               ])
        return obs, reward, done, info


class InForntOfWhatWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to which of
    empty cell/wall/door/key/box/ball is in the cell right in front of the agent
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        cell_in_front = self.env.unwrapped.grid.get(*self.env.unwrapped.front_pos)
        info['in_front_of_what'] = OBJECT_TO_IDX[cell_in_front.type] if cell_in_front else 0  # int 0--8
        return obs, reward, done, info


class ObjInInstrWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to whether an object
    described in the instruction appears in the current observation
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def obj_in_mission(self, instr):
        if isinstance(instr, PutNextInstr):
            return [(instr.desc_fixed.color, instr.desc_fixed.type),
                    (instr.desc_move.color, instr.desc_move.type)]
        if isinstance(instr, SeqInstr):
            return self.obj_in_mission(instr.instr_a) + self.obj_in_mission(instr.instr_b)
        else:
            return [(instr.desc.color, instr.desc.type)]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['obj_in_instr'] = any([obj in Grid.decode(obs['image'])
                                    for obj in self.obj_in_mission(self.env.unwrapped.instrs)])
        return obs, reward, done, info


class BotActionWrapper(gym.Wrapper):
    '''
    Wrapper that adds an entry to the info dic of the step function's output that corresponds to whether
    the action taken corresponds to the action the GOFAI bot would have taken
    '''

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.expert = Bot(self.env.unwrapped)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        try:
            expert_action = self.expert.step()
        except:
            expert_action = None

        info['bot_action'] = action == expert_action

        return obs, reward, done, info
