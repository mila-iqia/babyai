from abc import ABC, abstractmethod
import torch
from .. import utils
from babyai.bot import Bot
from babyai.model import ACModel
from random import Random


class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results['dist']
            value = model_results['value']
            self.memory = model_results['memory']

        if self.argmax:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()

        return {'action': action,
                'dist': dist,
                'value': value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.
        else:
            self.memory *= (1 - done)


class RandomAgent:
    """A newly initialized model-based agent."""

    def __init__(self, seed=0, number_of_actions=7):
        self.rng = Random(seed)
        self.number_of_actions = number_of_actions

    def act(self, obs):
        action = self.rng.randint(0, self.number_of_actions - 1)
        # To be consistent with how a ModelAgent's output of `act`:
        return {'action': torch.tensor(action),
                'dist': None,
                'value': None}


class DemoAgent(Agent):
    """A demonstration-based agent. This agent behaves using demonstrations."""

    def __init__(self, demos_name, env_name, origin):
        self.demos_path = utils.get_demos_path(demos_name, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
        self.demos = utils.demos.transform_demos(self.demos)
        self.demo_id = 0
        self.step_id = 0

    @staticmethod
    def check_obss_equality(obs1, obs2):
        if not(obs1.keys() == obs2.keys()):
            return False
        for key in obs1.keys():
            if type(obs1[key]) in (str, int):
                if not(obs1[key] == obs2[key]):
                    return False
            else:
                if not (obs1[key] == obs2[key]).all():
                    return False
        return True

    def act(self, obs):
        if self.demo_id >= len(self.demos):
            raise ValueError("No demonstration remaining")
        expected_obs = self.demos[self.demo_id][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        return {'action': self.demos[self.demo_id][self.step_id][1]}

    def analyze_feedback(self, reward, done):
        self.step_id += 1

        if done:
            self.demo_id += 1
            self.step_id = 0


class BotAgent:
    def __init__(self, env):
        """An agent based on a GOFAI bot."""
        self.env = env
        self.on_reset()

    def on_reset(self):
        self.bot = Bot(self.env)

    def act(self, obs=None, update_internal_state=True, *args, **kwargs):
        action = self.bot.replan()
        return {'action': action}

    def analyze_feedback(self, reward, done):
        pass


class HandCraftedMetacontroller:
    def __init__(self, env):
        """Obtain a metacontroller policy from a GOFAI bot."""
        self.env = env
        self.on_reset()

    def get_subgoal(env):
        'return baby-language subgoal instruction from a bot'
        metacontroller = BotAgent(env)
        stack = metacontroller.bot.stack
        subgoal = stack.pop()
        return metacontroller.bot._produce_instruction(subgoal)

    def get_action(env, obs):
        'get next action from a bot'
        metacontroller = BotAgent(env)
        return metacontroller.act(obs)['action']

    def int_to_action_name(actionInt):
        'convert action int into a printable string description'
        int_to_action = {
            0: 'left',
            1: 'right',
            2: 'forward',
            3: 'pickup',
            4: 'drop',
            5: 'toggle/open',
            6: 'done'
        }
        return int_to_action[actionInt]


def load_agent(env, model_name, demos_name=None, demos_origin=None, argmax=True, env_name=None):
    # env_name needs to be specified for demo agents
    if model_name == 'BOT':
        return BotAgent(env)
    elif model_name is not None:
        obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)
        return ModelAgent(model_name, obss_preprocessor, argmax)
    elif demos_origin is not None or demos_name is not None:
        return DemoAgent(demos_name=demos_name, env_name=env_name, origin=demos_origin)
