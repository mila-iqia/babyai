from abc import ABC, abstractmethod
import torch
import numpy as np
from .. import utils
from babyai.agents.bot import Bot


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

    def __init__(self, model_name, obss_preprocessor, argmax):
        self.obss_preprocessor = obss_preprocessor
        self.model = utils.load_model(model_name)
        self.argmax = argmax

        if self.model.recurrent:
            self._initialize_memory()

    def _initialize_memory(self):
        self.memory = torch.zeros(1, self.model.memory_size)

    def act(self, obs):
        preprocessed_obs = self.obss_preprocessor([obs])

        with torch.no_grad():
            if self.model.recurrent:
                dist, value, self.memory = self.model(preprocessed_obs, self.memory)
            else:
                dist, value = self.model(preprocessed_obs)

        if self.argmax:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()

        return {'action': action,
                'dist': dist,
                'value': value}

    def analyze_feedback(self, reward, done):
        if done and self.model.recurrent:
            self._initialize_memory()


class DemoAgent(Agent):
    """A demonstration-based agent. This agent behaves using demonstrations."""

    def __init__(self, demos, env_name, origin):
        self.demos_path = utils.get_demos_path(demos, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
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
        expected_obs = self.demos[self.demo_id][0][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        return {'action': self.demos[self.demo_id][0][self.step_id][1]}

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

    def act(self, *args, **kwargs):
        return {'action': self.bot.step()}

    def analyze_feedback(self, reward, done):
        pass


def load_agent(args, env):
    if args.model == 'BOT':
        return BotAgent(env)
    elif args.model is not None:
        obss_preprocessor = utils.ObssPreprocessor(args.model, env.observation_space)
        return ModelAgent(args.model, obss_preprocessor, args.argmax)
    elif args.demos_origin is not None or args.demos is not None:
        return DemoAgent(demos=args.demos, env_name=args.env, origin=args.demos_origin)
