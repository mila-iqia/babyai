from abc import ABC, abstractmethod
import torch

from .. import utils


class Agent(ABC):
    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    def __init__(self, model_name, observation_space, deterministic):
        self.obss_preprocessor = utils.ObssPreprocessor(model_name, observation_space)
        self.model = utils.load_model(model_name)
        self.deterministic = deterministic

        if self.model.recurrent:
            self._initialize_memory()

    def _initialize_memory(self):
        self.memory = torch.zeros(1, self.model.memory_size)

    def get_action(self, obs):
        if obs is None: return None
        preprocessed_obs = self.obss_preprocessor([obs])

        with torch.no_grad():
            if self.model.recurrent:
                dist, _, self.memory = self.model(preprocessed_obs, self.memory)
            else:
                dist, _ = self.model(preprocessed_obs)

        if self.deterministic:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()

        return action.item()

    def analyze_feedback(self, reward, done):
        if done and self.model.recurrent:
            self._initialize_memory()


class DemoAgent(Agent):
    def __init__(self, env_name, origin):
        self.demos = utils.load_demos(env_name, origin)
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

    def get_action(self, obs):
        if self.demo_id >= len(self.demos):
            raise ValueError("No demonstration remaining")

        expected_obs = self.demos[self.demo_id][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        return self.demos[self.demo_id][self.step_id][1]

    def analyze_feedback(self, reward, done):
        self.step_id += 1

        if done:
            self.demo_id += 1
            self.step_id = 0


def load_agent(args, env):
    if args.model is not None:
        return ModelAgent(args.model, env.observation_space, args.deterministic)
    elif args.demos_origin is not None:
        return DemoAgent(args.env, args.demos_origin)
