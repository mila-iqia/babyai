import torch
from torch_rl import RecurrentACModel

import utils

class Agent:
    def __init__(self, model_name, observation_space, action_space):
        self.obss_preprocessor = utils.ObssPreprocessor(model_name, observation_space)
        self.acmodel = utils.load_model(self.obss_preprocessor.obs_space, action_space, model_name)

        self.is_recurrent = isinstance(self.acmodel, RecurrentACModel)
        if self.is_recurrent:
            self._initialize_memory()
    
    def _initialize_memory(self):
        self.memory = torch.zeros(1, self.acmodel.memory_size)

    def get_action(self, obs, deterministic=False):
        preprocessed_obs = self.obss_preprocessor([obs])

        with torch.no_grad():
            if self.is_recurrent:
                dist, _, self.memory = self.acmodel(preprocessed_obs, self.memory)
            else:
                dist, _ = self.acmodel(preprocessed_obs)
        
        if deterministic:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()
        
        return action.item()
    
    def analyze_feedback(self, reward, done):
        if done and self.is_recurrent:
            self._initialize_memory()