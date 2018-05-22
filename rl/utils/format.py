import os
import json
import numpy
import re
import torch
import torch_rl

import utils

def get_vocab_path(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name, "vocab.json")

class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        self.vocab = {}
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))

    def __getitem__(self, token):
        if not(token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1        
        return self.vocab[token]

    def save(self):
        utils.create_folders_if_necessary(self.path)
        json.dump(self.vocab, open(self.path, "w"))

class ObssPreprocessor:
    def __init__(self, model_name, obs_space, normalize=False):
        self.normalize = normalize
        self.vocab = Vocabulary(model_name)
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = torch_rl.DictList()

        if "image" in self.obs_space.keys():
            images = numpy.array([obs["image"] for obs in obss])
            if self.normalize:
                images /= 10
            images = torch.tensor(images, device=device, dtype=torch.float)

            obs_.image = images

        if "instr" in self.obs_space.keys():
            raw_instrs = []
            max_instr_len = 0
            
            for obs in obss:
                tokens = re.findall("([a-z]+)", obs["mission"].lower())
                instr = numpy.array([self.vocab[token] for token in tokens])
                raw_instrs.append(instr)
                max_instr_len = max(len(instr), max_instr_len)
            
            instrs = numpy.zeros((len(obss), max_instr_len))

            for i, instr in enumerate(raw_instrs):
                instrs[i, :len(instr)] = instr
            
            instrs = torch.tensor(instrs, device=device, dtype=torch.long)
            
            obs_.instr = instrs

        return obs_

def reshape_reward(obs, action, reward, done):
    return 5*reward