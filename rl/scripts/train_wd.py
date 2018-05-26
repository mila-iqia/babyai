#!/usr/bin/env python3
# TODO add support for recurrent policy

import argparse
import gym
import levels
import time
import datetime
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch_rl
from torch_rl.utils import DictList


import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ORIGIN_il)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of demonstrations to use (default: 100)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs (default: 10)")
parser.add_argument("--batch-size", type=int, default=10,
                    help="batch size (default: 10)")
parser.add_argument("--model-instr", action="store_true", default=False,
                    help="use instructions in the model")
parser.add_argument("--model-mem", action="store_true", default=False,
                    help="use memory in the model")
parser.add_argument("--arch", default='cnn1',
                    help="image embedding architecture")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")


args = parser.parse_args()


utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)


demos = utils.load_demos(args.env, args.demos_origin)[:args.episodes]

obss_preprocessor = utils.ObssPreprocessor(args.model, env.observation_space)


# Define actor-critic model

acmodel = utils.load_model(obss_preprocessor.obs_space, env.action_space, args.model,
                            args.model_instr, args.model_mem, args.arch)


if torch.cuda.is_available():
    acmodel.cuda()

optimizer = torch.optim.Adam(acmodel.parameters(), args.lr, eps=args.optim_eps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# flat_demos = []
# for demo in demos:
#     flat_demos += demo

def starting_indexes(max_len):
    return np.arange(0, max_len*args.batch_size, args.recurrence)


def collect_experiences():
    batch_size = args.batch_size
    offset = 0
    assert len(demos) % batch_size == 0
    # demos.sort(key=len)

    for batch_index in range(len(demos)//batch_size):
        batch = demos[offset:offset+batch_size]
        max_len = max(len(item) for item in batch)
        obss = []

        reward_mat = np.zeros((batch_size,max_len),dtype=np.float64)
        action_true = np.zeros((batch_size,max_len),dtype=np.float64)
        mask_mat = np.zeros((batch_size,max_len),dtype = np.float64)
        for i in range(batch_size):
            temp = []
            for index in range(max_len):
                j = min(index,len(batch[i])-1)
                temp.append(batch[i][j][0])
                reward_mat[i,index] = (index<len(batch[i]))*batch[i][j][2]
                action_true[i,index] = (index<len(batch[i]))*batch[i][j][1]
                mask_mat[i,index] = (index<len(batch[i]))
            obss.append(temp)

        reward_mat = utils.reshape_reward(None,None,reward_mat,None)

        obss = np.stack(obss,axis=0)

        if args.model_mem:
            memories = torch.zeros([batch_size,max_len,acmodel.memory_size], device=device)
            memory = torch.zeros([batch_size,acmodel.memory_size], device=device)


        for i in range(max_len):
            preprocessed_obs = obss_preprocessor(obss[:,i], device=device)
            with torch.no_grad():
                if args.model_mem:
                    dist, _, new_memory = acmodel(preprocessed_obs, memory)
                else:
                    dist, _ = acmodel(preprocessed_obs)        

            if args.model_mem:
                memories[:,i] = memory
                memory = new_memory
        
        values = np.zeros((batch_size,max_len),dtype=np.float64)
        values[:,max_len-1] = reward_mat[:,max_len-1]

        for i in reversed(range(max_len-1)):
            values[:,i] = reward_mat[:,i] + args.discount*values[:,i+1]

        action_true = torch.tensor(action_true,dtype=torch.float,device=device)
        reward_mat = torch.tensor(reward_mat,dtype=torch.float,device=device)
        values = torch.tensor(values,dtype=torch.float,device=device)
        mask_mat = torch.tensor(mask_mat,dtype=torch.float,device=device)

        exps = DictList()
        exps.obs = [obs for temp in obss for obs in temp]
        if args.model_mem:
            exps.memory = memories.view(-1, *memories.shape[2:])
            exps.mask = mask_mat.view(-1, *mask_mat.shape[2:])
        exps.action_true = action_true.view(-1, *action_true.shape[2:])
        exps.returnn = values.view(-1, *values.shape[2:])


        exps.obs = obss_preprocessor(exps.obs, device=device)
        inds = starting_indexes(max_len)
        
        if args.model_mem:
            memory = exps.memory[inds]

        update_entropy = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        for _ in range(args.recurrence):
            # Create a sub-batch of experience
            sb = exps[inds]

            # Compute loss

            if args.model_mem:
                dist, value, memory = acmodel(sb.obs, memory)
            else:
                dist, value = acmodel(sb.obs)

            entropy = (dist.entropy()*sb.mask).mean()


            policy_loss = -(dist.log_prob(sb.action_true)*sb.mask).mean()

            value_loss = ((value - sb.returnn)*sb.mask).pow(2).mean()
            
            loss = policy_loss - args.entropy_coef * entropy + args.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

            # Update indexes

            inds += 1

        optimizer.zero_grad()
        update_loss.backward()
        optimizer.step()

        print("Batch %d : " % (batch_index+1),end="\t")
        print("Entropy %.3f" % update_entropy,end="\t")
        print("Value Loss %.3f" % update_value_loss,end="\t")
        print("Policy Loss %.3f" % update_policy_loss,end="\t")
        print("Total Loss %.3f" % update_loss)


        offset += batch_size


for i in range(args.epochs):
    print("Epoch %d" % (i+1))
    collect_experiences()

    if args.save_interval > 0 and i % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, args.model)
        if torch.cuda.is_available():
            acmodel.cuda()


