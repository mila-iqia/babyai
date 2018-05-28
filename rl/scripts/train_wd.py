#!/usr/bin/env python3

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
print(len(demos))
demos.sort(key=len,reverse=True)



default_model_name = "{}_{}_il".format(args.env, args.demos_origin)
model_name = args.model or default_model_name
print("The model is saved in {}".format(model_name))

obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)

# Define logger and Tensorboard writer

logger = utils.Logger(model_name)
if args.tb:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(utils.get_log_path(model_name, ext=False))


# Define actor-critic model

acmodel = utils.load_model(obss_preprocessor.obs_space, env.action_space, model_name,
                            args.model_instr, args.model_mem, args.arch)

acmodel.train()


# Log command, availability of CUDA, and model

logger.log(args, to_print=False)
logger.log("CUDA is {}available".format('' if torch.cuda.is_available() else 'not '))
logger.log(acmodel)


if torch.cuda.is_available():
    acmodel.cuda()

optimizer = torch.optim.Adam(acmodel.parameters(), args.lr, eps=args.optim_eps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_values():
    flat_demos = []
    value_inds = [0]

    for demo in demos:
        flat_demos += demo
        value_inds.append(value_inds[-1] + len(demo))
    flat_demos = np.array(flat_demos)
    value_inds = value_inds[:-1]
    value_inds = [index-1 for index in value_inds][1:] +[len(flat_demos)-1]

    values = np.zeros([len(flat_demos)],dtype=np.float64)

    reward, done = flat_demos[:,2], flat_demos[:,3]

    reward = utils.reshape_reward(None,None,reward,None)

    values[value_inds]= reward[value_inds]
    last_value = values[value_inds]
    
    while True:
        value_inds = [index-1 for index in value_inds]
        if value_inds[0] == -1:
            break
        done_step = done[value_inds]
        value_inds = value_inds[:len(value_inds)-sum(done_step)]
        last_value = last_value[:len(last_value)-sum(done_step)]
       
        values[value_inds]= reward[value_inds] + args.discount*last_value
        last_value = values[value_inds]
    
    flat_demos = [np.append(flat_demos[i],[values[i],]) for i in range(len(flat_demos))]
    new_demos = []
    offset = 0

    for demo in demos:
        new_demos.append(flat_demos[offset:offset+len(demo)])
        offset += len(demo)

    flat_demos = np.array(flat_demos)
    return new_demos,flat_demos

def run_epoch_norecur():
    np.random.shuffle(flat_demos)
    batch_size = args.batch_size

    log = {"entropy": [],"value_loss": [],"policy_loss": []}

    for j in range(0, len(flat_demos), batch_size):
        flat_batch = flat_demos[j:j + batch_size,:]
        obs, action_true, values, done = flat_batch[:,0], flat_batch[:,1], flat_batch[:,4], flat_batch[:,3]
        preprocessed_obs = obss_preprocessor(obs, device=device)

        action_true = torch.tensor([action for action in action_true],device=device,dtype=torch.float)
        values = torch.tensor([value for value in values],device=device,dtype=torch.float)
        memory = torch.zeros([batch_size,acmodel.memory_size], device=device)
        # Compute loss

        dist, value, memory = acmodel(preprocessed_obs,memory)

        entropy = dist.entropy().mean()

        policy_loss = -dist.log_prob(action_true).mean()

        value_loss = (value - values).pow(2).mean()
        loss = policy_loss - args.entropy_coef * entropy + args.value_loss_coef * value_loss

        # Update actor-critic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log["entropy"].append(float(entropy))
        log["policy_loss"].append(float(policy_loss))
        log["value_loss"].append(float(value_loss))

    return log
    


def run_epoch_recurrence():
    np.random.shuffle(demos)
    
    batch_size = args.batch_size
    offset = 0
    assert len(demos) % batch_size == 0
    log = {"entropy": [],"value_loss": [],"policy_loss": []}

    for batch_index in range(len(demos)//batch_size):
        batch = demos[offset:offset+batch_size]
        batch.sort(key=len,reverse=True)

        flat_batch = []
        inds = [0]
        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))
        flat_batch = np.array(flat_batch)
        inds = inds[:-1]


        obss, action_true, values, done = flat_batch[:,0], flat_batch[:,1], flat_batch[:,4], flat_batch[:,3]
        action_true = torch.tensor([action for action in action_true],device=device,dtype=torch.float)        
        values = torch.tensor([value for value in values],device=device,dtype=torch.float)

        
        memories = torch.zeros([len(flat_batch),acmodel.memory_size], device=device)
        memory = torch.zeros([batch_size,acmodel.memory_size], device=device)

        time_step = 0
        time_step_inds = inds
        
        while True:
            obs = obss[time_step_inds]
            done_step = done[time_step_inds]
            preprocessed_obs = obss_preprocessor(obs, device=device)
            with torch.no_grad():
                if args.model_mem:
                    _, _, new_memory = acmodel(preprocessed_obs, memory[:len(time_step_inds),:])    

            if args.model_mem:
                for i in range(len(time_step_inds)):
                    memories[time_step_inds[i],:] = memory[i,:]
                memory[:len(time_step_inds),:] = new_memory
            time_step_inds = time_step_inds[:len(time_step_inds)-sum(done_step)]
            if len(time_step_inds) == 0:
                break
            time_step_inds = [index+1 for index in time_step_inds]


        while True:
            memory = memories[inds,:]
            final_loss = 0
            for i in range(args.recurrence):
                obs = obss[inds]
                preprocessed_obs = obss_preprocessor(obs, device=device)
                action_step = action_true[inds]
                done_step = done[inds]
                if args.model_mem:
                    dist, value, memory = acmodel(preprocessed_obs, memory)

                entropy = dist.entropy().mean()
                policy_loss = -dist.log_prob(action_step).mean()
                value_loss = (value - values[inds]).pow(2).mean()
                loss = policy_loss - args.entropy_coef * entropy + args.value_loss_coef * value_loss
                final_loss += loss

                # Log some values
                log["entropy"].append(float(entropy))
                log["policy_loss"].append(float(policy_loss))
                log["value_loss"].append(float(value_loss))

                inds = inds[:len(inds)-sum(done_step)]
                memory = memory[:len(inds)]
                if len(inds) == 0:
                    break
                inds = [index+1 for index in inds]

            final_loss /= args.recurrence
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if len(inds) == 0:
                break     

        offset += batch_size
        return log

total_start_time = time.time()

demos,flat_demos = calculate_values()

for i in range(1,args.epochs+1): 
    update_start_time = time.time()   
    scheduler.step()
    
    if args.model_mem:
        log = run_epoch_recurrence()
    else:
        log = run_epoch_norecur()

    update_end_time = time.time()

    # Print logs
    total_ellapsed_time = int(time.time() - total_start_time)
    fps = len(flat_demos)/(update_end_time - update_start_time)
    duration = datetime.timedelta(seconds=total_ellapsed_time)

    for key in log:
        log[key] = np.mean(log[key])

    logger.log(
        "U {} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | vL {: .3f}"
            .format(i, fps, duration,
                    log["entropy"], log["policy_loss"],log["value_loss"]))

    if args.tb:
        writer.add_scalar("FPS", fps, i)
        writer.add_scalar("duration", total_ellapsed_time, i)
        writer.add_scalar("entropy", log["entropy"], i)
        writer.add_scalar("policy_loss", log["policy_loss"], i)

    if args.save_interval > 0 and i % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, model_name)
        if torch.cuda.is_available():
            acmodel.cuda()


