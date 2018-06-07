#!/usr/bin/env python3

import argparse
import gym
from babyai import levels
import time
import datetime
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch_rl
from torch_rl.utils import DictList
from scripts.evaluate import evaluate


import utils

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos-origin", required=True,
                    help="origin of the demonstrations: human | agent (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the model (default: ENV_ORIGIN_il)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of demonstrations to use (default: 100)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
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
                    help="batch size (In case of memory, the batch size is the number of demos, otherwise, it is the number of frames)(default: 10)")
parser.add_argument("--instr-model", default=None,
                    help="model to encode instructions, None if not using instructions, possible values: gru, conv, bow")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='cnn1',
                    help="image embedding architecture")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--validation-interval", type=int, default=10,
                    help="number of epochs between two validation checks (default: 10)")
parser.add_argument("--val-episodes", type=int, default=100,
                    help="number of episodes used for validation (default: 100)")
parser.add_argument("--patience", type=int, default=3,
                    help="patience for early stopping (default: 3)")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")


args = parser.parse_args()
utils.seed(args.seed)
env = gym.make(args.env)

train_demos = utils.load_demos(args.env, args.demos_origin)[:args.episodes]
train_demos.sort(key=len,reverse=True)



val_demos = utils.load_demos(args.env, args.demos_origin+"_valid")


default_model_name = "{}_{}_il".format(args.env, args.demos_origin)
model_name = args.model or default_model_name
print("The model is saved in {}".format(model_name))

obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)

# Define logger and Tensorboard writer
logger = utils.get_logger(model_name)
if args.tb:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(utils.get_log_dir(model_name))


# Define actor-critic model
acmodel = utils.load_model(model_name, raise_not_found=False)
if acmodel is None:
    acmodel = ACModel(obss_preprocessor.obs_space, env.action_space,
                      not args.no_instr, not args.no_mem, args.arch)
acmodel.train()


# Log command, availability of CUDA, and model
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)


if torch.cuda.is_available():
    acmodel.cuda()

optimizer = torch.optim.Adam(acmodel.parameters(), args.lr, eps=args.optim_eps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_values(demos):
    flat_demos = []
    value_inds = [0]

    for demo in demos:
        flat_demos += demo
        value_inds.append(value_inds[-1] + len(demo))

    # Value inds are pointing at the last episode of each of the observation
    flat_demos = np.array(flat_demos)
    value_inds = value_inds[:-1]
    value_inds = [index-1 for index in value_inds][1:] +[len(flat_demos)-1]

    # Value array with the length of flat_demos
    values = np.zeros([len(flat_demos)],dtype=np.float64)

    reward, done = flat_demos[:,2], flat_demos[:,3]

    # Reshaping the reward
    reward = utils.reshape_reward(None,None,reward,None)

    # Value for last episodes = reward at last episodes
    values[value_inds]= reward[value_inds]
    # last value keeps the values corresponding to last visited states
    last_value = values[value_inds]
    

    while True:
        value_inds = [index-1 for index in value_inds]
        if value_inds[0] == -1:
            break
        done_step = done[value_inds]
        # Removing indices of finished episodes
        value_inds = value_inds[:len(value_inds)-sum(done_step)]
        last_value = last_value[:len(last_value)-sum(done_step)]
        
        # Calculating value of the states using value of previous states 
        values[value_inds]= reward[value_inds] + args.discount*last_value
        last_value = values[value_inds]
    
    # Appending values to corresponding demos
    flat_demos = [np.append(flat_demos[i],[values[i],]) for i in range(len(flat_demos))]
    new_demos = []
    offset = 0

    # Reconstructing demos from flat_demos
    for demo in demos:
        new_demos.append(flat_demos[offset:offset+len(demo)])
        offset += len(demo)

    flat_demos = np.array(flat_demos)
    return new_demos,flat_demos

def run_epoch_norecur(flat_demos, is_training=False):
    np.random.shuffle(flat_demos)
    batch_size = args.batch_size

    log = {"entropy": [],"value_loss": [],"policy_loss": [], "accuracy": []}
    offset = 0
    for j in range(len(flat_demos)//batch_size):
        flat_batch = flat_demos[offset:offset+batch_size,:]
        obs, action_true, values, done = flat_batch[:,0], flat_batch[:,1], flat_batch[:,4], flat_batch[:,3]
        preprocessed_obs = obss_preprocessor(obs, device=device)

        action_true = torch.tensor([action for action in action_true],device=device,dtype=torch.long)
        values = torch.tensor([value for value in values],device=device,dtype=torch.float)
        memory = torch.zeros([batch_size,acmodel.memory_size], device=device)
        
        # Compute loss
        dist, value, memory = acmodel(preprocessed_obs,memory)

        entropy = dist.entropy().mean()

        policy_loss = -dist.log_prob(action_true).mean()

        value_loss = (value - values).pow(2).mean()
        action_pred = dist.probs.max(1, keepdim=True)[1]
        accuracy = float((action_pred == action_true.unsqueeze(1)).sum())/batch_size

        loss = policy_loss - args.entropy_coef * entropy + args.value_loss_coef * value_loss

        # Update actor-critic
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log["entropy"].append(float(entropy))
        log["policy_loss"].append(float(policy_loss))
        log["value_loss"].append(float(value_loss))
        log["accuracy"].append(float(accuracy))
        offset += batch_size

    return log
    
def starting_indexes(num_frames):
    if num_frames % args.recurrence == 0:
        return np.arange(0, num_frames, args.recurrence)
    else:
        return np.arange(0, num_frames, args.recurrence)[:-1]

def run_epoch_recurrence(demos, is_training=False):
    np.random.shuffle(demos)
    batch_size = args.batch_size
    offset = 0
    assert len(demos) % batch_size == 0
    
    # Log dictionary
    log = {"entropy": [],"value_loss": [],"policy_loss": [],"accuracy" : []}

    for batch_index in range(len(demos)//batch_size):
        batch = demos[offset:offset+batch_size]
        batch.sort(key=len,reverse=True)

        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]
        
        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        flat_batch = np.array(flat_batch)
        inds = inds[:-1]
        num_frames = len(flat_batch)
        
        mask = np.ones([len(flat_batch)],dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask,device=device,dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, values, done = flat_batch[:,0], flat_batch[:,1], flat_batch[:,4], flat_batch[:,3]
        action_true = torch.tensor([action for action in action_true],device=device,dtype=torch.long)        
        values = torch.tensor([value for value in values],device=device,dtype=torch.float)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch),acmodel.memory_size], device=device)
        memory = torch.zeros([batch_size,acmodel.memory_size], device=device)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = obss_preprocessor(obs, device=device)
            with torch.no_grad():
                # taking the memory till the length of time_step_inds, as demos beyond that have already finished
                _, _, new_memory = acmodel(preprocessed_obs, memory[:len(inds),:])    

            for i in range(len(inds)):
                # Copying to the memories at the corresponding locations
                memories[inds[i],:] = memory[i,:]
            
            memory[:len(inds),:] = new_memory
            
            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds)-sum(done_step)]
            if len(inds) == 0:
                break
            # Incrementing the remaining indices
            inds = [index+1 for index in inds]

        # Here, actual backprop upto args.recurrence happens
        final_loss = 0
        final_entropy, final_policy_loss, final_value_loss = 0,0,0

        indexes = starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0

        for _ in range(args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = obss_preprocessor(obs, device=device)
            action_step = action_true[indexes]
            mask_step = mask[indexes]
            dist, value, memory = acmodel(preprocessed_obs, memory*mask_step)
            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            value_loss = (value - values[indexes]).pow(2).mean()
            loss = policy_loss - args.entropy_coef * entropy + args.value_loss_coef * value_loss
            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum())/len(flat_batch)
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            final_value_loss += value_loss
            indexes += 1

        final_loss /= args.recurrence
        log["entropy"].append(float(final_entropy/args.recurrence))
        log["policy_loss"].append(float(final_policy_loss/args.recurrence))
        log["value_loss"].append(float(final_value_loss/args.recurrence))
        log["accuracy"].append(float(accuracy))
        
        if is_training:
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
                 
        offset += batch_size
    
    return log


# Validation based on the mean reward on demos
def validate():
    # Seed needs to be reset for each validation, to ensure consistency 
    env.seed(args.val_seed)
    utils.seed(args.val_seed)
    print("Validating the model")
    args.deterministic = False
    agent = utils.load_agent(args, env)
    returnn = 0
    # Setting the agent model to the current model
    agent.model = acmodel
    
    logs = evaluate(agent,env,args.val_episodes)
    return np.mean(logs["return_per_episode"])

total_start_time = time.time()

# calculating values for train_demos and val_demos
train_demos,flat_train_demos = calculate_values(train_demos)
val_demos,flat_val_demos = calculate_values(val_demos)

# best mean return to keep track of performance on validation set
best_mean_return = 0
patience = 0
i = 0

# Model saved initially to avoid "Model not found Exception" during first validation step 
utils.save_model(acmodel, model_name)

while True: 
    i += 1
    update_start_time = time.time()   
    
    # Learning rate scheduler
    scheduler.step()

    if not(args.no_mem):
        log = run_epoch_recurrence(train_demos, is_training=True)
    else:
        log = run_epoch_norecur(flat_train_demos, is_training=True)

    update_end_time = time.time()

    # Print logs
    total_ellapsed_time = int(time.time() - total_start_time)
    fps = len(flat_train_demos)/(update_end_time - update_start_time)
    duration = datetime.timedelta(seconds=total_ellapsed_time)

    for key in log:
        log[key] = np.mean(log[key])

    logger.info(
        "U {} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | vL {: .3f} | A {: .3f}"
            .format(i, fps, duration,
                    log["entropy"], log["policy_loss"],log["value_loss"], log["accuracy"]))

    if not(args.no_mem):
        val_log = run_epoch_recurrence(val_demos)
    else:
        val_log = run_epoch_norecur(flat_val_demos)

    for key in val_log:
        val_log[key] = np.mean(val_log[key])

    if args.tb:
        writer.add_scalar("FPS", fps, i)
        writer.add_scalar("duration", total_ellapsed_time, i)
        writer.add_scalar("entropy", log["entropy"], i)
        writer.add_scalar("policy_loss", log["policy_loss"], i)
        writer.add_scalar("accuracy", log["accuracy"], i)
        writer.add_scalar("validation_accuracy",val_log["accuracy"], i)


    if i % args.validation_interval == 0:
        if torch.cuda.is_available():
            acmodel.cpu()
        mean_return = validate()
        print("Mean Validation Return %.3f" % mean_return)

        if mean_return > best_mean_return:
            best_mean_return = mean_return
            patience = 0
            # Saving the model
            print("Saving best model")
            obss_preprocessor.vocab.save()
            if torch.cuda.is_available():
                acmodel.cpu()
            utils.save_model(acmodel, model_name)
            if torch.cuda.is_available():
                acmodel.cuda()
        else:
            print("Losing Patience")
            patience += 1
            if patience > args.patience:
                break
            if torch.cuda.is_available():
                acmodel.cuda()


