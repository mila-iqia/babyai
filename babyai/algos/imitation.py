import copy
import gym
import time
import datetime
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch_rl
from torch_rl.utils import DictList
from scripts.evaluate import evaluate


import babyai.utils as utils
from babyai.model import ACModel


class ImitationLearning(object):
    def __init__(self, args):
        self.args = args

        utils.seed(self.args.seed)
        self.env = gym.make(self.args.env)

        self.train_demos = utils.load_demos(self.args.env, self.args.demos_origin)[:self.args.episodes]

        self.val_demos = utils.load_demos(self.args.env, self.args.demos_origin+"_valid")[:500]

        default_model_name = "{}_{}_il".format(self.args.env, self.args.demos_origin)
        self.model_name = self.args.model or default_model_name
        print("The model is saved in {}".format(self.model_name))
        self.args.model = self.model_name

        self.obss_preprocessor = utils.ObssPreprocessor(self.model_name, self.env.observation_space)

        # Define actor-critic model
        self.acmodel = utils.load_model(self.model_name, raise_not_found=False)
        if self.acmodel is None:
            self.acmodel = ACModel(self.obss_preprocessor.obs_space, self.env.action_space,
                                not self.args.no_instr, self.args.instr_arch, not self.args.no_mem, self.args.arch)
        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flat_train_demos, self.flat_val_demos = [], []
        
        for demo in self.train_demos:
            self.flat_train_demos.append(demo)
        for demo in self.val_demos:
            self.flat_val_demos.append(demo)
        self.flat_train_demos = np.array(self.flat_train_demos)
        self.flat_val_demos = np.array(self.flat_val_demos)
        


    def run_epoch_norecur(self, flat_demos, is_training=False):
        flat_demos_t = copy.deepcopy(flat_demos)
        if is_training:
            np.random.shuffle(flat_demos_t)
        batch_size = self.args.batch_size

        log = {"entropy": [],"policy_loss": [], "accuracy": []}
        offset = 0
        for j in range(len(flat_demos_t)//batch_size):
            flat_batch = flat_demos_t[offset:offset+batch_size,:]
            obs, action_true, done = flat_batch[:,0], flat_batch[:,1], flat_batch[:,3]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)

            action_true = torch.tensor([action for action in action_true],device=self.device,dtype=torch.long)
            memory = torch.zeros([batch_size,self.acmodel.memory_size], device=self.device)
            
            # Compute loss
            dist, _, memory = self.acmodel(preprocessed_obs,memory)

            entropy = dist.entropy().mean()

            policy_loss = -dist.log_prob(action_true).mean()

            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy = float((action_pred == action_true.unsqueeze(1)).sum())/batch_size

            loss = policy_loss - self.args.entropy_coef * entropy

            # Update actor-critic
            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            log["entropy"].append(float(entropy))
            log["policy_loss"].append(float(policy_loss))
            log["accuracy"].append(float(accuracy))
            offset += batch_size

        return log
    
    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def run_epoch_recurrence(self, demos, is_training=False):
        demos_t = copy.deepcopy(demos)
        if is_training:
            np.random.shuffle(demos_t)
        batch_size = self.args.batch_size
        offset = 0
        
        # Log dictionary
        log = {"entropy": [],"policy_loss": [],"accuracy" : []}

        for batch_index in range(len(demos_t)//batch_size):
            batch = demos_t[offset:offset+batch_size]
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
            mask = torch.tensor(mask,device=self.device,dtype=torch.float).unsqueeze(1)

            # Observations, true action, values and done for each of the stored demostration
            obss, action_true, done = flat_batch[:,0], flat_batch[:,1], flat_batch[:,3]
            action_true = torch.tensor([action for action in action_true],device=self.device,dtype=torch.long)        

            # Memory to be stored
            memories = torch.zeros([len(flat_batch),self.acmodel.memory_size], device=self.device)
            memory = torch.zeros([batch_size,self.acmodel.memory_size], device=self.device)

            # Loop terminates when every observation in the flat_batch has been handled
            while True:
                # taking observations and done located at inds
                obs = obss[inds]
                done_step = done[inds]
                preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
                with torch.no_grad():
                    # taking the memory till the length of time_step_inds, as demos beyond that have already finished
                    _, _, new_memory = self.acmodel(preprocessed_obs, memory[:len(inds),:])    

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

            indexes = self.starting_indexes(num_frames)
            memory = memories[indexes]
            accuracy = 0

            for _ in range(self.args.recurrence):
                obs = obss[indexes]
                preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
                action_step = action_true[indexes]
                mask_step = mask[indexes]
                dist, value, memory = self.acmodel(preprocessed_obs, memory*mask_step)
                entropy = dist.entropy().mean()
                policy_loss = -dist.log_prob(action_step).mean()
                loss = policy_loss - self.args.entropy_coef * entropy
                action_pred = dist.probs.max(1, keepdim=True)[1]
                accuracy += float((action_pred == action_step.unsqueeze(1)).sum())/len(flat_batch)
                final_loss += loss
                final_entropy += entropy
                final_policy_loss += policy_loss
                indexes += 1

            final_loss /= self.args.recurrence
            log["entropy"].append(float(final_entropy/self.args.recurrence))
            log["policy_loss"].append(float(final_policy_loss/self.args.recurrence))
            log["accuracy"].append(float(accuracy))
            
            if is_training:
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
                     
            offset += batch_size
        
        return log

    def validate(self):
        # Seed needs to be reset for each validation, to ensure consistency 
        self.env.seed(self.args.val_seed)
        utils.seed(self.args.val_seed)

        print("Validating the model")
        self.args.deterministic = False
        agent = utils.load_agent(self.args, self.env)
        # Setting the agent model to the current model
        agent.model = self.acmodel
        
        returnn = 0
        
        logs = evaluate(agent,self.env,self.args.val_episodes)
        
        return np.mean(logs["return_per_episode"])

    def train(self, train_demos, logger, writer):
        # Model saved initially to avoid "Model not found Exception" during first validation step 
        utils.save_model(self.acmodel, self.model_name)

        # best mean return to keep track of performance on validation set
        best_mean_return, patience, i = 0, 0, 0
        total_start_time = time.time()

        while True: 
            i += 1
            update_start_time = time.time()   
            
            # Learning rate scheduler
            self.scheduler.step()

            if not(self.args.no_mem):
                log = self.run_epoch_recurrence(train_demos, is_training=True)
                total_len = sum([len(item) for item in train_demos])
            else:
                log = self.run_epoch_norecur(train_demos, is_training=True)
                total_len = len(train_demos)

            update_end_time = time.time()

            # Print logs
            total_ellapsed_time = int(time.time() - total_start_time)

            fps = total_len/(update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)

            for key in log:
                log[key] = np.mean(log[key])

            logger.info(
                "U {} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}"
                    .format(i, fps, duration,
                            log["entropy"], log["policy_loss"], log["accuracy"]))

            if not(self.args.no_mem):
                val_log = self.run_epoch_recurrence(self.val_demos)
            else:
                val_log = self.run_epoch_norecur(self.flat_val_demos)

            for key in val_log:
                val_log[key] = np.mean(val_log[key])

            if self.args.tb:
                writer.add_scalar("FPS", fps, i)
                writer.add_scalar("duration", total_ellapsed_time, i)
                writer.add_scalar("entropy", log["entropy"], i)
                writer.add_scalar("policy_loss", log["policy_loss"], i)
                writer.add_scalar("accuracy", log["accuracy"], i)
                writer.add_scalar("validation_accuracy",val_log["accuracy"], i)


            if i % self.args.validation_interval == 0:
                if torch.cuda.is_available():
                    self.acmodel.cpu()
                mean_return = self.validate()
                print("Mean Validation Return %.3f" % mean_return)

                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    patience = 0
                    # Saving the model
                    print("Saving best model")
                    self.obss_preprocessor.vocab.save()
                    if torch.cuda.is_available():
                        self.acmodel.cpu()
                    utils.save_model(self.acmodel, self.model_name)
                    if torch.cuda.is_available():
                        self.acmodel.cuda()
                else:
                    print("Losing Patience")
                    patience += 1
                    if patience > self.args.patience:
                        break
                    if torch.cuda.is_available():
                        self.acmodel.cuda()

