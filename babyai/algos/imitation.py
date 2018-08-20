import copy
import gym
import time
import datetime
import numpy as np
import sys
import torch
import torch.nn.functional as F
from babyai.evaluate import evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import os
import json


class ImitationLearning(object):
    def __init__(self, args):
        self.args = args

        utils.seed(self.args.seed)

        if type(self.args.env) == list:
            self.env = [gym.make(item[0]) for item in self.args.env]

            self.train_demos = [utils.load_demos(utils.get_demos_path(env=env, origin=demos_origin))[:episodes]
                                for env, demos_origin, episodes in self.args.env]

            self.train_demos = [[demo[0] for demo in demos] for demos in self.train_demos]
            self.val_demos = [utils.load_demos(utils.get_demos_path(env=env, origin=demos_origin, valid=True))[:1]
                              for env, demos_origin, _ in self.args.env]
            self.val_demos = [[demo[0] for demo in demos] for demos in self.val_demos]

            observation_space = self.env[0].observation_space
            action_space = self.env[0].action_space

        else:
            self.env = gym.make(self.args.env)

        demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
        demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)

        self.train_demos = utils.load_demos(demos_path)
        if args.episodes:
            assert args.episodes <= len(self.train_demos), "there are only {} train demos".format(len(self.train_demos))
            self.train_demos = self.train_demos[:args.episodes]

        self.val_demos = utils.load_demos(demos_path_valid)
        assert args.val_episodes <= len(self.val_demos), "there are only {} valid. demos".format(len(self.val_demos))
        self.val_demos = self.val_demos[:self.args.val_episodes]

        # Separating train offsets and train demos
        self.train_offsets = [item[1] for item in self.train_demos]
        self.train_demos = [item[0] for item in self.train_demos]
        self.val_demos = [item[0] for item in self.val_demos]

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        if type(self.args.env) == list:
            named_envs = '_'.join([item[0] for item in self.args.env])
        else:
            named_envs = self.args.env

        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = self.args.instr_arch if self.args.instr_arch else "noinstr"
        mem = "mem" if not args.no_mem else "nomem"
        model_name_parts = {
            'envs': named_envs,
            'arch': args.arch,
            'instr': instr,
            'mem': mem,
            'seed': args.seed,
            'suffix': suffix}
        default_model_name = "{envs}_IL_{arch}_{instr}_{mem}_seed{seed}_{suffix}".format(**model_name_parts)
        self.model_name = self.args.model or default_model_name
        print("The model is saved in {}".format(self.model_name))
        self.args.model = self.model_name

        self.obss_preprocessor = utils.ObssPreprocessor(self.model_name, observation_space)

        # Define actor-critic model
        self.acmodel = utils.load_model(self.model_name, raise_not_found=False)
        if self.acmodel is None:
            self.acmodel = ACModel(self.obss_preprocessor.obs_space, action_space, args.image_dim, args.memory_dim,
                                   not self.args.no_instr, self.args.instr_arch, not self.args.no_mem, self.args.arch)

        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.flat_train_demos, self.flat_val_demos = [], []
        if type(self.args.env) == list:
            for demos in self.train_demos:
                flat_demos = []
                for dm in demos:
                    flat_demos += dm
                flat_demos = np.array(flat_demos)
                self.flat_train_demos.append(flat_demos)
            for demos in self.val_demos:
                flat_demos = []
                for dm in demos:
                    flat_demos += dm
                flat_demos = np.array(flat_demos)
                self.flat_val_demos.append(flat_demos)
        else:
            for demo in self.train_demos:
                self.flat_train_demos += demo
            for demo in self.val_demos:
                self.flat_val_demos += demo
            self.flat_train_demos = np.array(self.flat_train_demos)
            self.flat_val_demos = np.array(self.flat_val_demos)

    def run_epoch_norecur(self, flat_demos, is_training=False):
        flat_demos_t = copy.deepcopy(flat_demos)
        if is_training:
            np.random.shuffle(flat_demos_t)
        batch_size = self.args.batch_size

        log = {"entropy": [], "policy_loss": [], "accuracy": []}
        offset = 0
        for j in range(len(flat_demos_t) // batch_size):
            flat_batch = flat_demos_t[offset:offset + batch_size, :]

            _log = self.run_epoch_norecur_one_batch(flat_batch, is_training=is_training)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])

            offset += batch_size

        return log

    def run_epoch_norecur_one_batch(self, flat_batch, is_training=False):
        obs, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 3]
        preprocessed_obs = self.obss_preprocessor(obs, device=self.device)

        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)
        memory = torch.zeros([self.args.batch_size, self.acmodel.memory_size], device=self.device)

        # Compute loss
        dist, _, memory = self.acmodel(preprocessed_obs, memory)

        entropy = dist.entropy().mean()

        policy_loss = -dist.log_prob(action_true).mean()

        action_pred = dist.probs.max(1, keepdim=True)[1]
        accuracy = float((action_pred == action_true.unsqueeze(1)).sum()) / self.args.batch_size

        loss = policy_loss - self.args.entropy_coef * entropy

        # Update actor-critic
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        log = {}
        log["entropy"] = float(entropy)
        log["policy_loss"] = float(policy_loss)
        log["accuracy"] = float(accuracy)
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
        log = {"entropy": [], "policy_loss": [], "accuracy": []}

        for batch_index in range(len(demos_t) // batch_size):
            batch = demos_t[offset:offset + batch_size]

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])

            offset += batch_size

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False):
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]

        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        flat_batch = np.array(flat_batch)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 3]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        memory = torch.zeros([self.args.batch_size, self.acmodel.memory_size], device=self.device)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            with torch.no_grad():
                # taking the memory till the length of time_step_inds, as demos beyond that have already finished
                _, _, new_memory = self.acmodel(preprocessed_obs, memory[:len(inds), :])

            for i in range(len(inds)):
                # Copying to the memories at the corresponding locations
                memories[inds[i], :] = memory[i, :]

            memory[:len(inds), :] = new_memory

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]

        # Here, actual backprop upto args.recurrence happens
        final_loss = 0
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0

        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence
        for _ in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            action_step = action_true[indexes]
            mask_step = mask[indexes]
            dist, value, memory = self.acmodel(preprocessed_obs, memory * mask_step)
            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            indexes += 1

        final_loss /= self.args.recurrence

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        log = {}
        log["entropy"] = float(final_entropy / self.args.recurrence)
        log["policy_loss"] = float(final_policy_loss / self.args.recurrence)
        log["accuracy"] = float(accuracy)

        return log

    def validate(self, verbose=True):
        # Seed needs to be reset for each validation, to ensure consistency
        utils.seed(self.args.val_seed)
        self.args.argmax = True
        if verbose:
            print("Validating the model")

        envs = self.env if type(self.env) == list else [self.env]
        agent = utils.load_agent(self.args, envs[0])
        # Setting the agent model to the current model
        agent.model = self.acmodel

        logs = []
        for env in envs:
            env.seed(self.args.val_seed)
            logs += [evaluate(agent, env, self.args.val_episodes)]

        if not self.args.no_mem:
            val_log = self.run_epoch_recurrence(self.val_demos)
        else:
            val_log = self.run_epoch_norecur(self.flat_val_demos)
        validation_accuracy = np.mean(val_log["accuracy"])

        if type(self.env) != list:
            assert len(logs) == 1
            return logs[0], validation_accuracy

        return logs, validation_accuracy

    def collect_returns(self):
        if torch.cuda.is_available():
            self.acmodel.cpu()
        mean_return = np.mean(self.validate(False)['return_per_episode'])
        if torch.cuda.is_available():
            self.acmodel.cuda()
        return mean_return

    def train(self, train_demos, logger, writer, csv_writer, status_path, header):
        # Load the status
        status = {'i': 0,
                  'num_frames': 0}
        if os.path.exists(status_path):
            with open(status_path, 'r') as src:
                status = json.load(src)

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.acmodel, self.model_name)

        # best mean return to keep track of performance on validation set
        best_mean_return, patience, i = 0, 0, 0
        total_start_time = time.time()

        # instantiate a valid_log with zeros for the first few updates before the evaluation of the validation metrics
        validation_data = [0.] * len([key for key in header if 'valid' in key])

        while True:
            status['i'] += 1
            i = status['i']
            update_start_time = time.time()

            # Learning rate scheduler
            self.scheduler.step()

            if not self.args.no_mem:
                log = self.run_epoch_recurrence(train_demos, is_training=True)
                total_len = sum([len(item) for item in train_demos])
            else:
                log = self.run_epoch_norecur(train_demos, is_training=True)
                total_len = len(train_demos)
            status['num_frames'] += total_len

            update_end_time = time.time()

            # Print logs
            if status['i'] % self.args.log_interval == 0:
                total_ellapsed_time = int(time.time() - total_start_time)

                fps = total_len / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                              log["entropy"], log["policy_loss"], log["accuracy"]]

                logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.validation_interval == 0
                if status['i'] % self.args.validation_interval != 0:
                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data + validation_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    if self.args.csv:
                        csv_writer.writerow(train_data + validation_data)

                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)

            if status['i'] % self.args.validation_interval == 0:
                if torch.cuda.is_available():
                    self.acmodel.cpu()
                valid_log, validation_accuracy = self.validate()
                mean_return = np.mean(valid_log['return_per_episode'])
                success_rate = np.mean([1 if r > 0 else 0 for r in valid_log['return_per_episode']])

                if status['i'] % self.args.log_interval == 0:
                    validation_data = [validation_accuracy, mean_return, success_rate]
                    logger.info("Validation: A {: .3f} | R {: .3f} | S {: .3f}".format(*validation_data))

                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data + validation_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    if self.args.csv:
                        csv_writer.writerow(train_data + validation_data)

                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)

                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    patience = 0
                    # Saving the model
                    logger.info("Saving best model")

                    self.obss_preprocessor.vocab.save()
                    if torch.cuda.is_available():
                        self.acmodel.cpu()
                    utils.save_model(self.acmodel, self.model_name)
                    if torch.cuda.is_available():
                        self.acmodel.cuda()
                else:
                    logger.info("Losing Patience")

                    patience += 1
                    if patience > self.args.patience:
                        break
                    if torch.cuda.is_available():
                        self.acmodel.cuda()
