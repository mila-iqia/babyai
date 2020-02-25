import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
from babyai.evaluate import batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import multiprocessing
import os
import json
import logging

logger = logging.getLogger(__name__)

import numpy


class EpochIndexSampler:
    """
    Generate smart indices for epochs that are smaller than the dataset size.

    The usecase: you have a code that has a strongly baken in notion of an epoch,
    e.g. you can only validate in the end of the epoch. That ties a lot of
    aspects of training to the size of the dataset. You may want to validate
    more often than once per a complete pass over the dataset.

    This class helps you by generating a sequence of smaller epochs that
    use different subsets of the dataset, as long as this is possible.
    This allows you to keep the small advantage that sampling without replacement
    provides, but also enjoy smaller epochs.
    """
    def __init__(self, n_examples, epoch_n_examples):
        self.n_examples = n_examples
        self.epoch_n_examples = epoch_n_examples

        self._last_seed = None

    def _reseed_indices_if_needed(self, seed):
        if seed == self._last_seed:
            return

        rng = numpy.random.RandomState(seed)
        self._indices = list(range(self.n_examples))
        rng.shuffle(self._indices)
        logger.info('reshuffle the dataset')

        self._last_seed = seed

    def get_epoch_indices(self, epoch):
        """Return indices corresponding to a particular epoch.

        Tip: if you call this function with consecutive epoch numbers,
        you will avoid expensive reshuffling of the index list.

        """
        seed = epoch * self.epoch_n_examples // self.n_examples
        offset = epoch * self.epoch_n_examples % self.n_examples

        indices = []
        while len(indices) < self.epoch_n_examples:
            self._reseed_indices_if_needed(seed)
            n_lacking = self.epoch_n_examples - len(indices)
            indices += self._indices[offset:offset + min(n_lacking, self.n_examples - offset)]
            offset = 0
            seed += 1

        return indices


class ImitationLearning(object):
    def __init__(self, args, ):
        self.args = args

        utils.seed(self.args.seed)
        self.val_seed = self.args.val_seed

        # args.env is a list when training on multiple environments
        if getattr(args, 'multi_env', None):
            self.env = [gym.make(item) for item in args.multi_env]

            self.train_demos = []
            for demos, episodes in zip(args.multi_demos, args.multi_episodes):
                demos_path = utils.get_demos_path(demos, None, None, valid=False)
                logger.info('loading {} of {} demos'.format(episodes, demos))
                train_demos = utils.load_demos(demos_path)
                logger.info('loaded demos')
                if episodes > len(train_demos):
                    raise ValueError("there are only {} train demos in {}".format(len(train_demos), demos))
                self.train_demos.extend(train_demos[:episodes])
                logger.info('So far, {} demos loaded'.format(len(self.train_demos)))

            self.val_demos = []
            for demos, episodes in zip(args.multi_demos, [args.val_episodes] * len(args.multi_demos)):
                demos_path_valid = utils.get_demos_path(demos, None, None, valid=True)
                logger.info('loading {} of {} valid demos'.format(episodes, demos))
                valid_demos = utils.load_demos(demos_path_valid)
                logger.info('loaded demos')
                if episodes > len(valid_demos):
                    logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(valid_demos)))
                self.val_demos.extend(valid_demos[:episodes])
                logger.info('So far, {} valid demos loaded'.format(len(self.val_demos)))

            logger.info('Loaded all demos')

            observation_space = self.env[0].observation_space
            action_space = self.env[0].action_space

        else:
            self.env = gym.make(self.args.env)

            demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
            demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)

            logger.info('loading demos')
            self.train_demos = utils.load_demos(demos_path)
            logger.info('loaded demos')
            if args.episodes:
                if args.episodes > len(self.train_demos):
                    raise ValueError("there are only {} train demos".format(len(self.train_demos)))
                self.train_demos = self.train_demos[:args.episodes]

            self.val_demos = utils.load_demos(demos_path_valid)
            if args.val_episodes > len(self.val_demos):
                logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.val_demos)))
            self.val_demos = self.val_demos[:self.args.val_episodes]

            observation_space = self.env.observation_space
            action_space = self.env.action_space

        self.obss_preprocessor = utils.ObssPreprocessor(args.model, observation_space,
                                                        getattr(self.args, 'pretrained_model', None))

        # Define actor-critic model
        self.acmodel = utils.load_model(args.model, raise_not_found=False)
        if self.acmodel is None:
            if getattr(self.args, 'pretrained_model', None):
                self.acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
            else:
                logger.info('Creating new model')
                self.acmodel = ACModel(self.obss_preprocessor.obs_space, action_space,
                                       args.image_dim, args.memory_dim, args.instr_dim,
                                       not self.args.no_instr, self.args.instr_arch,
                                       not self.args.no_mem, self.args.arch)
        self.obss_preprocessor.vocab.save()
        utils.save_model(self.acmodel, args.model)

        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def default_model_name(args):
        if getattr(args, 'multi_env', None):
            # It's better to specify one's own model name for this scenario
            named_envs = '-'.join(args.multi_env)
        else:
            named_envs = args.env

        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = args.instr_arch if args.instr_arch else "noinstr"
        model_name_parts = {
            'envs': named_envs,
            'arch': args.arch,
            'instr': instr,
            'seed': args.seed,
            'suffix': suffix}
        default_model_name = "{envs}_IL_{arch}_{instr}_seed{seed}_{suffix}".format(**model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
        return default_model_name

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def run_epoch_recurrence(self, demos, is_training=False, indices=None):
        if not indices:
            indices = list(range(len(demos)))
            if is_training:
                np.random.shuffle(indices)
        batch_size = min(self.args.batch_size, len(demos))
        offset = 0

        if not is_training:
            self.acmodel.eval()

        # Log dictionary
        log = {"entropy": [], "policy_loss": [], "accuracy": []}

        start_time = time.time()
        frames = 0
        for batch_index in range(len(indices) // batch_size):
            logger.info("batch {}, FPS so far {}".format(
                batch_index, frames / (time.time() - start_time) if frames else 0))
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames += sum([len(demo[3]) for demo in batch])

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])

            offset += batch_size
        log['total_frames'] = frames

        if not is_training:
            self.acmodel.train()

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False):
        batch = utils.demos.transform_demos(batch)
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
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), self.acmodel.memory_size], device=self.device)

        preprocessed_first_obs = self.obss_preprocessor(obss[inds], device=self.device)
        instr_embedding = self.acmodel._get_instr_embedding(preprocessed_first_obs.instr)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished
                new_memory = self.acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], instr_embedding[:len(inds)])['memory']

            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory
            episode_ids[inds] = range(len(inds))

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
            model_results = self.acmodel(
                preprocessed_obs, memory * mask_step,
                instr_embedding[episode_ids[indexes]])
            dist = model_results['dist']
            memory = model_results['memory']

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

    def validate(self, episodes, verbose=True):
        if verbose:
            logger.info("Validating the model")
        if getattr(self.args, 'multi_env', None):
            agent = utils.load_agent(self.env[0], model_name=self.args.model, argmax=True)
        else:
            agent = utils.load_agent(self.env, model_name=self.args.model, argmax=True)

        # Setting the agent model to the current model
        agent.model = self.acmodel

        agent.model.eval()
        logs = []

        for env_name in ([self.args.env] if not getattr(self.args, 'multi_env', None)
                         else self.args.multi_env):
            logs += [batch_evaluate(agent, env_name, self.val_seed, episodes)]
            self.val_seed += episodes
        agent.model.train()

        return logs

    def collect_returns(self):
        logs = self.validate(episodes=self.args.eval_episodes, verbose=False)
        mean_return = {tid: np.mean(log["return_per_episode"]) for tid, log in enumerate(logs)}
        return mean_return

    def train(self, train_demos, writer, csv_writer, status_path, header, reset_status=False):
        # Load the status
        def initial_status():
            return {'i': 0,
                    'num_frames': 0,
                    'patience': 0}

        status = initial_status()
        if os.path.exists(status_path) and not reset_status:
            with open(status_path, 'r') as src:
                status = json.load(src)
        elif not os.path.exists(os.path.dirname(status_path)):
            # Ensure that the status directory exists
            os.makedirs(os.path.dirname(status_path))

        # If the batch size is larger than the number of demos, we need to lower the batch size
        if self.args.batch_size > len(train_demos):
            self.args.batch_size = len(train_demos)
            logger.info("Batch size too high. Setting it to the number of train demos ({})".format(len(train_demos)))

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.acmodel, self.args.model)

        # best mean return to keep track of performance on validation set
        best_success_rate, patience, i = 0, 0, 0
        total_start_time = time.time()

        epoch_length = self.args.epoch_length
        if not epoch_length:
            epoch_length = len(train_demos)
        index_sampler = EpochIndexSampler(len(train_demos), epoch_length)

        while status['i'] < getattr(self.args, 'epochs', int(1e9)):
            if 'patience' not in status:  # if for some reason you're finetuining with IL an RL pretrained agent
                status['patience'] = 0
            # Do not learn if using a pre-trained model that already lost patience
            if status['patience'] > self.args.patience:
                break
            if status['num_frames'] > self.args.frames:
                break

            update_start_time = time.time()

            indices = index_sampler.get_epoch_indices(status['i'])
            log = self.run_epoch_recurrence(train_demos, is_training=True, indices=indices)
            
            # Learning rate scheduler
            self.scheduler.step()

            status['num_frames'] += log['total_frames']
            status['i'] += 1

            update_end_time = time.time()

            # Print logs
            if status['i'] % self.args.log_interval == 0:
                total_ellapsed_time = int(time.time() - total_start_time)

                fps = log['total_frames'] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                              log["entropy"], log["policy_loss"], log["accuracy"]]

                logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.val_interval == 0
                if status['i'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings when no validation is done
                    validation_data = [''] * len([key for key in header if 'valid' in key])
                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

            if status['i'] % self.args.val_interval == 0:

                valid_log = self.validate(self.args.val_episodes)
                mean_return = [np.mean(log['return_per_episode']) for log in valid_log]
                success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in
                                valid_log]

                val_log = self.run_epoch_recurrence(self.val_demos)
                validation_accuracy = np.mean(val_log["accuracy"])

                if status['i'] % self.args.log_interval == 0:
                    validation_data = [validation_accuracy] + mean_return + success_rate
                    logger.info(("Validation: A {: .3f} " + ("| R {: .3f} " * len(mean_return) +
                                                             "| S {: .3f} " * len(success_rate))
                                 ).format(*validation_data))

                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data + validation_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

                # In case of a multi-env, the update condition would be "better mean success rate" !
                if np.mean(success_rate) > best_success_rate:
                    best_success_rate = np.mean(success_rate)
                    status['patience'] = 0
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
                    # Saving the model
                    logger.info("Saving best model")

                    if torch.cuda.is_available():
                        self.acmodel.cpu()
                    utils.save_model(self.acmodel, self.args.model + "_best")
                    self.obss_preprocessor.vocab.save(utils.get_vocab_path(self.args.model + "_best"))
                    if torch.cuda.is_available():
                        self.acmodel.cuda()
                else:
                    status['patience'] += 1
                    logger.info(
                        "Losing patience, new value={}, limit={}".format(status['patience'], self.args.patience))

                if torch.cuda.is_available():
                    self.acmodel.cpu()
                utils.save_model(self.acmodel, self.args.model)
                self.obss_preprocessor.vocab.save()
                if torch.cuda.is_available():
                    self.acmodel.cuda()
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)

        return best_success_rate
