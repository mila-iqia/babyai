import torch

import torch.nn.functional as F
import numpy
from babyai.rl.utils import DictList

# dictionary that defines what head is required for each extra info used for auxiliary supervision
required_heads = {'seen_state': 'binary',
                  'see_door': 'binary',
                  'see_obj': 'binary',
                  'obj_in_instr': 'binary',
                  'in_front_of_what': 'multiclass9',  # multi class classifier with 9 possible classes
                  'visit_proportion': 'continuous01',  # continous regressor with outputs in [0, 1]
                  'bot_action': 'binary'
                  }

class ExtraInfoCollector:
    '''
    This class, used in rl.algos.base, allows connecting the extra information from the environment, and the
    corresponding predictions using the specific heads in the model. It transforms them so that they are easy to use
    to evaluate losses
    '''
    def __init__(self, aux_info, shape, device):
        self.aux_info = aux_info
        self.shape = shape
        self.device = device

        self.collected_info = dict()
        self.extra_predictions = dict()
        for info in self.aux_info:
            self.collected_info[info] = torch.zeros(*shape, device=self.device)
            if required_heads[info] == 'binary' or required_heads[info].startswith('continuous'):
                # we predict one number only
                self.extra_predictions[info] = torch.zeros(*shape, 1, device=self.device)
            elif required_heads[info].startswith('multiclass'):
                # means that this is a multi-class classification and we need to predict the whole proba distr
                n_classes = int(required_heads[info].replace('multiclass', ''))
                self.extra_predictions[info] = torch.zeros(*shape, n_classes, device=self.device)
            else:
                raise ValueError("{} not supported".format(required_heads[info]))

    def process(self, env_info):
        # env_info is now a tuple of dicts
        env_info = [{k: v for k, v in dic.items() if k in self.aux_info} for dic in env_info]
        env_info = {k: [env_info[_][k] for _ in range(len(env_info))] for k in env_info[0].keys()}
        # env_info is now a dict of lists
        return env_info

    def fill_dictionaries(self, index, env_info, extra_predictions):
        for info in self.aux_info:
            dtype = torch.long if required_heads[info].startswith('multiclass') else torch.float
            self.collected_info[info][index] = torch.tensor(env_info[info], dtype=dtype, device=self.device)
            self.extra_predictions[info][index] = extra_predictions[info]

    def end_collection(self, exps):
        collected_info = dict()
        extra_predictions = dict()
        for info in self.aux_info:
            # T x P -> P x T -> P * T
            collected_info[info] = self.collected_info[info].transpose(0, 1).reshape(-1)
            if required_heads[info] == 'binary' or required_heads[info].startswith('continuous'):
                # T x P x 1 -> P x T x 1 -> P * T
                extra_predictions[info] = self.extra_predictions[info].transpose(0, 1).reshape(-1)
            elif type(required_heads[info]) == int:
                # T x P x k -> P x T x k -> (P * T) x k
                k = required_heads[info]  # number of classes
                extra_predictions[info] = self.extra_predictions[info].transpose(0, 1).reshape(-1, k)
        # convert the dicts to DictLists, and add them to the exps DictList.
        exps.collected_info = DictList(collected_info)
        exps.extra_predictions = DictList(extra_predictions)

        return exps


class SupervisedLossUpdater:
    '''
    This class, used by PPO, allows the evaluation of the supervised loss when using extra information from the
    environment. It also handles logging accuracies/L2 distances/etc...
    '''
    def __init__(self, aux_info, supervised_loss_coef, recurrence, device):
        self.aux_info = aux_info
        self.supervised_loss_coef = supervised_loss_coef
        self.recurrence = recurrence
        self.device = device

        self.log_supervised_losses = []
        self.log_supervised_accuracies = []
        self.log_supervised_L2_losses = []
        self.log_supervised_prevalences = []

        self.batch_supervised_loss = 0
        self.batch_supervised_accuracy = 0
        self.batch_supervised_L2_loss = 0
        self.batch_supervised_prevalence = 0

    def init_epoch(self):
        self.log_supervised_losses = []
        self.log_supervised_accuracies = []
        self.log_supervised_L2_losses = []
        self.log_supervised_prevalences = []

    def init_batch(self):
        self.batch_supervised_loss = 0
        self.batch_supervised_accuracy = 0
        self.batch_supervised_L2_loss = 0
        self.batch_supervised_prevalence = 0

    def eval_subbatch(self, extra_predictions, sb):
        supervised_loss = torch.tensor(0., device=self.device)
        supervised_accuracy = torch.tensor(0., device=self.device)
        supervised_L2_loss = torch.tensor(0., device=self.device)
        supervised_prevalence = torch.tensor(0., device=self.device)

        binary_classification_tasks = 0
        classification_tasks = 0
        regression_tasks = 0

        for pos, info in enumerate(self.aux_info):
            coef = self.supervised_loss_coef[pos]
            pred = extra_predictions[info]
            target = dict.__getitem__(sb.collected_info, info)
            if required_heads[info] == 'binary':
                binary_classification_tasks += 1
                classification_tasks += 1
                supervised_loss += coef * F.binary_cross_entropy_with_logits(pred.reshape(-1), target)
                supervised_accuracy += ((pred.reshape(-1) > 0).float() == target).float().mean()
                supervised_prevalence += target.mean()
            elif required_heads[info].startswith('continuous'):
                regression_tasks += 1
                mse = F.mse_loss(pred.reshape(-1), target)
                supervised_loss += coef * mse
                supervised_L2_loss += mse
            elif required_heads[info].startswith('multiclass'):
                classification_tasks += 1
                supervised_accuracy += (pred.argmax(1).float() == target).float().mean()
                supervised_loss += coef * F.cross_entropy(pred, target.long())
            else:
                raise ValueError("{} not supported".format(required_heads[info]))
        if binary_classification_tasks > 0:
            supervised_prevalence /= binary_classification_tasks
        else:
            supervised_prevalence = torch.tensor(-1)
        if classification_tasks > 0:
            supervised_accuracy /= classification_tasks
        else:
            supervised_accuracy = torch.tensor(-1)
        if regression_tasks > 0:
            supervised_L2_loss /= regression_tasks
        else:
            supervised_L2_loss = torch.tensor(-1)

        self.batch_supervised_loss += supervised_loss.item()
        self.batch_supervised_accuracy += supervised_accuracy.item()
        self.batch_supervised_L2_loss += supervised_L2_loss.item()
        self.batch_supervised_prevalence += supervised_prevalence.item()

        return supervised_loss

    def update_batch_values(self):
        self.batch_supervised_loss /= self.recurrence
        self.batch_supervised_accuracy /= self.recurrence
        self.batch_supervised_L2_loss /= self.recurrence
        self.batch_supervised_prevalence /= self.recurrence

    def update_epoch_logs(self):
        self.log_supervised_losses.append(self.batch_supervised_loss)
        self.log_supervised_accuracies.append(self.batch_supervised_accuracy)
        self.log_supervised_L2_losses.append(self.batch_supervised_L2_loss)
        self.log_supervised_prevalences.append(self.batch_supervised_prevalence)

    def end_training(self, logs):
        logs["supervised_loss"] = numpy.mean(self.log_supervised_losses)
        logs["supervised_accuracy"] = numpy.mean(self.log_supervised_accuracies)
        logs["supervised_L2_loss"] = numpy.mean(self.log_supervised_L2_losses)
        logs["supervised_prevalence"] = numpy.mean(self.log_supervised_prevalences)

        return logs
