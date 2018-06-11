#!/usr/bin/env python3

import numpy as np
import argparse
import csv
from babyai.algos.imitation import ImitationLearning
import scripts.evaluate as evaluate
import babyai.utils as utils
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos-origin", required=True,
                    help="origin of the demonstrations: human | agent (REQUIRED)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.2,
                    help="entropy term coefficient (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=50,
                    help="batch size (In case of memory, the batch size is the number of demos, otherwise, it is the number of frames)(default: 50)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='cnn1',
                    help="image embedding architecture")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--value-loss-coef", type=float, default=0,
                    help="value loss term coefficient (default: 0)")
parser.add_argument("--validation-interval", type=int, default=20,
                    help="number of epochs between two validation checks (default: 20)")
parser.add_argument("--episodes-to-add", type=int, default=100,
                    help="number of episodes to add each time  (default: 100)")
parser.add_argument("--patience", type=int, default=3,
                    help="patience for early stopping (default: 3)")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")
parser.add_argument("--start-demo", type=int, default=50,
					help="the minimum number of demonstrations to start searching (default: 50)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ORIGIN_il)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--val-episodes", type=int, default=1000,
                    help="number of episodes used for validation (default: 1000)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, conv, bow (default: gru)")


args = parser.parse_args()
start_demo = args.start_demo
batch_size = args.batch_size


def add_new_demos(il_learn, logs, batch_size):
     offset = 0
     min_accuracy = 1.0
     min_index = 0
     for item in logs["accuracy"]:
          if item < min_accuracy:
               min_accuracy = item
               min_index = offset
          offset += batch_size
     print("Minimum Accuracy: %.3f" % min_accuracy)
    
     
     if args.no_mem:
          new_batch = il_learn.flat_train_demos[min_index:min_index+batch_size]
     else:
          new_batch = il_learn.train_demos[min_index:min_index+batch_size]
     
     
     new_demos = []
     if args.no_mem:
          indexes = [i for i in range(len(new_batch)) if new_batch[i][3] == 1]
          for i in range(len(indexes)-1):
               new_demos.append(new_batch[indexes[i]+1:indexes[i+1]+1])
     else:
          new_demos = new_batch
     return new_demos

def flatten(demos):
     flat_demos = []
     for demo in demos:
          flat_demos.extend(demo)
     return np.array(flat_demos)


def main():
     args.episodes = 5000
     il_learn = ImitationLearning(args)
     # Define logger 
     logger = utils.get_logger(il_learn.model_name)
     
     # Log command, availability of CUDA, and model
     logger.info(args)
     logger.info("CUDA available: {}".format(torch.cuda.is_available()))
     logger.info(il_learn.acmodel)
     avg_demo_len = len(il_learn.flat_train_demos)/len(il_learn.train_demos)


     
     train_demos = il_learn.train_demos[:start_demo]
    
     while True:
          writer = None
          if args.tb:
             from tensorboardX import SummaryWriter
             writer = SummaryWriter(utils.get_log_dir(il_learn.model_name+"_"+str(len(train_demos))))
          
          print("Training for %d demos" % len(train_demos))
          
          if args.no_mem:
               flat_train_demos = flatten(train_demos)
          
          args.batch_size = batch_size
          
          if not args.no_mem:
               il_learn.train(train_demos, logger, writer)
          else:
               il_learn.train(flat_train_demos, logger, writer)

          args.batch_size = args.episodes_to_add
          
          if torch.cuda.is_available():
               il_learn.acmodel.cpu()
     
          if not args.no_mem:
               logs = il_learn.run_epoch_recurrence(il_learn.train_demos)
          else:
               args.batch_size = int(args.batch_size*avg_demo_len)
               logs = il_learn.run_epoch_norecur(il_learn.flat_train_demos)
          
          if torch.cuda.is_available():
              il_learn.acmodel.cuda()
     
          train_demos = train_demos + add_new_demos(il_learn, logs, args.batch_size)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

          










