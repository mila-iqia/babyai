#!/usr/bin/env python3

"""
scripts/launch_demo_count.py --env BabyAI-GoToLocal-v0 --demos GoToLocal-bot-1m
"""

import os
import time
import subprocess
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_true', help="print launch commands without executing them")
parser.add_argument("--env", required=True, help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos", required=True, help="demos filename (REQUIRED or demos-origin required)")
parser.add_argument("--seed", type=int, default=0, help="random seed to use")
parser.add_argument("--min-demos", type=int, default=30000, help="minimum number of demos to test")
parser.add_argument("--max-demos", type=int, default=1000000, help="maximum number of demos to test")
parser.add_argument("--step-size", type=float, default=math.sqrt(2), help="step size for the demo counts")
args = parser.parse_args()

SBATCH_PARAMS = '--account=rpp-bengioy --mail-user maximechevalierb@gmail.com --mail-type ALL --ntasks=1'
MODEL_PARAMS = '--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-arch=attgru --instr-dim=256'

def launch_job(launch_cmd, job_name):
    log_name = 'slurm-{}-%j.out'.format(job_name)
    sbatch_cmd = 'sbatch {} --time=72:00:00 --gres=gpu:1 -c 4 --mem=16000M --error={} --output={} scripts/sbatch/run.sh {}'.format(SBATCH_PARAMS, log_name, log_name, launch_cmd)
    print('launching job:', sbatch_cmd)

    if not args.test:
        subprocess.check_call(sbatch_cmd, shell=True)
        time.sleep(1)

    print()

demo_counts = []
demo_count = args.max_demos
while demo_count >= args.min_demos:
    demo_counts.append(demo_count)
    demo_count = math.ceil(demo_count / args.step_size)

for demo_count in demo_counts:
    print(demo_count)
    model_name = 'il_{}_seed_{}_{}'.format(args.env, args.seed, time.strftime("%a_%H_%M_%S"))
    cmd = 'python3 -m scripts.train_il {} --seed {} --env {} --demos {} --validation-interval 2 --model {} --episodes {}'.format(MODEL_PARAMS, args.seed, args.env, args.demos, model_name, demo_count)
    launch_job(cmd, model_name)
