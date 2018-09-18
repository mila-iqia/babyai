#!/usr/bin/env python3

import os
import time
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_true', help="print launch commands without executing them")
args = parser.parse_args()

SBATCH_PARAMS = '--account=rpp-bengioy --mail-user maximechevalierb@gmail.com --mail-type ALL --ntasks=1'
#MODEL_PARAMS = '--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-dim=256'
MODEL_PARAMS = ''

def launch_job(launch_cmd, job_name, time_limit):
    log_name = 'slurm-{}-%j.out'.format(job_name)
    sbatch_cmd = 'sbatch {} --time={} --gres=gpu:1 -c 4 --mem=16000M --error={} --output={} scripts/sbatch/run.sh {}'.format(SBATCH_PARAMS, time_limit, log_name, log_name, launch_cmd)
    print('launching job:', sbatch_cmd)

    if not args.test:
        subprocess.check_call(sbatch_cmd, shell=True)
        time.sleep(1)

    print()

def launch_intelligent_expert(seed, level, time_limit):
    env = 'BabyAI-{}-v0'.format(level)
    demos = '{}-bot-1m'.format(level)
    model_name = 'ie_{}_seed_{}_{}'.format(level, seed, time.strftime("%a_%H_%M_%S"))
    cmd = 'python3 -m scripts.train_intelligent_expert {} --demos {} --seed {} --env {} --validation-interval 2 --model {}'.format(MODEL_PARAMS, demos, seed, env, model_name)
    launch_job(cmd, model_name, time_limit)

def launch_intelligent_curriculum(seed, level, time_limit):
    env = 'BabyAI-{}-v0'.format(level)
    demos = '{}-bot-1m'.format(level)
    model_name = 'ic_{}_seed_{}_{}'.format(level, seed, time.strftime("%a_%H_%M_%S"))
    cmd = 'python3 -m scripts.train_intelligent_curriculum {} --demos {} --seed {} --env {} --validation-interval 2 --model {}'.format(MODEL_PARAMS, demos, seed, env, model_name)
    launch_job(cmd, model_name, time_limit)

for seed in range(0, 4):
    #launch_intelligent_expert(seed, 'GoToRedBall')
    launch_intelligent_expert(seed, 'GoToRedBallObs', '12:00:00')
    launch_intelligent_expert(seed, 'GoToLocal', '48:00:00')
    #launch_intelligent_curriculum(seed, 'GoToLocal')
