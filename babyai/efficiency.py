#!/usr/bin/env python3

"""
scripts/launch_demo_count.py --env BabyAI-GoToLocal-v0 --demos GoToLocal-bot-1m
"""

import os
import time
import subprocess
import argparse
import math
from babyai.cluster_specific import launch_job

BIG_MODEL_PARAMS = '--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-arch=attgru --instr-dim=256'
SMALL_MODEL_PARAMS = '--batch-size=256'

def main(env, seed, min_demos, max_demos=None,
         step_size=math.sqrt(2), pretrained_model=None):
    level_size = 'big' if env == 'BabyAI-GoTo-v0' else 'small'
    demos = env + "-seed1"

    if not max_demos:
        max_demos = min_demos
        min_demos = max_demos - 1

    demo_counts = []
    demo_count = max_demos
    while demo_count >= min_demos:
        demo_counts.append(demo_count)
        demo_count = math.ceil(demo_count / step_size)

    for demo_count in demo_counts:
        # Decide on the parameters
        epoch_length = 25600
        if level_size == 'big':
            epoch_length = 51200
        target_examples = 1000000 * (40 if level_size == 'big' else 80)
        epochs = target_examples // epoch_length

        # Print info
        print('{} demos, {} epochs of {} examples'.format(demo_count, epochs, epoch_length))

        # Form the command
        model_name = '{}_seed{}_{}'.format(demos, seed, demo_count)
        if pretrained_model:
            model_name += '_{}'.format(pretrained_model)
        jobname = '{}_efficiency'.format(demos, min_demos, max_demos)
        model_params = BIG_MODEL_PARAMS if level_size == 'big' else SMALL_MODEL_PARAMS
        cmd = ('{model_params} '
               ' --seed {seed} --env {env} --demos {demos}'
               ' --val-interval 1 --log-interval 1 --epoch-length {epoch_length}'
               ' --model {model_name} --episodes {demo_count} --epochs {epochs}'
          .format(**locals()))
        if pretrained_model:
            cmd += ' --pretrained-model {}'.format(pretrained_model)
        launch_job(cmd, jobname)
