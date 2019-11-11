#!/usr/bin/env python3
"""
Code for launching imitation learning sample efficiency experiments.
"""

import os
import time
import subprocess
import argparse
import math
from babyai.cluster_specific import launch_job

BIG_MODEL_PARAMS = '--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-arch=attgru --instr-dim=256'
SMALL_MODEL_PARAMS = '--batch-size=256'

def main(env, seed, training_time, min_demos, max_demos=None,
         step_size=math.sqrt(2), pretrained_model=None, level_type='small',
         val_episodes=512):
    demos = env

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
        epoch_length = 25600 if level_type == 'small' else 51200
        epochs = training_time // epoch_length

        # Print info
        print('{} demos, {} epochs of {} examples'.format(demo_count, epochs, epoch_length))

        # Form the command
        model_name = '{}_seed{}_{}'.format(demos, seed, demo_count)
        if pretrained_model:
            model_name += '_{}'.format(pretrained_model)
        jobname = '{}_efficiency'.format(demos, min_demos, max_demos)
        model_params = BIG_MODEL_PARAMS if level_type == 'big' else SMALL_MODEL_PARAMS
        cmd = ('{model_params} --val-episodes {val_episodes}'
               ' --seed {seed} --env {env} --demos {demos}'
               ' --val-interval 1 --log-interval 1 --epoch-length {epoch_length}'
               ' --model {model_name} --episodes {demo_count} --epochs {epochs} --patience {epochs}'
          .format(**locals()))
        if pretrained_model:
            cmd += ' --pretrained-model {}'.format(pretrained_model)
        launch_job(cmd, jobname)

        seed += 1
