from __future__ import absolute_import, division, print_function

import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb
import os
import shutil
import json

import tensorflow as tf

from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics

import modules.eval as eval
from modules.agent import make_agent
from modules.environment import make_environments
from modules.utils import parse_args



if __name__ == "__main__":
    args = parse_args()
    with open(args.hparams) as json_file:
        hparams = json.load(json_file)

    # Tensorflow import checks    
    tf.get_logger().setLevel('ERROR')
    print("\n\n\ntf version:", tf.version.VERSION)
    print(tf.config.list_physical_devices('GPU'),'\n\n')

    # Set up a virtual display for rendering OpenAI gym environments.
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    # Environments
    train_env, eval_env, specs = make_environments(hparams['environment_name'])
    spec_obs, spec_action, spec_reward = specs
    
    # Agent
    agent = make_agent(train_env.time_step_spec(), spec_obs, spec_action, hparams)
    
    avg_reward = eval.compute_avg_reward(eval_env, agent.policy, hparams['num_eval_episodes'])
    print('\navg_reward:', avg_reward)