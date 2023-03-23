from __future__ import absolute_import, division, print_function

import json
import pyvirtualdisplay

import tensorflow as tf

from modules.agent import make_agent
from modules.environment import make_environments
from modules.utils import parse_args
from modules.training import run_training_loop


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
    train_tf_env, eval_tf_env, train_py_env = make_environments(hparams['environment_name'])
    
    # Agent
    agent = make_agent(
            train_tf_env.time_step_spec(), 
            train_tf_env.time_step_spec().observation, 
            train_tf_env.action_spec(), 
            hparams
        )

    # Training Loop
    rewards, training_duration = run_training_loop(agent, train_tf_env, eval_tf_env, train_py_env, hparams)

    print('\n\nTraining Duration:', training_duration)
    print('Average Reward', sum(rewards) / len(rewards))
    