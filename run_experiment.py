from __future__ import absolute_import, division, print_function

import json
import pyvirtualdisplay
#import pygad

import tensorflow as tf

from tf_agents.policies import policy_saver

from modules.experiment_runner import experiment_runner
from modules.utils import parse_args, create_policy_eval_video, create_rewards_chart


if __name__ == "__main__":
    args = parse_args()
    with open(args.hparams) as json_file:
        hparams = json.load(json_file)
        experiment_name = json_file.name.replace('.json', '')
        hparams['experiment_name'] = experiment_name

    # Tensorflow import checks
    tf.get_logger().setLevel('ERROR')
    print("\n\n\ntf version:", tf.version.VERSION)
    print(tf.config.list_physical_devices('GPU'), '\n\n')

    # Set up a virtual display for rendering OpenAI gym environments.
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    # Setup Weights and Biases
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=hparams,
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        hparams['track'] = True
    else:
        hparams['track'] = False

    rewards, agent, eval_tf_env, eval_py_env = experiment_runner(hparams)

    # Save Agent Policy
    policy_saver.PolicySaver(agent.policy, batch_size=hparams['batch_size']).save(
        f'model_saves/{experiment_name}')

    if args.track:
        # Create Plot
        #create_rewards_chart(wandb, hparams['num_iterations'], hparams['eval_interval'], rewards)

        # Create Video
        create_policy_eval_video(
            wandb, agent.policy, f"videos/{experiment_name}_trained-agent", eval_tf_env, eval_py_env)
