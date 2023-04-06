import numpy

from modules.agent import make_agent
from modules.environment import make_environments
from modules.utils import create_policy_eval_video
from modules.training import run_training_loop


def experiment_runner(hparams: dict):

    # Environments
    train_tf_env, eval_tf_env, train_py_env, eval_py_env = make_environments(
        hparams['environment_name'])

    # Agent
    agent = make_agent(
        train_tf_env.time_step_spec(),
        train_tf_env.time_step_spec().observation,
        train_tf_env.action_spec(),
        hparams
    )

    # Training Loop
    rewards, training_duration = run_training_loop(
        agent, train_tf_env, eval_tf_env, train_py_env, hparams)

    print('\n\nTraining Duration:', training_duration)
    print('Average Reward', sum(rewards) / len(rewards))

    return (rewards, agent)
