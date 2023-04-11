import argparse
from distutils.util import strtobool
from typing import List

import imageio
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=str, required=True,
                        help="file path to hyperparameter json file")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="simplerl",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="nhartzler",
                        help="the entity (team) of wandb's project")

    args = parser.parse_args()
    return args


def create_policy_eval_video(wandb, policy, filename, eval_env, eval_py_env, num_episodes=10, fps=30):
    filename = filename + ".mp4"

    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            observation = policy.get_initial_state(eval_env.batch_size)
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                policy_step = policy.action(time_step, observation)
                time_step = eval_env.step(policy_step.action)
                video.append_data(eval_py_env.render())
                observation = policy_step.state

    wandb.log({"trained-agent-policy": wandb.Video(filename)})


def create_rewards_chart(wandb, num_iterations: int, eval_interval: int, rewards: List[float]):
    iterations = range(0, num_iterations + 1, eval_interval)

    plt.plot(iterations, rewards)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    wandb.log({"Final Average Return Plot": plt})
