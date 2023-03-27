import numpy
import random
from timeit import default_timer
from typing import List, Tuple

import tensorflow as tf

from tf_agents.utils import common
from tf_agents.environments import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories import trajectory
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from modules.eval import compute_avg_reward
from modules.replay import start_replay_server

def run_training_loop(
        agent, 
        train_tf_env: TFPyEnvironment, 
        eval_tf_env: TFPyEnvironment,
        train_py_env: PyEnvironment,
        hparams: dict
    ) -> Tuple[List[numpy.float32], float]:
    # Check tracking
    tracking = False
    if hparams['track']:
        import wandb
        tracking = True


    # Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    print('\nRunning initial agent evaluation')
    avg_reward = lambda: compute_avg_reward(eval_tf_env, agent.policy, hparams['num_eval_episodes'])
    rewards = [avg_reward()]

    # Reset the training environment.
    time_step = train_py_env.reset()

    # Setup the policy and get the initial state of the agent
    policy = PyTFEagerPolicy(agent.policy, use_tf_function=True)
    observation = policy.get_initial_state(train_tf_env.batch_size)

    # Start the Replay server
    replay = []
    get_replay_sample = lambda x: [random.choice(replay) for _ in range(x)]

    # Populate the Replay server with an initial set of game plays
    print("\nPopulating the Replay server with an initial set of {} environment steps".format(hparams['initial_collect_steps']))
    for _ in range(hparams['initial_collect_steps']):
        policy_step = policy.action(time_step, observation)
        next_time_step = train_py_env.step(policy_step.action)
        next_observation = policy_step.state
        # Add step to replay server
        policy_step_with_previous_observation = policy_step._replace(state=observation)
        trajectory_to_write = trajectory.from_transition(
            time_step, policy_step_with_previous_observation, next_time_step
        )
        replay.append(trajectory_to_write)
        # Set next time step and observation
        time_step = next_time_step
        observation = next_observation


    # Reset the training environment and agent observation.
    time_step = train_py_env.reset()
    observation = policy.get_initial_state(train_tf_env.batch_size)
    
    # Setup the timer
    training_start = default_timer()
    episode_times = []

    # Start the training loop
    print('\n\n--------Running Training Loop--------\n')
    for _ in range(hparams['num_iterations']):
        episode_start = default_timer()

        policy_step = policy.action(time_step, observation)
        next_time_step = train_py_env.step(policy_step.action)
        next_observation = policy_step.state
        # Add step to replay server
        policy_step_with_previous_observation = policy_step._replace(state=observation)
        trajectory_to_write = trajectory.from_transition(
            time_step, policy_step_with_previous_observation, next_time_step
        )
        replay.append(trajectory_to_write)
        # Set next time step and observation
        time_step = next_time_step
        observation = next_observation
        
        # Sample a batch of data from the replay server and update the agent's network.
        experience = get_replay_sample(hparams['batch_size'])
        train_loss = agent.train(experience).loss
        if tracking:
            wandb.log({'loss': train_loss})
        step = agent.train_step_counter.numpy()
        
        if step % hparams['log_interval'] == 0:
            # Log progress to console
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % hparams['eval_interval'] == 0:
            # Evaluate the agent
            print('step = {0}: Evaluating...'.format(step))
            current_avg_reward = avg_reward()
            current_avg_episode_duration = sum(episode_times) / len(episode_times)
            print('step = {0}: Average Reward = {1}'.format(step, current_avg_reward))
            print('step = {0}: Average Episode Duration = {1}'.format(step, current_avg_episode_duration))
            rewards.append(current_avg_reward)
            
            if tracking:
                wandb.log({'avg_reward': current_avg_reward, 'avg_episode_duration': current_avg_episode_duration})


        episode_duration = default_timer() - episode_start
        episode_times.append(episode_duration)
    
    training_duration = default_timer() - training_start
    
    return (rewards, training_duration)
    


