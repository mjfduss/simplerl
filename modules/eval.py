import numpy
from tf_agents.environments import tf_py_environment
from tf_agents.policies import TFPolicy


def compute_avg_reward(
      tf_env: tf_py_environment.TFPyEnvironment, 
      policy: TFPolicy, 
      num_episodes=10
    ) -> numpy.float32:

  total_reward = 0.0
  for _ in range(num_episodes):

    observation = policy.get_initial_state(tf_env.batch_size)
    time_step = tf_env.reset()
    episode_reward = 0.0

    while not time_step.is_last():
      policy_step = policy.action(time_step, observation)
      time_step = tf_env.step(policy_step.action)
      episode_reward += time_step.reward
      observation = policy_step.state
      
    total_reward += episode_reward

  avg_reward = total_reward / num_episodes
  return avg_reward.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics