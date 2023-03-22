from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

def make_environments(env_name: str):
    env = suite_gym.load(env_name)
    env.reset()
    specs = (
            env.time_step_spec().observation,
            env.action_spec(),
            env.time_step_spec().reward
        )
    
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return (train_env, eval_env, specs)

