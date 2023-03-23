from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

def make_environments(env_name: str):
    
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return (train_tf_env, eval_tf_env, train_py_env)

