from typing import Iterator, Tuple

from tf_agents.drivers.py_driver import PyDriver
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.environments import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies import TFPolicy

from modules.replay import start_replay_server

def create_data_collection_driver(
            agent,
            train_tf_env: TFPyEnvironment,
            train_py_env: PyEnvironment,
            collect_policy: TFPolicy,
            collect_steps_per_iteration: int,
            initial_collection_steps: int,
            batch_size: int,
            replay_buffer_max_length: int,
        ) -> Tuple[PyDriver, Iterator]:
    
    # Start the Replay server
    replay_server_observer, replay_iterator = start_replay_server(agent, replay_buffer_max_length, batch_size)
    
    # Create the controller for the agent in the environment
    driver = lambda max_steps: PyDriver(
                                    env=train_py_env,
                                    policy=PyTFEagerPolicy(collect_policy, use_tf_function=True),
                                    observers=[replay_server_observer],
                                    max_steps=max_steps
                                )

    # Populate the Replay server with an initial set of game plays
    print("\nPopulating the Replay server with an initial set of {} environment steps".format(initial_collection_steps))
    time_step = train_py_env.reset()
    policy_state = agent.policy.get_initial_state(train_tf_env.batch_size)
    driver(initial_collection_steps).run(time_step, policy_state)

    # Create the main driver to be used during training
    data_collection_driver = driver(collect_steps_per_iteration)
    
    return (data_collection_driver, replay_iterator)