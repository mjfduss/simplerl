import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import TimeStep
from tf_agents.typing.types import NestedTensorSpec
from tf_agents.utils import common

from modules.network import QRnnCfcNetwork
from modules.optimizer import make_optimizer

def make_agent(
            time_step_spec: TimeStep, 
            observation_spec, 
            action_spec: NestedTensorSpec, 
            hparams: dict
        ) -> dqn_agent.DdqnAgent:
    
    agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_network=QRnnCfcNetwork(
                observation_spec,
                action_spec,
                hparams['cfc_params']
            ),
            optimizer=make_optimizer(hparams['optimizer'], hparams['learning_rate']),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0)
        )
    agent.initialize()

    return agent