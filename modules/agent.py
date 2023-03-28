import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.trajectories import TimeStep
from tf_agents.typing.types import NestedTensorSpec

from modules.tf_cfc import MixedCfcCell

def make_agent(
            time_step_spec: TimeStep, 
            observation_spec, 
            action_spec: NestedTensorSpec, 
            hparams: dict
        ) -> ppo_agent.PPOAgent:
    
    agent = ppo_agent.PPOAgent(
        time_step_spec,
        action_spec,
        optimizer=_make_optimizer(hparams['optimizer'], hparams['learning_rate']),
        actor_net=actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                    observation_spec, 
                    action_spec,
                    rnn_construction_fn=_make_rnn_function,
                    rnn_construction_kwargs={'cfc_params': hparams['cfc_params']}
                ),
        value_net=value_rnn_network.ValueRnnNetwork(observation_spec)
    )
    agent.initialize()

    return agent

def _make_rnn_function(cfc_params: dict):
    create_cell = lambda: MixedCfcCell(units=cfc_params['size'][0], hparams=cfc_params)

    if cfc_params['size'][1] == 1:
        cell = create_cell()
    else:
        cell = tf.keras.layers.StackedRNNCells(
            [create_cell()
                for _ in range(0, cfc_params['size'][1])])
        
    rnn_network = dynamic_unroll_layer.DynamicUnroll(cell)
    return rnn_network


def _make_optimizer(opt_type: str, learning_rate: float):
    
    opt = (
        tf.keras.optimizers.Adam
        if opt_type == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate=learning_rate)

    return optimizer