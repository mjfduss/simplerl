from typing import Tuple

import tensorflow as tf

from tf_agents.networks import network, q_network, encoding_network
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

from modules.tf_cfc import MixedCfcCell, lecun_tanh


class QRnnCfcNetwork(network.Network):
    """Recurrent Q network with Closed Form Continuous Time Cell"""

    def __init__(
        self,
        input_tensor_spec,
        action_spec,
        cfc_params: dict,
        conv_layer_params:Tuple[int,int,int]=None,
        dtype=tf.float32,
        name='QRnnCfcNetwork',
    ):
        """Creates an instance of `QRnnCfcNetwork`.
        Args:
        input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            observations.
         action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
            actions.
        cfc_params: A dictionary of hyperparamters for the Cfc cell
        conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size, stride).
        dtype: The dtype to use by the layers.
        name: A string representing name of the network.
        """
        q_network.validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1

        q_projection = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2),
            dtype=dtype,
            name='num_action_project/dense')
    
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
        self._input_encoder = encoding_network.EncodingNetwork(
                input_tensor_spec, 
                activation_fn=lecun_tanh, 
                conv_layer_params=conv_layer_params,
                dtype=dtype, 
                kernel_initializer=kernel_initializer
            )

        # Create RNN cell
        create_cell = lambda: MixedCfcCell(units=cfc_params['size'][0], hparams=cfc_params)

        if cfc_params['size'][1] == 1:
            cell = create_cell()
        else:
            cell = tf.keras.layers.StackedRNNCells(
                [create_cell()
                    for _ in range(0, cfc_params['size'][1])])
            
            lstm_network = dynamic_unroll_layer.DynamicUnroll(cell)

        
        counter = [-1]

        def create_spec(size):
            counter[0] += 1
            return tf.TensorSpec(
                    size, 
                    dtype=dtype, 
                    name='network_state_%d' % counter[0]
                )

        state_spec = tf.nest.map_structure(create_spec,
                                        lstm_network.cell.state_size)
        

        super(QRnnCfcNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)

        
        self._lstm_network = lstm_network
        self._output_encoder = [q_projection]

    def call(self,
            observation,
            step_type,
            network_state=(),
            training=False):
        """Apply the network.
        Args:
        observation: A tuple of tensors matching `input_tensor_spec`.
        step_type: A tensor of `StepType.
        network_state: (optional.) The network state.
        training: Whether the output is being used for training.
        Returns:
        `(outputs, network_state)` - the network output and next network state.
        Raises:
        ValueError: If observation tensors lack outer `(batch,)` or
            `(batch, time)` axes.
        """
        num_outer_dims = nest_utils.get_outer_rank(observation,
                                                self.input_tensor_spec)
        if num_outer_dims not in (1, 2):
            raise ValueError(
                'Input observation must have a batch or batch x time outer shape.')

        has_time_dim = num_outer_dims == 2
        if not has_time_dim:
            # Add a time dimension to the inputs.
            observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                                observation)
            step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                                step_type)

        state, _ = self._input_encoder(
            observation, step_type=step_type, network_state=(), training=training)

        network_kwargs = {}
        if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
            network_kwargs['reset_mask'] = tf.equal(step_type,
                                                    time_step.StepType.FIRST,
                                                    name='mask')

        # Unroll over the time sequence.
        output = self._lstm_network(
            inputs=state,
            initial_state=network_state,
            training=training,
            **network_kwargs)

        if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
            state, network_state = output
        else:
            state = output[0]
            network_state = tf.nest.pack_sequence_as(
                self._lstm_network.cell.state_size, tf.nest.flatten(output[1:]))

        for layer in self._output_encoder:
            state = layer(state, training=training)

            if not has_time_dim:
                # Remove time dimension from the state.
                state = tf.squeeze(state, [1])            

        return state, network_state