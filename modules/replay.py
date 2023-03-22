import reverb

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers.reverb_utils import ReverbAddTrajectoryObserver
from tf_agents.specs import tensor_spec


def start_replay_server(agent, replay_buffer_max_length: int, batch_size: int) -> ReverbAddTrajectoryObserver:

    table_name = 'uniform_table'

    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    )

    get_replay = lambda _ : next(iter(dataset))

    return (rb_observer, get_replay)