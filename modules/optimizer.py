import tensorflow as tf

def make_optimizer(hparams: dict):
    
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        hparams['base_lr'], hparams['num_iterations'], hparams['decay_lr']
    )
    opt = (
        tf.keras.optimizers.Adam
        if hparams["optimizer"] == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate_fn, clipnorm=hparams["clipnorm"])
  