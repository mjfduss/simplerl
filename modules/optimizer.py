import tensorflow as tf

def make_optimizer(opt_type: str, learning_rate: float):
    
    opt = (
        tf.keras.optimizers.Adam
        if opt_type == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate=learning_rate)

    return optimizer