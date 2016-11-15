import tensorflow as tf

def transfer_with_adapter(dim, inputs, transfer, keep_prob, output_size, inputs_size=None, activation_fn=None):
    inputs_size = inputs.get_shape(dim) if inputs_size is None else inputs_size
    dropped_transfer = tf.nn.dropout(transfer, keep_prob)
    transfer_scalar = tf.get_variable("transfer_scalar", dtype=tf.float32, initializer=1e-1)
    adapted_transfer = tf.contrib.layers.fully_connected(dropped_transfer * transfer_scalar, inputs_size,
                                                         activation_fn=tf.tanh,
                                                         weights_initializer=None,
                                                         scope="adapter")
    if isinstance(inputs, list):
        concat = tf.concat(dim, [adapted_transfer] + inputs)
    else:
        concat = tf.concat(dim, [adapted_transfer, inputs])

    output = tf.contrib.layers.fully_connected(concat, output_size,
                                               activation_fn=activation_fn,
                                               weights_initializer=None,
                                               scope="projection")

    return output


def gated_transfer(dim, inputs, transfer, keep_prob, output_size, inputs_size=None, activation_fn=None):
    if isinstance(inputs, list):
        inputs = tf.concat(dim, inputs)

    inputs_size = inputs.get_shape()[dim].value if inputs_size is None else inputs_size
    dropped_transfer = tf.nn.dropout(transfer, keep_prob)
    transfer_scalar = tf.get_variable("transfer_scalar", dtype=tf.float32, initializer=1e-1)
    adapted_transfer = tf.contrib.layers.fully_connected(dropped_transfer * transfer_scalar, inputs_size,
                                                         activation_fn=tf.tanh,
                                                         weights_initializer=None,
                                                         scope="adapter")

    input_projection = tf.contrib.layers.fully_connected(inputs, output_size,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         scope="input_projection")

    transfer_projection = tf.contrib.layers.fully_connected(adapted_transfer, output_size,
                                                            activation_fn=None,
                                                            weights_initializer=None,
                                                            scope="transfer_projection")

    gate = tf.contrib.layers.fully_connected(tf.concat(2, [input_projection, transfer_projection]), output_size,
                                             activation_fn=tf.sigmoid,
                                             weights_initializer=None,
                                             scope="gate")

    output = gate * input_projection + (1-gate) * transfer_projection
    if activation_fn is not None:
        output = activation_fn(output)

    return output
