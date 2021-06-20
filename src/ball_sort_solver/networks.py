import tensorflow as tf


def build_network(
    input_size,
    inner_layers_neurons,
    dropout_rate,
    inner_layers,
    inner_activation,
    output_size,
    output_activation,
    name
):
    layers = [
        tf.keras.layers.InputLayer(input_shape=input_size)
    ]
    for _ in range(inner_layers):
        layers.append(tf.keras.layers.Dense(inner_layers_neurons, activation=inner_activation))
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(output_size, activation=output_activation))
    return tf.keras.models.Sequential(layers=layers, name=name)
