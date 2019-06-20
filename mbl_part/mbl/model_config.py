import tensorflow as tf
import numpy as np

def make_mlp(inputs_ph, num_fc=2, num_fwd_hidden=500, use_layer_norm=True):
    x = inputs_ph
    with tf.variable_scope('phi') as scope:
        for i in range(num_fc):
            x = tf.contrib.layers.fully_connected(x, num_outputs=num_fwd_hidden, activation_fn=None)
            if use_layer_norm:
                x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.relu(x)
    return x

def get_make_mlp_model(num_fc, num_fwd_hidden, layer_norm=True):
    def _thunk(inputs_ph):
        return make_mlp(inputs_ph, num_fc, num_fwd_hidden, use_layer_norm=layer_norm)
    return _thunk

