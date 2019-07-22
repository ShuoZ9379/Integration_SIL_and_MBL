import tensorflow as tf
import numpy as np

from baselines import logger

def make_placeholder(num_input, num_output):
    inputs = tf.placeholder(tf.float32, (None, num_input))
    targets = tf.placeholder(tf.float32, (None, num_output))
    return inputs, targets

class NN(object):
    def __init__(self, num_input, num_output, make_model, lr=1e-3, optimizer=tf.compat.v1.train.AdamOptimizer, **kwargs):        
        logger.log('NN', locals())
        self.inputs_ph, self.targets_ph = make_placeholder(num_input, num_output)
        phi = make_model(self.inputs_ph)
        self.pred = self._make_pred(phi, num_output)
        
        self.loss = tf.losses.mean_squared_error(labels=self.targets_ph, predictions=self.pred)
        self.train_op = optimizer(learning_rate=lr).minimize(self.loss) 
       
    def _make_pred(self, phi, num_output):
        pred = tf.contrib.layers.fully_connected(phi, num_outputs=num_output, activation_fn=tf.identity)
        return pred

    def train(self, inputs, targets):
        sess = tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.inputs_ph: inputs, self.targets_ph: targets})
        return loss

    def predict(self, inputs):      
        sess = tf.get_default_session()
        pred = sess.run(self.pred, feed_dict={self.inputs_ph: inputs})
        return pred

    def eval(self, inputs, targets):
        sess = tf.get_default_session()
        loss = sess.run(self.loss, feed_dict={self.inputs_ph: inputs, self.targets_ph: targets})
        return loss



        
    
