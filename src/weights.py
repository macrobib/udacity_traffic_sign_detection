# Seperate out weights from the layers file.

import tensorflow as tf

class weights:
    """class to store weights"""
    def __init__(self, network, grascale=False):
        if network == 'sermanet':
            self.weights, self.biases = sermanet_params(grayscale)
        elif network == 'oneCNN':
            self.weights, self.biases = onCNN_params(grayscale)
        else:
            self.weights, self.biases = standard_params(grayscale)


    def sermanet_params(grayscale):
        """Create dictionary of sermanet weights and biases"""
        weights = {}
        biases = {}

        return weights, biases

    def oneCNN_params(grayscale):
        """Create dictionary of onecnn weights and biases."""
        weights = {}
        biases = {}

        return weights, biases

    def standard_params(grayscale):
        """Create dictionary of weights and biases"""
        weights = {}
        biases = {}
        if grayscale == False:
            weights['w1'] = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 6), mean=0., stddev=0.1), name='W1')
        else:
            weights['w1'] = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 6), mean=0., stddev=0.1), name='W1')
        weights['w2'] = tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean=0., stddev=0.1), name='W2')
        weights['w3'] = tf.Variable(tf.truncated_normal(shape=(8*8*16, 400), mean=0., stddev=0.1), name='W3')
        weights['w4'] = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0., stddev=0.1), name='W4')
        wegiths['w5'] = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0., stddev=0.1), name='W5')
        weights['w6'] = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=0., stddev=0.1), name='W6')

        biases['b1'] = tf.Variable(tf.truncated_normal([6], mean=0., stddev=0.1),name='B1')
        biases['b2'] = tf.Variable(tf.truncated_normal([16], mean=0., stddev=0.1),name='B2')
        biases['b3'] = tf.Variable(tf.truncated_normal([400], mean=0., stddev=0.1),name='B3')
        biases['b4'] = tf.Variable(tf.truncated_normal([120], mean=0., stddev=0.1),name='B4')
        biases['b5'] = tf.Variable(tf.truncated_normal([84], mean=0., stddev=0.1),name='B5')
        biases['b6'] = tf.Variable(tf.truncated_normal([43], mean=0., stddev=0.1),name='B6')

        return weights, biases



