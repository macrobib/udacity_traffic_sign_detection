import tensorflow as tf
from tensorflow.contrib.layers import flatten


class layers:
    """Generic class to implement the layers of the network"""

    def __init__(weights, biases,shape,prob, network="standard"):
        """Construct the network based on the input."""
        self.weights = weights # Weights for the netwrk.
        self.biases = biases # Biases for the network.
        self.network=network # Network type.
        self.shape = shape   # Shape of the input and output.
        self.prob = prob     # Dropout probability.
        self.parameters = {} # Parameter list for the layers.
        self.beta = 0.001
        self.early_stopping = False


    def initialize_parameters():
        """Create tensorflow parameters for the network."""
        self.parameters['input'] = tf.placeholder(
                tf.float32, [None, self.shape[0]])
        self.parameters['output'] = tf.placeholder(
                tf.float32, [None, self.shape[1]])

    def early_stopping(val):
        """Enable/Disable early stopping of the network."""
        self.early_stopping = val

    def conv_layer(data, weight, bias, stride=[1, 1, 1, 1],maxpool=True,dropout=False, batch=False):
        """Create a single convolution layer."""
        conv = tf.nn.conv2d(data, weight, strides=stride)
        conv = tf.nn.bias_add(conv, bias)
        if maxpool == True:
            conv = tf.nn.max_pool(conv)
        conv = tf.nn.relu(conv) #TODO: class condition to vary rectification.
        if batch == True:
            conv = tf.contrib.layers.batch_norm(conv, is_training=is_training, trainable=True)
        if dropout == True:
            conv = tf.nn.dropout(conv, prob)
        return conv

    def fully_connected_layer(data, weight, bias, relu=True):
        """Create a fully connected layer."""
        ful_conn = tf.add(tf.matmul(data, weight), bias)
        if relu == True:
            ful_conn = tf.nn.dropout(ful_conn)
            ful_conn = tf.nn.relu(ful_conn)
        return ful_conn

    def regularize(data):
        """Regularizer"""
        keys = self.weights.keys()
        reg = tf.nn.l2_loss(self.weights[keys[0]])
        for val in keys[1:]:
            reg += tf.nn.l2_loss(self.weights[val])
        cost = self.beta * reg
        return cost

    def Sermanet():
        # Sermanet like approach.
        strides = [1, 1, 1, 1]
        layer_1_conv = conv_layer(x, self.weights['w1'], self.biases['b1'], stride, True, False, True )
        flatten_1    = flatten(layer_1_conv)
        layer_2_conv = conv_layer(layer_1_conv, self.weights['w2'], self.biases['b2'], stride, True, False, True )
        flatten_2    = flatten(layer_2_conv)
        concat_layer = tf.concat(1, [flatten_1, flatten_2])
        fc1          = fully_connected_layer(concat_layer, self.weights['w3'], self.biases['b3'], True)
        fc2          = fully_connected_layer(concat_layer, self.weights['w4'], self.biases['b4'], False)
        cost         = regularizer(fc2)
        optimizer    = tf.train.AdapOptimizer(learning_rate=0.001).minimize(cost)
        return (cost, optimizer)

    def Standard():
        # A Standard CNN Network similar to lenet model.
        stride = [1, 1, 1, 1]
        layer_1_conv = conv_layer(x, self.weights['w1'], self.biases['b1'], stride, True, False, True )
        layer_2_conv = conv_layer(layer_1_conv, self.weights['w2'], self.biases['b2'], stride, True, False, True )

        # layer 3 - First fully connected.
        fc0 = flatten(layer_2_conv)
        #fc0 = tf.reshape(layer_2_conv, [-1, weights['w3'].get_shape().as_list()[0]])
        fc1   = fully_connected_layer(fc0, self.weights['w3'], self.biases['b3'], True)
        fc2   = fully_connected_layer(fc1, self.weights['w4'], self.biases['b4'], True)
        fc3   = fully_connected_layer(fc2, self.weights['w5'], self.biases['b5'], True)
        final = fully_connected_layer(fc3, self.weights['w6'], self.biases['b6'], False)
        cost  = regularize(final)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        return (cost, optimizer)

    def OneCNN():
        # OneCNN Based network.
        stride = [1, 1, 1, 1]
        layer_1_conv = conv_layer(x, self.weights['w1'], self.biases['b1'], stride, True, False, True)
        flatten_1    = flatten(layer_1_conv)
        layer_2_conv = conv_layer(x, self.weights['w2'], self.biases['b2'], stride, True, False, True)
        flatten_2    = flatten(layer_2_conv)
        layer_3_conv = conv_layer(x, self.weights['w3'], self.biases['b3'], stride, True, False, True)
        flatten_3    = flatten(layer_3_conv)
        concat_layer = tf.concat(1, [flatten_1, flatten_2, flatten_3])
        fc1          = fully_connected_layer(concat_layer, self.weights['w4'], self.biases['b4'], True)
        fc2          = fully_connected_layer(concat_layer, self.weights['w5'], self.biases['b5'], True)
        fc3          = fully_connected_layer(concat_layer, self.weights['w6'], self.biases['b6'], False)
        cost         = regularize(fc3)
        optimizer    = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        return (cost, optimizer)

    def model():
        """Compose a single network model."""
        if self.network == 'standard':
            # Standard CNN Network.
        elif self.network == 'onecnn':
            # OneCNN architecture for traffic sign detection.
        elif self.network == 'sermanet':
            # Sermanet network for traffic sign detection.
        else:
            pass
        return network






