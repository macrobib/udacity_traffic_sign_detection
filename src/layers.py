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


    def initialize_parameters():
        """Create tensorflow parameters for the network."""
        self.parameters['input'] = tf.placeholder(
                tf.float32, [None, self.shape[0]])
        self.parameters['output'] = tf.placeholder(
                tf.float32, [None, self.shape[1]])


    def conv_layer(data, weight,stride=[1, 1, 1, 1],maxpool=True,dropout=False):
        """Create a single convolution layer."""
        conv = tf.nn.conv2d(data, weight, strides=stride)
        if maxpool == True:
            conv = tf.nn.max_pool(conv)
        
        conv = tf.nn.relu(conv) #TODO: class condition to vary rectification.
        
        if dropout == True:
            conv = tf.nn.dropout(conv, prob)
        return conv
    
    def fully_connected_layer(data, kernel, biases, dropout=False):
        """Create a fully connected layer."""
        ful_conn = tf.add(tf.matmul(data, kernel), biases)
        if dropout == True:
            ful_conn = tf.nn.dropout(ful_conn)
        return ful_conn


    def normalize_layer(data, type='BATCH'):
        """Create a normalization layer."""
        norm = data
        if type=='BATCH':
            norm = tf.nn.batch_normalization(norm)
        elif type == 'L2':
            # L2 Normalize.
        else:
            # Not defined.
        return norm

def Sermanet():
    # Sermanet model.

def Standard():
    # A Standard CNN Network.

def OneCNN():
    # One CNN Based network.
    layer_1_conv = conv_layer(data)
    layer_2_conv = conv_layer(layer_1_conv)


    def model():
        """Compose a single network model."""
        if self.network == 'standard':
            # Standard CNN Network.
        elif self.network == 'onecnn':
            # OneCNN architecture for traffic sign detection.
        elif self.network == 'sermanet':
            # Sermanet network for traffic sign detection.
        else:
            # TODO: Ciresan network.
        return network






