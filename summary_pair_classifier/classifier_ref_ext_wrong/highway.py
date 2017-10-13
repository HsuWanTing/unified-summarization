import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
import pdb

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        with tf.variable_scope('output_lin_%d' % idx):
            output = f(_linear(output, size, 0))
            tf.get_variable_scope().reuse_variables()
            transform_gate = tf.sigmoid(_linear(input_, size, 0) + bias)
            carry_gate = 1. - transform_gate
            output = transform_gate * output + carry_gate * input_
    return output
