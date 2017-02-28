import numpy as np
import tensorflow as tf

class Network():
    def __init__(self, size=19):
        self.size = size
        self._init_graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _init_graph(self):
        size = self.size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder('float',
                    [None, size, size, 1], name='input')
            self.y = self._conv2d(self.x, 1, label='_final')

    def _conv2d(self, x, k_out, size=3, stride=1, label=''):
        k_in = x.get_shape()[-1].value
        shape = [size, size, k_in, k_out]
        strides = [1, stride, stride, 1]
        with self.graph.as_default():
            W_init = tf.truncated_normal(shape, stddev=0.1)
            W = tf.Variable(W_init, 'W_'+label)
            b_init = tf.constant(0.1, shape=[k_out])
            b = tf.Variable(b_init, 'b_'+label)
            y = tf.nn.conv2d(x, W, strides, 'SAME')
        return y

    def _linear(self, x, d_out, label=''):
        """ Create linear layer on x with output dim d_out. """
        d_in = x.get_shape()[-1].value
        with self.graph.as_default():
            W_init = tf.truncated_normal([d_in, d_out], stddev=0.1)
            W = tf.Variable(W_init, 'W_'+label)
            b_init = tf.constant(0.1, shape=[d_out])
            b = tf.Variable(b_init, 'b_'+label)
            y = tf.matmul(x, W) + b
        return y

    def run(self, board, color):
        feed_dict = {self.x: color*board[None, :, :, None]}
        y = self.sess.run(self.y, feed_dict=feed_dict).squeeze()
        return np.unravel_index(y.argmax(), y.shape)

class Bot():
    def __init__(self, engine, color):
        self.engine = engine
        self.color = color
        self.net = Network(self.engine.size)

    def act(self):
        self.net.run(self.engine.board, self.color)
        while True:
            move = tuple(np.random.randint(self.engine.size, size=2))
            if self.engine.legal(move, self.color):
                return move

if __name__ == '__main__':
    net = Network()
