import numpy as np
import tensorflow as tf

from engine import *


class Network():
    def __init__(self, size=19):
        self.size = size
        self._init_graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            shape = [None, self.size, self.size, 1]
            self.x = tf.placeholder('float', shape, name='input')
            self.y = self._conv2d(self.x, 1, label='_final')
            # TODO: Mask out non-empty spaces?

    def _conv2d(self, x, k_out, size=3, stride=1, label=''):
        k_in = x.get_shape()[-1].value
        shape = [size, size, k_in, k_out]
        strides = [1, stride, stride, 1]
        with self.graph.as_default():
            W_init = tf.truncated_normal(shape, stddev=0.1)
            W = tf.Variable(W_init, 'W_'+label)
            b_init = tf.constant(0.1, shape=[k_out])
            b = tf.Variable(b_init, 'b_'+label)
            y = b + tf.nn.conv2d(x, W, strides, 'SAME')
        return y

    def _linear(self, x, d_out, label=''):
        """ Create linear layer on x with output dim d_out. """
        d_in = x.get_shape()[-1].value
        with self.graph.as_default():
            W_init = tf.truncated_normal([d_in, d_out], stddev=0.1)
            W = tf.Variable(W_init, 'W_'+label)
            b_init = tf.constant(0.1, shape=[d_out])
            b = tf.Variable(b_init, 'b_'+label)
            y = b + tf.matmul(x, W)
        return y

    def forward(self, board, color):
        feed_dict = {self.x: color*board[None, :, :, None]}
        return self.sess.run(self.y, feed_dict=feed_dict).squeeze()

class Bot():
    def __init__(self, size=19):
        self.size = size
        self.net = Network(size)

    def gen_move(self, engine, color):
        y = self.net.forward(engine.board, color)
        soft = np.exp(y)
        idxs = np.argsort(-soft, axis=None)
        for idx in idxs:
            move = np.unravel_index(idx, y.shape)
            if engine.legal(move, color):
                return move

    def act(self, engine, color):
        move = self.gen_move(engine, color)
        engine.play(move, color)


if __name__ == '__main__':
    engine = Engine()
    bot = Bot()
