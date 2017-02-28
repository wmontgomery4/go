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
            # Inputs.
            self.x = tf.placeholder('float',
                    [None, size, size, 1], name='input')
            x_ = tf.reshape(self.x, [-1, size*size])
            y_ = self._linear(x_, size*size, label='_final')
            y_ = tf.nn.softmax(y_)
            self.y = tf.reshape(y_, [-1, size, size, 1])

    def _linear(self, x, d_out, label=''):
            """ Create linear layer on x with output dim d_out. """
            d_in = x.get_shape()[-1].value
            W_init = tf.truncated_normal([d_in, d_out], stddev=0.1)
            W = tf.Variable(W_init, 'W_'+label)
            b_init = tf.constant(0.1, shape=[d_out])
            b = tf.Variable(b_init, 'b_'+label)
            y = tf.matmul(x, W) + b
            return y

    def run(self, board):
        feed_dict = {self.x: board[None, :, :, None]}
        y = self.sess.run(self.y, feed_dict=feed_dict).squeeze()
        return np.unravel_index(y.argmax(), y.shape)

class Bot():
    def __init__(self, engine, color):
        self.engine = engine
        self.color = color
        self.net = Network(self.engine.size)

    def act(self):
        self.net.run(self.engine.board)
        while True:
            move = tuple(np.random.randint(self.engine.size, size=2))
            if self.engine.legal(move, self.color):
                return move
