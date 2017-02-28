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
            self.input_tensor = tf.placeholder('float',
                    [None, size, size, 1], name='input')
            self.input_flat = tf.reshape(self.input_tensor, [-1, size*size])
            # TODO: Hidden layers.
            # Output.
            init = tf.truncated_normal([size*size, size*size], stddev=0.1)
            W = tf.Variable(init, 'W_final')
            init = tf.constant(0.1, shape=[size*size])
            b = tf.Variable(init, 'b_final')
            out = tf.matmul(self.input_flat, W) + b
            out = tf.nn.softmax(out)
            self.output = tf.reshape(out, [-1, size, size, 1])

    def run(self, board):
        feed_dict = {self.input_tensor: board[None, :, :, None]}
        out = self.sess.run(self.output, feed_dict=feed_dict)
        out = out.squeeze()
        return np.unravel_index(out.argmax(), out.shape)

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
