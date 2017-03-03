import numpy as np
import tensorflow as tf

from engine import *


class Bot():
    def __init__(self, size=19, arch=[(5, 32)]*10):
        self.size = size
        self.arch = arch
        self._init_graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Board input variable.
            self.x = tf.placeholder(tf.float32,
                    [None, self.size, self.size, 1])
            # Use dropout regularization.
            self.dropout = tf.placeholder(tf.float32)
            # Add on the layers.
            y = self.x
            for l, (size, k_out) in enumerate(self.arch):
                y = self._conv2d(y, size, k_out, label=str(l))
                y = tf.nn.relu(y)
            y = tf.nn.dropout(y, self.dropout)
            y = self._conv2d(y, 1, 1, label='final_conv')
            self.y = tf.reshape(y, [-1, self.size*self.size])
            # Set up training.
            self.learning_rate = tf.placeholder(tf.float32)
            self.labels = tf.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.y, labels=self.labels))
            solver = tf.train.MomentumOptimizer(self.learning_rate, 0.9,
                    use_nesterov=True)
            self.train_op = solver.minimize(self.loss)
            return

    def _conv2d(self, x, size, k_out, label=''):
        k_in = x.get_shape()[-1].value
        shape = [size, size, k_in, k_out]
        with self.graph.as_default():
            W_init = tf.truncated_normal(shape, stddev=0.1)
            W = tf.Variable(W_init, name='W_'+label)
            b_init = tf.constant(0.1, shape=[k_out])
            b = tf.Variable(b_init, name='b_'+label)
            y = b + tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
        return y

    def _linear(self, x, d_out, label=''):
        d_in = x.get_shape()[-1].value
        with self.graph.as_default():
            W_init = tf.truncated_normal([d_in, d_out], stddev=0.1)
            W = tf.Variable(W_init, name='W_'+label)
            b_init = tf.constant(0.1, shape=[d_out])
            b = tf.Variable(b_init, name='b_'+label)
            y = b + tf.matmul(x, W)
        return y

    def gen_move(self, engine, color):
        if engine.last_move == PASS:
            return PASS
        x = color*engine.board[None, :, :, None]
        y = self.sess.run(self.y,
                feed_dict={self.x: x, self.dropout: 1.0})
        idxs = np.argsort(-y, axis=None)
        for idx in idxs:
            move = (idx // self.size, idx % self.size)
            if engine.legal(move, color):
                return move
        return PASS

    def train(self, boards, labels, batch_size=256, epochs=1, lr=0.1):
        N = boards.shape[0]
        # Shuffle data.
        idxs = np.random.permutation(N)
        x = boards[idxs, :, :, None]
        labels = labels[idxs]
        # Run through epochs.
        iters = int(epochs * (N / batch_size))
        for i in range(iters):
#            start = i*batch_size % N
#            idx = slice(start, start+batch_size)
            idx = np.random.choice(N, size=batch_size)
            loss, _ = self.sess.run([self.loss, self.train_op],
                    feed_dict={self.x: x[idx], self.labels: labels[idx],
                            self.dropout:0.5, self.learning_rate:lr})
            print "Batch {}, Loss: {}".format(i, loss)
