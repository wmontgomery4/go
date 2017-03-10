import numpy as np
import tensorflow as tf

from engine import *
from data_utils import *


class Bot():
    def __init__(self, size=19, arch=[(64, 3, 1)]*6):
        self.size = size
        self.arch = arch
        self._init_graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input variable.
            self.x = tf.placeholder(tf.float32,
                    [None, self.size, self.size, NUM_FEATURES])
            # Add on the layers.
            y = self.x
            for (k_out, size, rate) in self.arch:
                y = self._conv2d(y, k_out, size, rate, tf.nn.relu)
            y = self._conv2d(y, 1, 1, 1) # (?, size, size, 1)
            self.y = tf.reshape(y, [-1, self.size, self.size])
            # Set up training.
            y_ = tf.reshape(self.y, [-1, self.size*self.size])
            self.labels = tf.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=y_, labels=self.labels))
            solver = tf.train.AdamOptimizer()
            self.train_op = solver.minimize(self.loss)
            return

    def _conv2d(self, x, k_out, size, rate, activation_fn=None):
        with self.graph.as_default():
            init = tf.contrib.layers.xavier_initializer_conv2d()
            norm_fn = tf.contrib.layers.layer_norm
            y = tf.contrib.layers.conv2d(x, num_outputs=k_out,
                    kernel_size=size, rate=rate, padding='SAME',
                    weights_initializer=init, normalizer_fn=norm_fn,
                    activation_fn=activation_fn)
        return y

    def gen_move(self, engine, color):
        if engine.last_move == PASS:
            return PASS
        images = d8_forward(engine.get_features(color))
        y = self.sess.run(self.y, feed_dict={self.x: images})
        y = d8_backward(y)
        idxs = np.argsort(-y, axis=None)
        for idx in idxs:
            move = (idx // self.size, idx % self.size)
            if engine.legal(move, color):
                return move
        return PASS

    def train(self, images, labels, batch_size=64):
        N = images.shape[0]
        # Shuffle data.
        idxs = np.random.permutation(N)
        iters = N // batch_size
        for i in range(iters):
            idx = idxs[i*batch_size:(i+1)*batch_size]
            x = images[idx]
            y = labels[idx]
            loss, _ = self.sess.run([self.loss, self.train_op],
                    feed_dict={self.x: x, self.labels: y})
            print "Batch {}/{}, Loss: {}".format(i, iters, loss)
