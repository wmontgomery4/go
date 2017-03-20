import os
import string
import numpy as np
import tensorflow as tf

from engine import *
from data_utils import *

DIRNAME = './bots/{}'
CKPT_BASE = './bots/{}/checkpoint'
CKPT_FULL = './bots/{}/checkpoint-{}'

class Bot():
    def __init__(self, size=19, name=None, global_step=None):
        self.size = size
        self.name = name
        self.global_step = global_step

        # Create new bot or make sure checkpoint exists.
        if self.name is None:
            assert self.global_step is None
            self.name = ''.join([string.lowercase[i] for i in
                    np.random.choice(26, size=6)])
        else:
            assert self.global_step is not None
            assert os.path.exists(DIRNAME.format(name))

        # Graph variables.
        self.graph = tf.Graph()
        self.x = None # (?, size, size, NUM_FEATURES)
        self.y = None # (?, size, size)
        self.labels = None # (?,)
        self.loss = None
        self.minimize = None
        self.saver = None
        self._build_graph()

        # Begin the session.
        with self.graph.as_default():
            self.sess = tf.Session()
            if self.global_step is None:
                self.sess.run(tf.global_variables_initializer())
                self.global_step = 0
            else:
                self.saver.restore(self.sess,
                        CKPT_FULL.format(self.name, self.global_step))

    def _build_graph(self):
        with self.graph.as_default():
            # Input variable.
            self.x = tf.placeholder(tf.float32,
                    [None, self.size, self.size, NUM_FEATURES])
            # Add on the layers.
            # TODO: Config instead of hardcoded?
            y = self._conv2d(self.x, 64, 5)
            y = tf.concat([self.x, y], axis=3)
            for i in range(3):
                y = self._dense_block(y, 4, 24)
                num_outputs = y.get_shape()[3] // 2
                y = self._conv2d(y, num_outputs, 1)
                y = tf.concat([self.x, y], axis=3)
            y = self._conv2d(y, 1, 1, activation_fn=None)
            y = tf.reshape(y, [-1, self.size, self.size])
            b = tf.Variable(-0.1*tf.zeros([self.size, self.size]))
            self.y = y + b
            # Set up training.
            # TODO: Add regularization.
            self.labels = tf.placeholder(tf.int32, [None])
            y_ = tf.reshape(self.y, [-1, self.size*self.size])
            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=y_, labels=self.labels))
            solver = tf.train.AdamOptimizer()
            self.minimize = solver.minimize(self.loss)
            # Set up saving.
            self.saver = tf.train.Saver()

    def _conv2d(self, x, num_outputs, size, activation_fn=tf.nn.elu):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        norm_fn = tf.contrib.layers.layer_norm
        return tf.contrib.layers.conv2d(x, num_outputs=num_outputs,
                kernel_size=size, padding='SAME', normalizer_fn=norm_fn,
                weights_initializer=init, activation_fn=activation_fn)

    def _dense_block(self, x, layers, growth_rate):
        for l in range(layers):
            bottleneck = self._conv2d(x, 4*growth_rate, 1)
            y = self._conv2d(bottleneck, growth_rate, 3)
            x = tf.concat([x, y], axis=3)
        return x

    def gen_move(self, engine, color):
        if engine.last_move == PASS:
            return PASS
        x = engine.get_features(color)
        y = self._forward(x)
        idxs = np.argsort(-y, axis=None)
        for idx in idxs:
            move = (idx // self.size, idx % self.size)
            if engine.legal(move, color):
                return move
        return PASS

    def _forward(self, x, use_symmetry=True):
        if use_symmetry:
            x = d8_forward(x)
        else:
            x = x[None, :, :, :]
        y = self.sess.run(self.y, feed_dict={self.x: x})
        if use_symmetry:
            y = d8_backward(y)
        return y.squeeze()

    def train(self, images, labels, batch_size=16, epochs=1.0):
        N = images.shape[0]
        iters = int(epochs * N / batch_size)
        for i in range(iters):
            # Pick random minibatch and train.
            idx = np.random.choice(N, batch_size, replace=False)
            idx.sort()
            x = images[idx.tolist()]
            y = labels[idx.tolist()]
            loss, _ = self.sess.run([self.loss, self.minimize],
                    feed_dict={self.x: x, self.labels: y})
            print "Batch {}/{}, Loss: {}".format(i, iters, loss)
            # Update global step and save periodically.
            self.global_step += 1
            if self.global_step % 25 == 0:
                self._save()

    def _save(self):
        # Initialize the save directory if it hasn't been created.
        dirname = DIRNAME.format(self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # Save the current state of the session.
        self.saver.save(self.sess, CKPT_BASE.format(self.name),
                global_step=self.global_step)
