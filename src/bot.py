import os, re
import string
import numpy as np
import tensorflow as tf

from engine import *
from data_utils import *

DIRNAME = './bots/{}'
CKPT_BASE = './bots/{}/checkpoint'
CKPT_FULL = './bots/{}/checkpoint-{}'
CKPT_STEP_REGEX = r'checkpoint-(\d+).meta'
SAVE_INTERVAL = 500

class Bot():
    def __init__(self, name=None, global_step=None, size=19):
        # Process arguments.
        if name is None:
            assert global_step is None
            name = ''.join([string.lowercase[i] for i in
                    np.random.choice(26, size=6)])
            ckpt = None
        else:
            dirname = DIRNAME.format(name)
            assert os.path.exists(dirname)
            if global_step is None:
                files = os.listdir(dirname)
                step_regex = re.compile(CKPT_STEP_REGEX)
                steps = step_regex.findall(''.join(files))
                global_step = max(map(int, steps))
            ckpt = CKPT_FULL.format(name, global_step)

        # Store processed arguments.
        self.name = name
        self.global_step = global_step
        self.size = size

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
            if ckpt is None:
                self.sess.run(tf.global_variables_initializer())
                self.global_step = 0
            else:
                self.saver.restore(self.sess, ckpt)

    def _build_graph(self):
        with self.graph.as_default():
            # Input variable.
            self.x = tf.placeholder(tf.float32,
                    [None, self.size, self.size, NUM_FEATURES])
            # Add on the layers.
            # TODO: Config instead of hardcoded?
            y = self._conv2d(self.x, 128, 5, tf.nn.crelu)
            for layers in [5, 5, 5]:
                y = self._dense_block(y, layers, 12)
                num_outputs = y.get_shape()[3] // 2
                y = self._conv2d(y, num_outputs, 1)
            y = self._conv2d(y, 1, 1, activation_fn=None)
            y = tf.reshape(y, [-1, self.size, self.size])
            bias = tf.Variable(-0.1*tf.zeros([self.size, self.size]))
            self.y = y + bias
            # Set up loss.
            self.labels = tf.placeholder(tf.int32, [None])
            y_ = tf.reshape(self.y, [-1, self.size*self.size])
            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=y_, labels=self.labels))
            # Add regularization.
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = self.loss + reg_losses
            # Build train op.
            solver = tf.train.AdamOptimizer()
            self.minimize = solver.minimize(loss)
            # Set up saving.
            self.saver = tf.train.Saver()

    def _conv2d(self, x, num_outputs, size, activation_fn=tf.nn.elu):
        norm_fn = tf.contrib.layers.layer_norm
        init = tf.contrib.layers.xavier_initializer_conv2d()
        l2_reg = tf.contrib.layers.l2_regularizer(1e-4)
        return tf.contrib.layers.conv2d(x, num_outputs=num_outputs,
                kernel_size=size, padding='SAME', normalizer_fn=norm_fn,
                weights_initializer=init, weights_regularizer=l2_reg,
                activation_fn=activation_fn)

    def _dense_block(self, x, layers, growth_rate):
        for l in range(layers):
            bottleneck = self._conv2d(x, 4*growth_rate, 1)
            y = self._conv2d(bottleneck, growth_rate, 3)
            x = tf.concat([x, y], axis=3)
        return x

    def forward(self, image, use_symmetry=True):
        # Process image.
        shape = image.shape
        rank = len(shape)
        if use_symmetry:
            assert rank == 2
            images = d8_forward(image)
        elif rank == 2:
            images = image[None, ...]
        else:
            images = image
        x = input_features(images) # rank == 4.
        # Run forward pass.
        y = self.sess.run(self.y, feed_dict={self.x: x})
        if use_symmetry:
            y = d8_backward(y)
        elif rank == 2:
            y = y[0]
        return y

    def gen_move(self, engine, color):
        if engine.last_move == PASS:
            return PASS
        image = color*engine.board
        y = self.forward(image)
        idxs = np.argsort(-y, axis=None)
        for idx in idxs:
            move = (idx // self.size, idx % self.size)
            if engine.legal(move, color):
                return move
        return PASS

    def train(self, images, labels, batch_size=16, epochs=1.0):
        N = images.shape[0]
        iters = int(epochs * N / batch_size)
        for i in range(iters):
            # Pick random minibatch and train.
            batch = np.random.choice(N, batch_size)
            x, y = augment_data(images[batch], labels[batch])
            loss, _ = self.sess.run([self.loss, self.minimize],
                    feed_dict={self.x: x, self.labels: y})
            print "{}, loss: {:f}, batch: {}/{}, step: {}".format(
                    self.name, loss, i, iters, self.global_step)
            # Update global step and save periodically.
            self.global_step += 1
            if self.global_step % SAVE_INTERVAL == 0:
                self._save()

    def _save(self):
        # Initialize the save directory if it hasn't been created.
        dirname = DIRNAME.format(self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # Save the current state of the session.
        self.saver.save(self.sess, CKPT_BASE.format(self.name),
                global_step=self.global_step)
