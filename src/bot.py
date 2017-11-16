import os, re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from engine import *
from data_utils import *


DIR_FORMAT = './bots/{}'
SAVE_INTERVAL = 500


class Bot(nn.Module):
    def __init__(self, name=None, global_step=0, size=19):
        super(Bot,self).__init__()

        # Process arguments.
        if name is None:
            assert global_step is None
            name = ''.join([string.lowercase[i] for i in np.random.choice(26, size=4)])

        # Store processed arguments.
        self.name = name
        self.global_step = global_step
        self.size = size

        # Net variables
        self.x = torch.Tensor()

    def forward(self, image):
        from IPython import embed; embed()
        # Process image.
        size = image.size()
        rank = len(size)
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
            print("{}, loss: {:f}, batch: {}/{}, step: {}".format(
                    self.name, loss, i, iters, self.global_step))
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
