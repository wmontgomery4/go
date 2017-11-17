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
    def __init__(self, name=None, global_step=None, size=19):
        super(Bot,self).__init__()

        # Create random name
        if name is None:
            assert global_step is None
            idxs = np.random.choice(26, size=4)
            name = ''.join([string.ascii_lowercase[i] for i in idxs])

        # Store processed arguments.
        self.name = name
        self.global_step = global_step
        # TODO: get rid of size?
        self.size = size

        # Initialize weights
        # TODO: config files
        n_layers = 12
        n_channels = 192

        layers = [nn.Conv2d(1, n_channels, 3, padding=1)]
        layers.append(nn.ReLU())
        for i in range(n_layers-2):
            layers.append(nn.Conv2d(n_channels, n_channels, 3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_channels, 1, 3, padding=1))

        self.model = nn.Sequential(*layers)

    def gen_move(self, engine, color):
        # TODO: add passing move
        # TODO: add ko/handicap input
        if engine.last_move == PASS:
            return PASS

        # TODO? float16
        image = color*engine.board.astype('float32')
        image = Variable(torch.from_numpy(image).unsqueeze(0).unsqueeze(0))
        moves = self.model(image).squeeze()

        # Sort moves and play optimal
        idxs = np.argsort(-moves.data.numpy(), axis=None)
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
