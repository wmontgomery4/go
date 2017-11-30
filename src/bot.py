import re
import json
import glob
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from engine import *
from data_utils import *


ROOT = 'bots/{}'
CONFIG_JSON = 'bots/{}/config.json'
WEIGHTS_DAT = 'bots/{}/weights.{:09d}.dat'


class Bot(nn.Module):
    def __init__(self, name, step=None):
        super(Bot, self).__init__()

        # TODO: random seeding

        # Store args
        self.name = name
        self.step = step
        self.size = 19

        # Load config
        with open(CONFIG_JSON.format(name)) as f:
            self.config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))


        # Initialize weights
        # TODO: model.py
        # TODO: more complex models
        C = self.config.model.n_channels
        L = self.config.model.n_layers

        layers = [nn.Conv2d(NUM_FEATURES, C, 3, padding=1)]
        layers.append(nn.ReLU())
        for i in range(L-2):
            layers.append(nn.Conv2d(C, C, 3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(C, 1, 3, padding=1))

        self.model = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        # TODO: optim.py
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def gen_move(self, engine, color):
        # TODO: add passing move
        # TODO: add ko/handicap input
        if engine.last_move == PASS:
            return PASS

        # TODO? float16
        boards = d8_forward(engine.board)
        images = input_features(boards, color).swapaxes(0,1)
        images = Variable(torch.from_numpy(images))
        moves = self.model(images).squeeze()
        moves = d8_backward(moves.data.numpy())

        # Sort moves and play optimal
        idxs = np.argsort(-moves, axis=None)
        for idx in idxs:
            move = (idx // self.size, idx % self.size)
            if engine.legal(move, color):
                return move
        return PASS

    # TODO: label smoothing
    def train(self):
        batch_size  = 8
        max_iters   = 10000
        data_source = 'data/Go_Seigen/*.sgf'

        # Get data
        images = np.empty([0, self.size, self.size])
        labels = np.empty(0, dtype=int)
        for sgf in glob.iglob(data_source):
            print("Loading",sgf)
            # TODO: how to do try/except properly?
            # TODO: why does that one Go Seigen game fail?
            result = None
            try:
                result = data_from_sgf(sgf)
            except:
                print("Can't load:",sgf)
                #print("Can't load:",e)
            if result is not None:
                images = np.r_[images, result[0]]
                labels = np.r_[labels, result[1]]

        # Train on minibatches
        print("Training on",images.shape[0],"moves!")
        for i in range(max_iters):
            # Forward/backward
            self.optim.zero_grad()
            idxs = np.random.choice(images.shape[0], batch_size)
            x, y = augment_data(images[idxs], labels[idxs])
            x = Variable(torch.from_numpy(x), requires_grad=True)
            y_hat = self.model(x).view([batch_size, -1])
            y = Variable(torch.from_numpy(y))
            J = self.loss(y_hat, y)
            J.backward()

            # Gradient step
            # TODO: better way to access J.data[0]
            print("Step: {}/{}, loss: {:.3f}".format(i, max_iters, J.data[0]))
            self.optim.step()

    def _save(self):
        # Initialize the save directory if it hasn't been created.
        dirname = DIRNAME.format(self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # Save the current state of the session.
        self.saver.save(self.sess, CKPT_BASE.format(self.name),
                global_step=self.global_step)
