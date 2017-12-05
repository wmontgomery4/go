# TODO: move to source.py
import glob

import numpy as np
import torch
import torch.nn as nn

from data_utils import *
from engine import *


class Bot(nn.Module):
    def __init__(self, config, step=None):
        super(Bot, self).__init__()

        # TODO: random seeding

        # Store args
        self.config = config
        # TODO: load last weights if possible
        self.step = step or 0
        self.size = 19

        # TODO: move all below init to separate method

        # Initialize model
        # TODO: add value network
        # TODO: model.py/optim.py/loss.py/source.py
        C = self.config['model']['n_channels']
        H = self.config['model']['n_hidden_layers']
        layers = [nn.Conv2d(NUM_FEATURES, C, 3, padding=1)]
        layers.append(nn.ReLU())
        for i in range(H):
            layers.append(nn.Conv2d(C, C, 3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(C, 1, 3, padding=1))
        self.model = nn.Sequential(*layers)
        # TODO: cuda config
        if torch.cuda.is_available():
            self.model.cuda()

        # Load weights
        if self.step:
            weights_dat = config['weights_dat'].format(step)
            # TODO: cleaner loading
            try:
                self.model.load_state_dict(torch.load(weights_dat))
            except AssertionError:
                self.model.load_state_dict(torch.load(weights_dat,
                                                      map_location=lambda s,l: s))

        # Build optim/loss
        lr = self.config['optim']['lr']
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def gen_move(self, engine, color):
        # TODO: add passing move
        if engine.last_move == PASS:
            return PASS

        # TODO? float16
        boards = d8_forward(engine.board) # (8, 19, 19)
        images = to_torch_var(board_to_image(boards, color)) # (8, 3, 19, 19)
        moves = self.model(images).squeeze() # (8, 19, 19)
        moves = d8_backward(moves.data.cpu().numpy()) # (19, 19)

        # Sort moves and play optimal
        idxs = np.argsort(-moves, axis=None)
        for idx in idxs:
            move = (idx // self.size, idx % self.size)
            if engine.legal(move, color):
                return move
        return PASS

    # TODO: label smoothing or max_ent
    def train(self):
        max_iters = 10000000
        batch_size = 8
        log_interval = 100
        save_interval = 10000
        data_source = 'data/Takagawa/*.sgf'

        # Get data
        # TODO: proper data loader
        images = []
        labels = []
        for sgf in glob.iglob(data_source):
            # TODO: why does that one Go Seigen game fail?
            try:
                _images, _labels = data_from_sgf(sgf)
                images.append(_images)
                labels.append(_labels)
            except Exception as e:
                print("Can't load:", e)
        images = np.concatenate(images)
        labels = np.concatenate(labels)

        # Train on minibatches
        print("Training on", images.shape[0], "moves!")
        while self.step < max_iters:
            # Forward pass
            idxs = np.random.choice(images.shape[0], batch_size)
            X, Y = augment_data(images[idxs], labels[idxs])
            # NOTE: have to copy X because rotating makes negative strides
            X = to_torch_var(X.copy(), requires_grad=True)
            Y_hat = self.model(X).view([batch_size, -1])
            Y = to_torch_var(Y, dtype=int)
            J = self.loss(Y_hat, Y)

            # Backward pass
            self.optim.zero_grad()
            J.backward()
            self.optim.step()

            # Saving/Logging
            self.step += 1
            if self.step % log_interval == 0:
                print("Step: {}/{}, loss: {:.3f}".format(self.step, max_iters, J.data[0]))
            if self.step % save_interval == 0:
                self.save()

    def save(self):
        weights_dat = self.config['weights_dat'].format(self.step)
        torch.save(self.model.state_dict(), weights_dat)
