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

        # TODO: move all below init to separate method?

        # Initialize model
        # TODO: model.py/optim.py/loss.py/source.py
        # TODO: more complex models
        C = self.config['model']['n_channels']
        L = self.config['model']['n_layers']
        layers = [nn.Conv2d(NUM_FEATURES, C, 3, padding=1)]
        layers.append(nn.ReLU())
        for i in range(L-2):
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
            self.model.load_state_dict(torch.load(weights_dat))

        # Build optim/loss
        lr = self.config['optim']['lr']
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def gen_move(self, engine, color):
        # TODO: add passing move
        # TODO: add ko/handicap input
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
        max_iters = 1000000
        batch_size = 8
        save_interval = 1000
        data_source = 'data/**/*.sgf'

        # TODO: proper data loader
        files = glob.glob(data_source)
        sgfs = []
        for sgf in glob.iglob(data_source):
            print("Loading", sgf)
            # TODO: how to do try/except properly?
            # TODO: why does that one Go Seigen game fail?
            with open(sgf) as f:
                sgfs.append(f.read())

        #images = []
        #labels = []
        #for sgf in glob.iglob(data_source):
        #    print("Loading", sgf)
        #    # TODO: how to do try/except properly?
        #    # TODO: why does that one Go Seigen game fail?
        #    try:
        #        _images, _labels = data_from_sgf(sgf)
        #        images.append(_images)
        #        labels.append(_labels)
        #    except:
        #        print("Can't load:", sgf)
        #        #print("Can't load:",e)
        #images = np.concatenate(images)
        #labels = np.concatenate(labels)

        # Train on minibatches
        print("Training on", len(sgfs), "games!")
        for i in range(max_iters):
            # Load minibatch
            images = []
            labels = []
            while len(images) < batch_size:
                sgf_idx = np.random.choice(len(sgfs))
                try:
                    _images, _labels = data_from_sgf(sgfs[sgf_idx])
                    move_idx = np.random.choice(_images.shape[0])
                except:
                    continue
                images.append(_images[move_idx])
                labels.append(_labels[move_idx])
            images = np.stack(images)
            labels = np.stack(labels)

            # Forward pass
            X, Y = augment_data(images, labels)
            # NOTE: have to copy X because rotating makes negative strides
            X = to_torch_var(X.copy(), requires_grad=True)
            Y_hat = self.model(X).view([batch_size, -1])
            Y = to_torch_var(Y, dtype=int)
            J = self.loss(Y_hat, Y)
            print("Step: {}/{}, loss: {:.3f}".format(i, max_iters, J.data[0]))

            # Backward pass
            self.optim.zero_grad()
            J.backward()
            self.optim.step()

            # Save
            self.step += 1
            if self.step % save_interval == 0:
                self.save()

    def save(self):
        weights_dat = self.config['weights_dat'].format(self.step)
        torch.save(self.model.state_dict(), weights_dat)
