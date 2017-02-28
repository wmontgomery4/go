import numpy as np
import tensorflow as tf


class Bot():
    def __init__(self, engine, color):
        self.engine = engine
        self.color = color

    def act(self):
        # TODO: neural net stuff.
        while True:
            move = tuple(np.random.randint(self.engine.size, size=2))
            if self.engine.legal(move, self.color):
                return move
