import sys
import h5py
from IPython import embed

from engine import *
from data_utils import *
from bot import Bot
from cli import CLI

def rollout(engine, black, white, moves=500):
    for i in range(moves // 2):
        # Black plays a move.
        move = black.gen_move(engine, BLACK)
        if move == engine.last_move == PASS:
            break
        engine.make_move(move, BLACK)
        # White plays a move.
        move = white.gen_move(engine, WHITE)
        if move == engine.last_move == PASS:
            break
        engine.make_move(move, WHITE)
    return engine


if __name__ == "__main__":
    print "Starting"
    human = CLI()

    # Create bot.
    # TODO: Storing/loading bots.
    bot = Bot()
    engine = Engine()

    # Create bot and train on data.
    db = h5py.File(PRO_H5, 'r')
    images = db["images"]
    labels = db["labels"]
    print "Training {}".format(bot.name)
    bot.train(images, labels)

    # Interactive session with bot.
    rollout(engine, bot, human)
