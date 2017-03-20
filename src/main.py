import sys, argparse
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
    print "Parsing args"
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-t', '--global_step', default=None)
    args = parser.parse_args()

    print "Creating bot"
    bot = Bot(name=args.name, global_step=args.global_step)
    engine = Engine()

    print "Loading data"
    db = h5py.File(PRO_H5, 'r')
    boards = db["boards"][:]
    labels = db["labels"][:]

    print "Training {}".format(bot.name)
    bot.train(boards, labels, epochs=8)
