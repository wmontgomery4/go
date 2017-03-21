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
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--name', default=None)
    p.add_argument('-t', '--train', action='store_true')
    p.add_argument('-s', '--global_step', type=int, default=None)
    p.add_argument('-d', '--data',  default=PRO_H5)
    p.add_argument('-e', '--epochs', type=int, default=8)
    args = p.parse_args()

    print "Creating bot"
    bot = Bot(name=args.name, global_step=args.global_step)
    engine = Engine()

    if args.train:
        print "Loading data"
        db = h5py.File(args.data, 'r')
        boards = db["boards"][:]
        labels = db["labels"][:]

        print "Training {}".format(bot.name)
        bot.train(boards, labels, epochs=args.epochs)
