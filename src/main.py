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
    p.add_argument('--train', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('-t', '--global_step', type=int, default=None)
    p.add_argument('-e', '--epochs', type=int, default=8.0)
    args = p.parse_args()

    if args.name:
        print "Loading bot"
        bot = Bot(name=args.name, global_step=args.global_step)

    engine = Engine()
    cli = CLI()

    rollout(engine,bot,bot,300)
    embed()

    if args.train:
        print "Loading training data"
        db = h5py.File('/tmp/train.h5', 'r')
        #db = h5py.File(TRAIN_H5, 'r')
        images = db["images"][:]
        labels = db["labels"][:]

        print "Training {}".format(bot.name)
        bot.train(images, labels, epochs=args.epochs)

    if args.test:
        print "Loading testing data"
        db = h5py.File(TEST_H5, 'r')
        images = db["images"][:]
        labels = db["labels"][:]

        print "Testing {}".format(bot.name)
        import IPython; IPython.embed()
        bot.train(images, labels, epochs=args.epochs)
