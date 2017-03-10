import sys
import cPickle
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
    size = 19
    arch = [(64, 5, 1)]
    arch += [(64, 3, 1)]*7
#    arch = [(64, 5, 1)] + [(64, 3, 1)]*7
    bot = Bot(size, arch)
    engine = Engine(size)

    # Store training data.
    # TODO: Use h5.
    dirname = PRO_DIRNAME
    images = np.empty((0, size, size, NUM_FEATURES))
    labels = np.empty(0, dtype=int)
    for fname in os.listdir(dirname):
        if not fname.endswith(".sgf"):
            continue
        fname = dirname + fname
        print "Loading {}".format(fname)
        bs, ls = data_from_sgf(fname)
        images = np.concatenate((images, bs))
        labels = np.concatenate((labels, ls))
    with open('data/images.npy', 'wb') as f:
        np.save(f, images, -1)
    with open('data/labels.npy', 'wb') as f:
        np.save(f, labels, -1)

    # Load training data.
    with open('data/images.npy') as f:
        images = np.load(f)
    with open('data/labels.npy') as f:
        labels = np.load(f)
    print "Data is loaded"

    # Create bot and train on data.
    bot.train(images, labels, batch_size=32)

    # Interactive session with bot.
    rollout(engine, bot, human)
