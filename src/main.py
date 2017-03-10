import sys
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
    engine = Engine()
    human = CLI()

    # Load training data.
    # TODO: Storing/loading bots.
    arch = [(64, 5, 1)] + [(64, 3, 1)]*7
    size = 19
    dirname = PRO_DIRNAME
#    arch = [(64, 3, 3)]*6
#    size = 9
#    dirname = MINIGO_DIRNAME

    boards = np.empty((0, size, size))
    labels = np.empty(0, dtype=int)
    for fname in os.listdir(dirname):
        if not fname.endswith(".sgf"):
            continue
        fname = dirname + fname
        print "Loading {}".format(fname)
        bs, ls = data_from_sgf(fname)
        boards = np.concatenate((boards, bs))
        labels = np.concatenate((labels, ls))

    # Create bot and train on data.
    bot = Bot(size, arch)
    bot.train(boards, labels, batch_size=64, lr=1e-3)

    # Interactive session with bot.
    rollout(engine, bot, human)
