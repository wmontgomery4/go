import sys, argparse
import h5py
from IPython import embed

from engine import *
from data_utils import *
from bot import Bot
from cli import CLI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--name', default=None)
    p.add_argument('-t', '--train', action='store_true')
    p.add_argument('-i', '--interactive', action='store_true')
    args = p.parse_args()

    # Create bot
    bot = Bot(name=args.name)
    print("Created", bot.name)
    if args.train:
        bot.train()
    if args.interactive:
        engine = Engine()
        cli = CLI()
        rollout(engine, cli, bot)
