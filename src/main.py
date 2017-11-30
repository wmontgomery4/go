import os
import shutil
import random
import string
import argparse

import engine
import data_utils
from bot import Bot, CONFIG_JSON
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
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--train', action='store_true')
    p.add_argument('-i', '--interactive', action='store_true')
    p.add_argument('-c', '--config')
    #p.add_argument('-c', '--config', DEFAULT_CONFIG)
    p.add_argument('-n', '--name')
    p.add_argument('-s', '--step')
    args = p.parse_args()

    # TODO: Add/Train/Test subcommands
    # TODO: Sweeps over configs
    # Create a new bot if config is given
    if not (args.config or args.name):
        args.config = 'config.json'
    if args.config:
        assert args.step is None
        if not args.name:
            letters = string.ascii_lowercase
            args.name = ''.join(random.choice(letters) for _ in range(4))

        # Create directory and store config
        # TODO: this is hacky
        fname = CONFIG_JSON.format(args.name)
        assert not os.path.exists(fname), "woah a collision!"
        os.makedirs(os.path.dirname(fname))
        shutil.copyfile(args.config, fname)


    # Load bot
    bot = Bot(name=args.name, step=args.step)
    print("Created", bot.name)
    if args.train:
        bot.train()
    if args.interactive:
        engine = Engine()
        cli = CLI()
        rollout(engine, cli, bot)
