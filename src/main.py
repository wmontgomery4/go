import os
import json
import shutil
import random
import string
import argparse

import engine
import data_utils
from bot import Bot
from cli import CLI

# TODO: framework.py?
ROOT = 'bots/{}'
CONFIG_JSON = 'config.json'
WEIGHTS_DAT = 'weights.{:09d}.dat'

def new_bot(config_json):
    # Create random name
    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(4))
    root = ROOT.format(name)
    assert not os.path.exists(root), "woah a collision!"
    os.makedirs(root)

    # Load config
    with open(config_json) as f:
        config = json.load(f)

    # Add some info and store in root
    config['root'] = root
    config['weights_dat'] = os.path.join(root, WEIGHTS_DAT)
    with open(os.path.join(root, CONFIG_JSON), 'w') as f:
        json.dump(config, f)
    return name


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
    p.add_argument('-n', '--name')
    p.add_argument('-s', '--step')
    args = p.parse_args()

    assert bool(args.name) != bool(args.config), "Must give --name or --config"

    # TODO: Add/Train/Test subcommands
    # TODO: Sweeps over configs

    # Create a new bot if config is given
    if args.config:
        assert args.step is None
        args.name = new_bot(args.config)

    # Load bot
    with open(CONFIG_JSON.format(args.name)) as f:
        config = json.load(f)
    bot = Bot(config, step=args.step)
    print("Created", args.name)

    # Main logic
    if args.train:
        bot.train()
    if args.interactive:
        engine = Engine()
        cli = CLI()
        rollout(engine, cli, bot)
