import os
import json
import random
import string
import argparse

from engine import *
from bot import Bot
from cli import CLI

ROOT = 'bots/{}'
CONFIG_JSON = 'bots/{}/config.json'
WEIGHTS_FORMAT = 'weights.{:09d}.dat'


# TODO: framework.py
# TODO: Sweeps over configs
def add_bot(config_json, name=None):
    # Create new bot directory, generating a random name if None given
    if not name:
        name = ''.join(random.choice(string.ascii_lowercase) for _ in range(4))
    root = ROOT.format(name)
    assert not os.path.exists(root), "woah a collision!"
    os.makedirs(root)

    # Load config, add weight format, save config in bot directory
    with open(config_json) as f:
        config = json.load(f)
    config['weights_dat'] = os.path.join(root, WEIGHTS_FORMAT)
    with open(CONFIG_JSON.format(name), 'w') as f:
        json.dump(config, f)
    return name


def rollout(engine, black, white, moves=500):
    for i in range(moves // 2):
        # Black plays a move.
        move = black.gen_move(engine, BLACK)
        if move == engine.last_move:
            break
        engine.make_move(move, BLACK)
        # White plays a move.
        move = white.gen_move(engine, WHITE)
        if move == engine.last_move:
            break
        engine.make_move(move, WHITE)
    return engine


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--train', action='store_true')
    p.add_argument('-i', '--interactive', action='store_true')
    p.add_argument('-c', '--config')
    p.add_argument('-n', '--name')
    p.add_argument('-s', '--step', type=int)
    args = p.parse_args()

    # TODO: Add/Train/Test subcommands
    if args.config:
        if args.name:
            assert not os.path.exists(ROOT.format(args.name)), "Name already exists!"
        name = add_bot(args.config, args.name)
    elif args.name:
        name = args.name
    else:
        name = add_bot('default_config.json')

    # Load bot
    with open(CONFIG_JSON.format(name)) as f:
        config = json.load(f)
    bot = Bot(config, step=args.step)
    print("Loaded", args.name, "step", args.step)

    # Main logic
    if args.train:
        bot.train()
    if args.interactive:
        engine = Engine()
        cli = CLI()
        rollout(engine, bot, cli)
