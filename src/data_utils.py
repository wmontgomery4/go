import os, re, glob
import pandas as pd
from engine import *
from data_utils import *

MINIGO_DIRNAME = './data/misc/9x9/Minigo/'
PRO_DIRNAME = './data/pro/2014/01/'

MINIGO_STORE = './data/minigo.h5'
PRO_STORE = './data/pro.h5'

SGF_SIZE_REGEX = r'SZ\[(\d*)\]'
SGF_MOVE_REGEX = r';([BW])\[([a-t])([a-t])\]'

def data_from_sgf(fname):
    # Extract the sgf data.
    with open(fname) as f:
        lines = f.read()
    size_regex = re.compile(SGF_SIZE_REGEX)
    match = size_regex.search(lines)
    if match:
        size = int(match.group(1))
    else:
        size = 19
    move_regex = re.compile(SGF_MOVE_REGEX)
    matches = move_regex.findall(lines)
    # Play through the game and store the inputs/outputs.
    engine = Engine(size)
    boards = np.empty((len(matches), size, size))
    labels = np.empty(len(matches), dtype=int)
    for t, match in enumerate(matches):
        # Convert sgf format to engine format.
        color, col, row = match
        color = BLACK if color == 'B' else WHITE
        col = ord(col) - ord('a')
        row = ord(row) - ord('a')
        move = (row, col)
        # TODO: Handle weird passing case better.
        if col >= 19 or row >= 19:
            continue
        # Play move and store board state and move.
        try:
            engine.make_move(move, color)
        except:
            from IPython import embed; embed()
        boards[t] = color*engine.board
        labels[t] = row*size + col
    return boards, labels

def save_data(fname, game_id, boards, labels):
    df = pd.Dataframe
    with pd.HDFStore(fname) as store:
        pass

if __name__ == '__main__':
    from IPython import embed; embed()
