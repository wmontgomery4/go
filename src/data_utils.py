import os, re
from engine import *

MINIGO_DIR = './data/misc/9x9/Minigo/'
PRO_DIR = './data/pro/2014/01/'

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


if __name__ == '__main__':
    arch = [(3, 64)]*10
    size = 19
    dirname = PRO_DIR
#    arch = [(3, 64)]*10
#    size = 9
#    dirname = MINIGO_DIR
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

    from bot import Bot
    bot = Bot(size, arch)
    bot.train(boards, labels, batch_size=128, epochs=0.1, lr=1e-4)

    from main import *
    human = CLI()
    engine = Engine(size)
    rollout(engine, bot, human)
