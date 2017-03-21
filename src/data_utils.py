import sys, os, re, glob
import h5py
import tensorflow as tf
from engine import *

PRO_H5 = './data/pro.h5'

SGF_SIZE_REGEX = r'SZ\[(\d*)\]'
SGF_MOVE_REGEX = r';([BW])\[([a-t])([a-t])\]'
SGF_TREE_REGEX = r'\(.*\)'

NUM_FEATURES = 3

def input_features(board):
    shape = board.shape + (NUM_FEATURES,)
    image = np.zeros(shape)
    image[..., 0] = board > EMPTY
    image[..., 1] = board < EMPTY
    image[..., 2] = board == EMPTY
    return image

def rot90_labels(labels, size):
    rows = labels // size
    cols = labels % size
    return (size - cols - 1)*size + rows

def flipud_labels(labels, size):
    rows = labels // size
    cols = labels % size
    return (size - rows - 1)*size + cols

def augment_data(boards, labels):
    idx = np.random.choice(8)
    _, size, _ = boards.shape
    if idx == 0:
        pass
    elif idx == 1:
        boards = np.rot90(boards, axes=(1,2))
        labels = rot90_labels(labels, size)
    elif idx == 2:
        boards = np.rot90(boards, k=2, axes=(1,2))
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
    elif idx == 3:
        boards = np.rot90(boards, k=-1, axes=(1,2))
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
    elif idx == 4:
        boards = np.flip(boards, axis=1)
        labels = flipud_labels(labels, size)
    elif idx == 5:
        boards = np.rot90(boards, axes=(1,2))
        boards = np.flip(boards, axis=1)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    elif idx == 6:
        boards = np.rot90(boards, k=2, axes=(1,2))
        boards = np.flip(boards, axis=1)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    elif idx == 7:
        boards = np.rot90(boards, k=-1, axes=(1,2))
        boards = np.flip(boards, axis=1)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    return input_features(boards), labels

def d8_forward(board):
    # Get all flips/rotations of the board.
    shape = (8,) + board.shape
    boards = np.empty(shape, dtype=board.dtype)
    boards[0] = board
    boards[1] = np.rot90(boards[0])
    boards[2] = np.rot90(boards[1])
    boards[3] = np.rot90(boards[2])
    boards[4] = np.flipud(boards[0])
    boards[5] = np.flipud(boards[1])
    boards[6] = np.flipud(boards[2])
    boards[7] = np.flipud(boards[3])
    return boards

def d8_backward(images):
    # Revert all flipped/rotated images back to original space.
    refls = np.empty(images.shape, dtype=images.dtype)
    refls[0] = images[0]
    refls[1] = np.rot90(images[1], k=-1)
    refls[2] = np.rot90(images[2], k=2)
    refls[3] = np.rot90(images[3])
    refls[4] = np.flipud(images[4])
    refls[5] = np.rot90(np.flipud(images[5]), k=-1)
    refls[6] = np.rot90(np.flipud(images[6]), k=2)
    refls[7] = np.rot90(np.flipud(images[7]))
    return refls.mean(axis=0)

def data_from_sgf(fname):
    # Extract the sgf data.
    with open(fname) as f:
        lines = f.read()
    # Ignore SGFs that have tree data.
    # TODO: Handle tree data issue better.
    # NOTE: Right now it throws out some okay games.
    lines = lines.strip()
    assert lines[0] == '(' and lines[-1] == ')'
    lines = lines[1:-1]
    assert not re.search(SGF_TREE_REGEX, lines, re.DOTALL)
    # Extract size.
    match = re.search(SGF_SIZE_REGEX, lines)
    if match:
        size = int(match.group(1))
    else:
        size = 19
    # Play through the game and store the inputs/outputs.
    matches = re.findall(SGF_MOVE_REGEX, lines)
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
        assert col < size and row < size
        # Store current image/label, including flips/rotations.
        board = color*engine.board
        boards[t] = board
        labels[t] = row*size + col
        # Update engine.
        engine.make_move(move, color)
    return boards, labels

def init_h5(fname, size):
    # Initialize h5 database.
    db = h5py.File(fname, 'x')
    boards = db.create_dataset("boards",
            data=np.empty((0, size, size)),
            maxshape=(None, size, size))
    labels = db.create_dataset("labels",
            data=np.empty((0,)),
            maxshape=(None,))
    return db

def add_sgf(fname, db):
    print "Adding {}".format(fname)
    # Extract game record from sgf.
    new_boards, new_labels = data_from_sgf(fname)
    M, s1, _ = new_boards.shape
    # Add new data to database.
    boards = db["boards"]
    labels = db["labels"]
    N, s2, _ = boards.shape
    assert s1 == s2
    boards.resize((N+M, s1, s1))
    boards[N:N+M] = new_boards
    labels.resize((N+M,))
    labels[N:N+M] = new_labels
    db.flush()

def add_dir(dirname, db):
    # Walk the directory and add all sgfs.
    for (dirpath, dirnames, fnames) in os.walk(dirname):
        for fname in fnames:
            if not fname.endswith(".sgf"):
                continue
            fname = os.path.join(dirpath, fname)
            try:
                add_sgf(fname, db)
            except:
                print "{} seems broken, skipping".format(fname)
                continue

if __name__ == "__main__":
    # Create pro database.
    db = init_h5(PRO_H5, 19)
    dirs = ['./data/pro/2000/',
            './data/pro/2001/',
            './data/pro/2013/',
            './data/pro/2014/']
    # Add all of the data to the h5 store.
    for dirname in dirs:
        add_dir(dirname, db)
    db.close()
