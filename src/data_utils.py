import os
import re
import torch

from engine import *

# TODO: sgf.py
# SGF Regexes for size, handicap, add_black and black_white
SGF_SZ = r'SZ\[(\d*)\]'
SGF_HA = r'HA\[(\d*)\]'
SGF_AB = r'AB\[([a-s][a-s])\]'
SGF_AW = r'AW\[([a-s][a-s])\]'
SGF_BW = r';([BW])\[([a-s])([a-s])\]'

SGF_TREE = r'\(.*\)'

# TODO: add ko
# TODO: Add pass as possible output?
NUM_FEATURES = 3


######################################################################
# Data extraction

def board_to_image(board, color_to_play):
    rank = len(board.shape)
    if rank == 2:
        board = board[None] # (1, size, size)
    else:
        assert rank == 3, "Shape must be (size, size) or (N, size, size)"
    image = np.empty((NUM_FEATURES,) + board.shape, dtype=int) # (3, N, 19, 19)
    image[0] = (board == EMPTY)
    image[1] = (board == color_to_play)
    image[2] = (board == -color_to_play)
    return image.swapaxes(0,1) # (N, 3, 19, 19)


def data_from_sgf(fname):
    # Extract the sgf data.
    with open(fname) as f:
        sgf = f.read()

    # Extract size.
    sz = re.search(SGF_SZ, sgf)
    size = int(sz) if sz else 19

    # Ignore handicap games.
    # TODO: implement handicap games properly
    if re.search(SGF_HA, sgf) or re.search(SGF_AB, sgf) or re.search(SGF_AW, sgf):
        raise NotImplementedError

    # Play through the game and store the inputs/outputs.
    bws = re.findall(SGF_BW, sgf)
    engine = Engine(size)
    images = np.empty([len(bws), NUM_FEATURES, size, size], dtype=int)
    labels = np.empty(len(bws), dtype=int)
    for t, bw in enumerate(bws):
        # Convert sgf format to engine format.
        color, col, row = bw
        color = BLACK if color == 'B' else WHITE
        col = ord(col) - ord('a')
        row = ord(row) - ord('a')
        # Store current image/label.
        images[t] = board_to_image(engine.board, color)
        labels[t] = row*size + col
        # Update engine.
        engine.make_move((row, col), color)
    return images, labels

######################################################################
# Input Feature Utils

def to_torch_var(arr, requires_grad=False, cuda=True):
    arr = arr.astype('float32')
    var = torch.autograd.Variable(torch.from_numpy(arr), requires_grad=requires_grad)
    return var.cuda() if cuda and torch.cuda.is_available() else var

def rot90_labels(labels, size):
    rows = labels // size
    cols = labels % size
    return (size - cols - 1)*size + rows

def flipud_labels(labels, size):
    rows = labels // size
    cols = labels % size
    return (size - rows - 1)*size + cols

def augment_data(images, labels):
    assert len(images.shape) == 4
    idx = np.random.choice(8)
    N, _, size, _ = images.shape
    if idx == 0:
        pass
    elif idx == 1:
        images = np.rot90(images, axes=(2,3))
        labels = rot90_labels(labels, size)
    elif idx == 2:
        images = np.rot90(images, k=2, axes=(2,3))
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
    elif idx == 3:
        images = np.rot90(images, k=-1, axes=(2,3))
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
    elif idx == 4:
        images = np.flip(images, axis=1)
        labels = flipud_labels(labels, size)
    elif idx == 5:
        images = np.rot90(images, axes=(2,3))
        images = np.flip(images, axis=1)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    elif idx == 6:
        images = np.rot90(images, k=2, axes=(2,3))
        images = np.flip(images, axis=1)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    elif idx == 7:
        images = np.rot90(images, k=-1, axes=(2,3))
        images = np.flip(images, axis=1)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    # Convert the images to input_features
    return images, labels

def d8_forward(board):
    # Get all flips/rotations of the board.
    assert len(board.shape) == 2
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

def d8_backward(boards):
    # Revert all flipped/rotated boards back to original space.
    assert len(boards.shape) == 3
    refls = np.empty(boards.shape, dtype=boards.dtype)
    refls[0] = boards[0]
    refls[1] = np.rot90(boards[1], k=-1)
    refls[2] = np.rot90(boards[2], k=2)
    refls[3] = np.rot90(boards[3])
    refls[4] = np.flipud(boards[4])
    refls[5] = np.rot90(np.flipud(boards[5]), k=-1)
    refls[6] = np.rot90(np.flipud(boards[6]), k=2)
    refls[7] = np.rot90(np.flipud(boards[7]))
    return refls.mean(axis=0)
