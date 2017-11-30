import sys, os, re, glob
import h5py
from engine import *

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
    images = np.empty([len(bws), size, size])
    labels = np.empty(len(bws), dtype=int)
    for t, bw in enumerate(bws):
        # Convert sgf format to engine format.
        color, col, row = bw
        color = BLACK if color == 'B' else WHITE
        col = ord(col) - ord('a')
        row = ord(row) - ord('a')
        # Store current image/label.
        images[t] = color*engine.board
        labels[t] = row*size + col
        # Update engine.
        engine.make_move((row, col), color)
    return images, labels

######################################################################
# Input Feature Utils

def input_features(board, color):
    x = np.zeros((NUM_FEATURES,) + board.shape, dtype='float32')
    x[0] = (board == EMPTY)
    x[1] = (board == color)
    x[2] = (board == -color)
    return x

def rot90_labels(labels, size):
    rows = labels // size
    cols = labels % size
    return (size - cols - 1)*size + rows

def flipud_labels(labels, size):
    rows = labels // size
    cols = labels % size
    return (size - rows - 1)*size + cols

def augment_data(images, labels):
    idx = np.random.choice(8)
    N, size, _ = images.shape
    if idx == 0:
        pass
    elif idx == 1:
        images = np.rot90(images, axes=(1,2))
        labels = rot90_labels(labels, size)
    elif idx == 2:
        images = np.rot90(images, k=2, axes=(1,2))
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
    elif idx == 3:
        images = np.rot90(images, k=-1, axes=(1,2))
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
    elif idx == 4:
        images = np.flip(images, axis=1)
        labels = flipud_labels(labels, size)
    elif idx == 5:
        images = np.rot90(images, axes=(1,2))
        images = np.flip(images, axis=1)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    elif idx == 6:
        images = np.rot90(images, k=2, axes=(1,2))
        images = np.flip(images, axis=1)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    elif idx == 7:
        images = np.rot90(images, k=-1, axes=(1,2))
        images = np.flip(images, axis=1)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = rot90_labels(labels, size)
        labels = flipud_labels(labels, size)
    # Convert the images to input_features
    # TODO: input_features abstraction is ugly right now
    # TODO: data ordering
    images = input_features(images, BLACK).swapaxes(0,1)
    # TODO? use row/col, not label
    return images, labels

def d8_forward(image):
    # Get all flips/rotations of the image.
    shape = (8,) + image.shape
    images = np.empty(shape, dtype=image.dtype)
    images[0] = image
    images[1] = np.rot90(images[0])
    images[2] = np.rot90(images[1])
    images[3] = np.rot90(images[2])
    images[4] = np.flipud(images[0])
    images[5] = np.flipud(images[1])
    images[6] = np.flipud(images[2])
    images[7] = np.flipud(images[3])
    return images

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
