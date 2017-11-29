import sys, os, re, glob
import h5py
from engine import *

# TODO: Add pass as possible output?
SGF_MOVE_REGEX = r';([BW])\[([a-s])([a-s])\]'
SGF_TREE_REGEX = r'\(.*\)'
SGF_SIZE_REGEX = r'SZ\[(\d*)\]'
SGF_HANDICAP_REGEX = r'HA\[(\d*)\]'

#TODO: add ko
NUM_FEATURES = 3

######################################################################
# Input Feature Utils

def input_features(engine, color):
    x = np.zeros([NUM_FEATURES, engine.size, engine.size], dtype='float32')
    x[0] = (engine.board == EMPTY)
    x[1] = (engine.board == color)
    x[2] = (engine.board == -color)
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
    _, size, _ = images.shape
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
    return input_features(images), labels

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


######################################################################
# Data extraction

def data_from_sgf(fname):
    # Extract the sgf data.
    with open(fname) as f:
        lines = f.read()

    # Ignore handicap games.
    match = re.search(SGF_HANDICAP_REGEX, lines)
    if match:
        handicap = int(match.group(1))
        if handicap > 0:
            print("{} stone handicap game, skipping".format(handicap))
            raise NotImplementedError

    # Extract size.
    match = re.search(SGF_SIZE_REGEX, lines)
    if match:
        size = int(match.group(1))
    else:
        size = 19

    # Play through the game and store the inputs/outputs.
    matches = re.findall(SGF_MOVE_REGEX, lines)
    engine = Engine(size)
    images = np.empty((len(matches), size, size))
    labels = np.empty(len(matches), dtype=int)
    for t, match in enumerate(matches):
        # Convert sgf format to engine format.
        color, col, row = match
        color = BLACK if color == 'B' else WHITE
        col = ord(col) - ord('a')
        row = ord(row) - ord('a')
        move = (row, col)
        # Store current image/label.
        images[t] = color*engine.board
        labels[t] = row*size + col
        # Update engine.
        engine.make_move(move, color)
    return images, labels
