import os, re, glob
import tensorflow as tf
from engine import *

MINIGO_DIRNAME = './data/misc/9x9/Minigo/'
PRO_DIRNAME = './data/pro/2014/01/'

MINIGO_STORE = './data/minigo.h5'
PRO_STORE = './data/pro.h5'

SGF_SIZE_REGEX = r'SZ\[(\d*)\]'
SGF_MOVE_REGEX = r';([BW])\[([a-t])([a-t])\]'

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

def d8_forward_labels(move, size):
    # Helper functions to mimic numpy versions.
    def rot90(move, size):
        row, col = move
        return (size - col - 1, row)
    def flipud(move, size):
        row, col = move
        return (size - row - 1, col)
    # Get all flips/rotations of the move.
    moves = [move]
    moves.append(rot90(moves[0], size))
    moves.append(rot90(moves[1], size))
    moves.append(rot90(moves[2], size))
    moves.append(flipud(moves[0], size))
    moves.append(flipud(moves[1], size))
    moves.append(flipud(moves[2], size))
    moves.append(flipud(moves[3], size))
    # Flatten to get the labels.
    labels = [row*size + col for (row, col) in moves]
    return labels

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
    images = np.empty((8*len(matches), size, size, NUM_FEATURES))
    labels = np.empty(8*len(matches), dtype=int)
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
        # Update engine.
        try:
            engine.make_move(move, color)
        except:
            from IPython import embed; embed()
        # Store updated features and move, including flips/rotations.
        image = engine.get_features(color)
        images[8*t:8*(t+1)] = d8_forward(image)
        labels[8*t:8*(t+1)] = d8_forward_labels(move, size)
    return images, labels

def save_data(fname, game_id, images, labels):
    df = pd.Dataframe
    with pd.HDFStore(fname) as store:
        pass

if __name__ == '__main__':
    from IPython import embed; embed()
