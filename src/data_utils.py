import sys, os, re, glob
import h5py
import tensorflow as tf
from engine import *

TRAIN_H5 = './data/train.h5'
VAL_H5 = './data/val.h5'
TEST_H5 = './data/test.h5'

# TODO: Add pass as possible output?
SGF_MOVE_REGEX = r';([BW])\[([a-s])([a-s])\]'
SGF_TREE_REGEX = r'\(.*\)'
SGF_SIZE_REGEX = r'SZ\[(\d*)\]'
SGF_HANDICAP_REGEX = r'HA\[(\d*)\]'

NUM_FEATURES = 3

#########################
## Input Feature Utils ##
#########################

def input_features(image):
    shape = image.shape + (NUM_FEATURES,)
    x = np.zeros(shape)
    x[..., 0] = image > EMPTY
    x[..., 1] = image < EMPTY
    x[..., 2] = image == EMPTY
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

#############################
## Data Extraction/Storage ##
#############################

def data_from_sgf(fname):
    # Extract the sgf data.
    with open(fname) as f:
        lines = f.read()
    # Ignore handicap games.
    match = re.search(SGF_HANDICAP_REGEX, lines)
    if match:
        handicap = int(match.group(1))
        if handicap > 0:
            print "{} stone handicap game, skipping".format(handicap)
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

def init_h5(fname, size=19):
    # Initialize h5 database.
    db = h5py.File(fname, 'x')
    images = db.create_dataset("images", dtype='i1',
            data=np.empty((0, size, size)),
            maxshape=(None, size, size))
    labels = db.create_dataset("labels", dtype='i2',
            data=np.empty((0,)),
            maxshape=(None,))
    return db

def add_sgf(fname, db):
    # Extract game record from sgf.
    new_images, new_labels = data_from_sgf(fname)
    M, s1, _ = new_images.shape
    # Add new data to database.
    images = db["images"]
    labels = db["labels"]
    N, s2, _ = images.shape
    assert s1 == s2
    images.resize((N+M, s1, s1))
    images[N:N+M] = new_images
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
            except NotImplementedError:
                continue
            except AssertionError:
                print "{} seems broken, skipping".format(fname)
                continue
            except:
                print "New error"
                import IPython; IPython.embed()

if __name__ == "__main__":
    # Create val/test databases.
    db = init_h5(VAL_H5)
    add_dir('./data/sgf/val/', db)
    db.close()

    db = init_h5(TEST_H5)
    add_dir('./data/sgf/test/', db)
    db.close()

    # Create train database.
    db = init_h5(TRAIN_H5, 19)
    dirs = ['./data/sgf/train/{:04}'.format(i)
            for i in range(48)]
    # Add all of the data to the h5 store.
    for dirname in dirs:
        print "Adding {}".format(dirname)
        add_dir(dirname, db)
    db.close()
