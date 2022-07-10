# ===================================================================
# Imports
# ===================================================================

import tarfile

import numpy as np
import tensorflow as tf
from config import BATCH_SIZE, DATA_PATH, IMG_SIZE

from utils.pre_process import prepare_raw_bytes_for_model

# ===================================================================
# Functions
# ===================================================================

# will yield raw image bytes from tar with corresponding label
def raw_generator_with_label_from_tar(tar_fn, files, labels):
    label_by_fn = dict(zip(files, labels))
    with tarfile.open(tar_fn) as f:
        while True:
            m = f.next()
            if m is None:
                break
            if m.name in label_by_fn:
                yield f.extractfile(m).read(), label_by_fn[m.name]



def batch_generator(items, batch_size):
    """
    Implement batch generator that yields items in batches of size batch_size.
    There's no need to shuffle input items, just chop them into batches.
    Remember about the last batch that can be smaller than batch_size!
    Input: any iterable (list, generator, ...). You should do `for item in items: ...`
        In case of generator you can pass through your items only once!
    Output: In output yield each batch as a list of items.
    """
    
    ###
    batch = []
    for i, item in enumerate(items):
        batch.append(item)
        if i % batch_size == batch_size-1:
            yield batch
            batch = []
    # return last (incomplete) batch
    if batch:
        yield [item for item in batch if item]


def train_generator(files, labels):
    n_classes = len(np.unique(labels))
    while True:  # so that Keras can loop through this as long as it wants
        for batch in batch_generator(raw_generator_with_label_from_tar(
                DATA_PATH, files, labels), BATCH_SIZE):
            # prepare batch images
            batch_imgs = []
            batch_targets = []
            for raw, label in batch:
                img = prepare_raw_bytes_for_model(raw, IMG_SIZE)
                batch_imgs.append(img)
                batch_targets.append(label)
            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = tf.keras.utils.to_categorical(batch_targets, n_classes)
            yield batch_imgs, batch_targets


def test_generator(files, labels):
    while True:  # so that Keras can loop through this as long as it wants
        for batch in batch_generator(raw_generator_with_label_from_tar(
                DATA_PATH, files, labels), BATCH_SIZE):
            # prepare batch images
            batch_imgs = []
            for raw, _ in batch:
                img = prepare_raw_bytes_for_model(raw, IMG_SIZE)
                batch_imgs.append(img)
            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)
            yield batch_imgs