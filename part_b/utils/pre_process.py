# ===================================================================
# Imports
# ===================================================================

from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

import tarfile

import cv2  # for image processing
from config import BATCH_SIZE, EPOCHS, RESULTS_DIR, SEED
from sklearn.model_selection import train_test_split

# ===================================================================
# Functions
# ===================================================================

def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    For cropping use numpy slicing.
    """
    
    h, w = img.shape[0], img.shape[1]
    m = min(h, w)
    cropped_img = img[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2, :]
    
    return cropped_img


def prepare_raw_bytes_for_model(raw_bytes, img_size, normalize_for_model=True):
    img = decode_image_from_raw_bytes(raw_bytes)  # decode image raw bytes to matrix
    img = image_center_crop(img)  # take squared center crop
    img = cv2.resize(img, (img_size, img_size))  # resize for our model
    if normalize_for_model:
        img = img.astype("float32")  # prepare for normalization
        img = tf.keras.applications.inception_v3.preprocess_input(img)  # normalize for model
    return img


# reads bytes directly from tar by filename (slow, but ok for testing, takes ~6 sec)
def read_raw_from_tar(tar_fn, fn):
    with tarfile.open(tar_fn) as f:
        m = f.getmember(fn)
        return f.extractfile(m).read()


# read filenames firectly from tar
def get_all_filenames(tar_fn):
    with tarfile.open(tar_fn) as f:
        return [m.name for m in f.getmembers() if m.isfile()]


def split_dataset(all_files, all_labels, train_size, valid_size, seed=SEED):
    # split into train/test+val
    tr_files, rem_files, tr_labels, rem_labels = \
    train_test_split(all_files, all_labels, train_size=train_size, random_state=seed, stratify=all_labels)
    # split into val/test
    te_files, va_files, te_labels, va_labels = \
    train_test_split(rem_files, rem_labels, test_size=valid_size/train_size, random_state=seed, stratify=rem_labels)

    return tr_files, tr_labels, te_files, va_files, te_labels, va_labels


def create_directories():
    """
    Create directories to save training log data

    Returns
    -------
    * today: [string] ID and description of the log
    """
    today = datetime.today().strftime("%b-%d-%Y-%H-%M")  +  f'_bs={BATCH_SIZE}_epochs={EPOCHS}'
    Path.cwd().joinpath(RESULTS_DIR, today).mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath(RESULTS_DIR, today,'model_weights').mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath(RESULTS_DIR, today, 'figs').mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath(RESULTS_DIR, today, 'model').mkdir(parents=True, exist_ok=True)
    
    return today

