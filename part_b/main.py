# ===================================================================
# Imports
# ===================================================================

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

print(tf.__version__)
import scipy.io
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (BATCH_SIZE, DATA_PATH, EPOCHS, LABEL_PATH, MODEL_PATH,
                    RESULTS_DIR, SEED, TRAIN_SIZE, VALID_SIZE)
from utils.generators import train_generator, test_generator
from utils.models import inception, plot_training, xception, vgg19
from utils.pre_process import (create_directories, get_all_filenames,
                               split_dataset)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ===================================================================
# Create directories for saving the model & results
# ===================================================================
today = create_directories()
SAVE_PATH = Path.cwd().joinpath(RESULTS_DIR, today)
CHECKPOINT_PATH = Path.cwd().joinpath(RESULTS_DIR, today, 'model_weights','model.{epoch:02d}-{val_loss:.2f}')
LOG_DIR = Path.cwd().joinpath(RESULTS_DIR, today, 'logs', 'scalars')
MODEL_PATH = Path.cwd().joinpath(RESULTS_DIR, today, 'model')   


# ===================================================================
# Prepare for training
# ===================================================================

# list all files in tar sorted by name
all_files = sorted(get_all_filenames(DATA_PATH)) 
# read class labels (0, 1, 2, ...)
all_labels = scipy.io.loadmat(LABEL_PATH)['labels'][0] - 1  

# all_files and all_labels are aligned now
N_CLASSES = len(np.unique(all_labels))
print('Classes:', N_CLASSES)

tr_files, tr_labels, te_files, va_files, te_labels, va_labels = split_dataset(all_files, all_labels, TRAIN_SIZE, VALID_SIZE)

# # ===================================================================
# # Training
# # ===================================================================

model = inception(N_CLASSES)
# model = xception(N_CLASSES)
# model = vgg19(N_CLASSES)

model.summary()
print('# of layers:', len(model.layers))

# set all layers trainable by default
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        # we do aggressive exponential smoothing of batch norm
        # parameters to faster adjust to our new dataset
        layer.momentum = 0.9
    
xl = 4
# fix deep layers (fine-tuning only last xl of layers)
for layer in model.layers[:-len(model.layers) // xl]:
    # fix all but batch norm layers, because we need to update moving averages for a new dataset
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# compile new model
model.compile(
    loss='categorical_crossentropy',  # 102-way classification
    optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-2), 
    metrics=['accuracy'] 
)

# set callbacks
callbacks = [ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, save_best_only=False),
             EarlyStopping(verbose=1, patience=5, monitor='val_loss', restore_best_weights=True)]

# fine tune for 2 epochs (full passes through all training data)
# we make 2*8 epochs, where epoch is 1/8 of our training data to see progress more often
history = model.fit(
          train_generator(tr_files, tr_labels), 
          steps_per_epoch=len(tr_files) // BATCH_SIZE // 4,
          epochs=EPOCHS,
          validation_data=train_generator(va_files, va_labels), 
          validation_steps=len(va_files) // BATCH_SIZE // 2,
          callbacks=callbacks,
          verbose=1,
)


plot_training(history, SAVE_PATH)
model.save(MODEL_PATH)

model = tf.keras.models.load_model(r'C:\School\Master\Courses\Computational Learning\THREE\part_b\wd\results\Jul-08-2022-22-37_bs=32_epochs=32\model')
SAVE_PATH = r"C:\School\Master\Courses\Computational Learning\THREE\part_b\wd\results\Jul-08-2022-22-37_bs=32_epochs=32"

# Accuracy on test set
print('Evaluating test set')

test_accuracy = model.evaluate(
    train_generator(te_files, te_labels), 
    verbose=1, steps=len(te_files) // BATCH_SIZE + 1)[1]
print('Test accuracy: ', test_accuracy)