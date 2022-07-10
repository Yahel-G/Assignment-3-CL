# ===================================================================
# Imports
# ===================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from config import IMG_SIZE

# ===================================================================
# Functions
# ===================================================================

def inception(n_classes):
    # load pre-trained model graph, don't add final layer
    model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                              weights='imagenet')
    # add global pooling just like in InceptionV3
    new_output = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = tf.keras.layers.Dense(n_classes, activation='softmax')(new_output)
    model = tf.keras.Model(model.inputs, new_output)
    return model

def xception(n_classes):
    # load pre-trained model graph, don't add final layer
    model = tf.keras.applications.Xception(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                              weights='imagenet')
    # add global pooling just like in InceptionV3
    new_output = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = tf.keras.layers.Dense(n_classes, activation='softmax')(new_output)
    model = tf.keras.Model(model.inputs, new_output)
    return model


def vgg19(n_classes):
    # load pre-trained model graph, don't add final layer
    model = tf.keras.applications.VGG19(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                              weights='imagenet')
    # add global pooling just like in InceptionV3
    new_output = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = tf.keras.layers.Dense(n_classes, activation='softmax')(new_output)
    model = tf.keras.Model(model.inputs, new_output)
    return model


def plot_training(history, save_path):
  """
  Plot learning curves of the trained model
  """
  # Summarize history for accuracy
  plt.figure()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.grid(True)
  plt.savefig(Path.joinpath(save_path, 'figs', 'accuracy.png'), bbox_inches="tight")
  # Summarize history for loss
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.grid(True)
  plt.savefig(Path.joinpath(save_path, 'figs', 'loss.png'), bbox_inches="tight")

