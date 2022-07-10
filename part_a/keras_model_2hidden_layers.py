from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from metrics import Metrics

print('Loading dataset...')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)

# Normalize to [-1, 1] range:
X = ((X / 255.) - .5) * 2

def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary

# Split into training, validation, and test set:
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=123, stratify=y)

X_test, X_valid, y_test, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=123, stratify=y_temp)


# optional to free up some memory by deleting non-used arrays:
del X_temp, y_temp, X, y

# define model
model = Sequential()
model.add(Input(shape=(28*28,)))
model.add(Dense(units=50, activation='sigmoid'))
model.add(Dense(units=50, activation='sigmoid'))
model.add(Dense(units=10))
model.summary()

# compile model
model.compile(loss='mse', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])

# train
# set callbacks
callbacks = [EarlyStopping(verbose=1, patience=10, monitor='val_loss', restore_best_weights=True)]
history = model.fit(X_train, int_to_onehot(y_train, 10), batch_size=100, epochs=150, callbacks=callbacks, validation_data=[X_valid, int_to_onehot(y_valid, 10)])
model.save('keras_model')

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
  plt.savefig(Path.joinpath(save_path, 'figures', 'keras_model_acc.png'), bbox_inches="tight")
  # Summarize history for loss
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.grid(True)
  plt.savefig(Path.joinpath(save_path, 'figures', 'keras_model_loss.png'), bbox_inches="tight")

plot_training(history, save_path=Path.cwd())

# evaluate
score = model.evaluate(X_test, int_to_onehot(y_test,10))
print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])


# compute AUC score
probas = model.predict(X_test)
test_pred = np.argmax(probas, axis=1)
eval = Metrics(n_classes=10)
eval.compute_roc_auc(int_to_onehot(y_test,10), y_score=probas)
eval.compute_macro_auc()
eval.plot_roc(int_to_onehot(y_test,10), probas,  model_name='keras')

