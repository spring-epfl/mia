import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
import keras
import numpy as np

from keras import layers
from keras.datasets import cifar10


WIDTH = 32
HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 10

SHADOW_DATASET_SIZE = 200


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype="uint8")[y]


def get_cifar_model():
    model = keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

@pytest.fixture
def target_clf(data):
    (X_train, y_train), _ = data
    clf = get_cifar_model()
    clf.fit(X_trian, y_train, num_batches=1)
    yield clf

def test_binary_downsample():
    X = np.zeros([10, 10])
    y = [1, 1, 1, 1, 1, 0, 1, 1, 0, 0]  # 70% positive
    target_y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 50% positive
