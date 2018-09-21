import pytest
import keras

from keras.datasets import cifar10


@pytest.fixture(scope='module')
def data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    y_train = y_train.astype('float')
    y_test = y_test.astype('float')
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)

