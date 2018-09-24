import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
import keras
import skorch
import torch
import numpy as np

from keras import layers
from keras.datasets import cifar10
from torch import nn
from torch.nn import functional as F

from mia.estimators import AttackModelBundle
from mia.estimators import ShadowModelBundle
from mia.estimators import prepare_attack_data
from mia.serialization import BaseModelSerializer

torch.set_default_tensor_type("torch.FloatTensor")


WIDTH = 32
HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 10

SHADOW_DATASET_SIZE = 200


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype="uint8")[y]


class CrossEntropyOneHot(object):
    def __init__(self):
        self.cel = nn.CrossEntropyLoss()

    def __call__(self, out, target):
        _, labels = target.max(dim=1)
        return self.cel(out, labels)


def keras_shadow_model_fn():
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


class ShadowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x, **kwargs):
        del kwargs  # Unused.

        x = x.view(-1, CHANNELS, WIDTH, HEIGHT)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def torch_shadow_model_fn():
    model = skorch.NeuralNetClassifier(
        module=ShadowNet, max_epochs=5, criterion=CrossEntropyOneHot, train_split=None
    )
    return model


# TODO: Make mock tests for the serializer.
class SkorchSerializer(BaseModelSerializer):
    def save(self, model_id, model):
        print("Saved %s" % model_id)
        model.save_params(self.get_model_path(model_id))

    def load(self, model_id):
        model = self.model_fn()
        model.initialize()
        model.load_params(self.get_model_path(model_id))
        print("Loaded %s" % model_id)
        return model


@pytest.fixture
def torch_shadow_serializer():
    prefix = "__serialized_torch_shadow_test_dir"
    serializer = SkorchSerializer(torch_shadow_model_fn, prefix)
    yield serializer
    shutil.rmtree(prefix)


@pytest.fixture
def torch_attack_serializer():
    prefix = "__serialized_torch_attack_test_dir"
    serializer = SkorchSerializer(torch_attack_model_fn, prefix)
    yield serializer
    shutil.rmtree(prefix)


def keras_attack_model_fn():
    model = keras.models.Sequential()

    # Input layer
    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    # Hidden layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    # Output layer
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


class AttackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_CLASSES, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.softmax = nn.Linear(64, 1)

    def forward(self, x, **kwargs):
        del kwargs  # Unused.

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.softmax(x))
        return x


def torch_attack_model_fn():
    model = skorch.NeuralNetClassifier(
        module=AttackNet, max_epochs=5, criterion=nn.BCELoss, train_split=None
    )
    return model


@pytest.mark.parametrize("num_models", [1, 3])
@pytest.mark.parametrize(
    "shadow_model_fn,model_serializer",
    [
        (keras_shadow_model_fn, None),
        (torch_shadow_model_fn, None),
        (torch_shadow_model_fn, pytest.lazy_fixture("torch_shadow_serializer")),
    ],
)
def test_shadow_models_are_created_and_data_is_transformed(
    data, shadow_model_fn, model_serializer, num_models
):

    (X_train, y_train), _ = data
    smb = ShadowModelBundle(
        shadow_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=num_models,
        serializer=model_serializer,
    )
    assert not hasattr(smb, "shadow_models_")

    X_shadow, y_shadow = smb.fit_transform(
        X_train, y_train, fit_kwargs=dict(epochs=5, verbose=False)
    )

    # X_shadow are prediction vectors for random examples in X_train
    assert len(X_shadow) == len(y_shadow) == (2 * num_models * SHADOW_DATASET_SIZE)

    # X_shadow are concatenations of prediction vectors and true
    # class.
    assert X_shadow.shape[1] == 2 * NUM_CLASSES


@pytest.mark.parametrize("num_models", [3])
@pytest.mark.parametrize("max_models_for_transform", [1, 2])
@pytest.mark.parametrize(
    "shadow_model_fn,model_serializer",
    [
        (keras_shadow_model_fn, None),
        (torch_shadow_model_fn, None),
        (torch_shadow_model_fn, pytest.lazy_fixture("torch_shadow_serializer")),
    ],
)
def test_shadow_transform_with_custom_shadow_indices(
    data, shadow_model_fn, model_serializer, num_models, max_models_for_transform
):

    (X_train, y_train), _ = data
    smb = ShadowModelBundle(
        shadow_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=num_models,
        serializer=model_serializer,
    )

    smb._fit(X_train, y_train, fit_kwargs=dict(epochs=5, verbose=False))

    shadow_indices = range(max_models_for_transform)
    X_shadow, y_shadow = smb._transform(shadow_indices=shadow_indices)
    assert X_shadow.shape[0] == 2 * max_models_for_transform * SHADOW_DATASET_SIZE
    assert y_shadow.shape[0] == 2 * max_models_for_transform * SHADOW_DATASET_SIZE


@pytest.mark.parametrize(
    "shadow_model_fn,shadow_model_serializer,"
    "attack_model_fn,attack_model_serializer",
    [
        (keras_shadow_model_fn, None, keras_attack_model_fn, None),
        (keras_shadow_model_fn, None, torch_attack_model_fn, None),
        (torch_shadow_model_fn, None, torch_attack_model_fn, None),
        (torch_shadow_model_fn, None, keras_attack_model_fn, None),
        (
            torch_shadow_model_fn,
            pytest.lazy_fixture("torch_shadow_serializer"),
            torch_attack_model_fn,
            None,
        ),
        (
            torch_shadow_model_fn,
            pytest.lazy_fixture("torch_shadow_serializer"),
            torch_attack_model_fn,
            pytest.lazy_fixture("torch_attack_serializer"),
        ),
        (
            torch_shadow_model_fn,
            pytest.lazy_fixture("torch_shadow_serializer"),
            keras_attack_model_fn,
            None,
        ),
    ],
)
def test_attack_models_are_created(
    data,
    shadow_model_fn,
    shadow_model_serializer,
    attack_model_fn,
    attack_model_serializer,
):

    (X_train, y_train), _ = data
    smb = ShadowModelBundle(
        shadow_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=3,
        serializer=shadow_model_serializer,
    )
    X_shadow, y_shadow = smb.fit_transform(
        X_train, y_train, fit_kwargs=dict(epochs=5, verbose=False)
    )

    amb = AttackModelBundle(
        attack_model_fn, num_classes=NUM_CLASSES, serializer=attack_model_serializer
    )

    # Fit the attack models.
    amb.fit(X_shadow, y_shadow, fit_kwargs=dict(epochs=5, verbose=False))

    # Predict membership for some training data.
    membership_guesses = amb.predict(X_shadow[:100])
    assert membership_guesses.shape == (len(X_shadow[:100]),)


@pytest.mark.parametrize("shadow_model_fn", [torch_shadow_model_fn])
def test_prepare_attack_data(data, shadow_model_fn):
    (X_train, y_train), (X_test, y_test) = data
    clf = shadow_model_fn()
    clf.fit(X_train, y_train, epochs=3, verbose=False)
    X_attack, y_attack = prepare_attack_data(
        clf, (X_train[:100], y_train[:100]), (X_test[:100], y_test[:100])
    )
    assert X_attack.shape == (200, 20)
    assert y_attack.shape == (200,)
    assert np.all(y_attack[:100])
    assert not np.any(y_attack[100:])


def test_serializer_creates_dir():
    prefix = "__serializer_test_dir"
    serializer = BaseModelSerializer(model_fn=None, prefix=prefix)
    assert prefix in os.listdir(".")
    shutil.rmtree(prefix)
