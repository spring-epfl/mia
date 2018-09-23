import pytest
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from mia.wrappers import TorchWrapper, ExpLrScheduler

torch.set_default_tensor_type("torch.FloatTensor")

WIDTH = 32
HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 10


class Net(nn.Module):
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


@pytest.mark.parametrize(
    "lr_scheduler", [None, ExpLrScheduler(lr_decay_every_epochs=1)]
)
@pytest.mark.parametrize("enable_cuda", [False])
@pytest.mark.parametrize("use_torch_arrays", [True, False])
def test_torch_wrapper_fit(data, lr_scheduler, enable_cuda, use_torch_arrays):
    model = TorchWrapper(
        Net, nn.CrossEntropyLoss, torch.optim.Adam, enable_cuda=enable_cuda
    )

    (X_train, y_train), (X_test, y_test) = data
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    if use_torch_arrays:
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)

    model.fit(X_train, y_train, epochs=2, verbose=True, validation_split=0.1)

    # Expected accuracy is greater than 30%.
    assert np.mean(model.predict(X_test) == y_test) > 0.3
