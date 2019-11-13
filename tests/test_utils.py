import numpy as np

from mia.utils import _binary_class_downsample


def test_binary_downsample():
    X = np.arange(100).reshape([10, 10])
    y = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0])  # 70% positive
    target_y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 50% positive
    downsampled_X, downsampled_y = _binary_class_downsample(X, y, target_y)
    assert len(downsampled_X) == len(downsampled_y)
    assert downsampled_y.mean() == target_y.mean()
