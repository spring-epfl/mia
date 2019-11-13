import numpy as np
from sklearn.model_selection import train_test_split


def _infer_output_format(outputs):
    """
    Infer if the classifier outputs binary predictions, or one-hot vectors.

    Args:
        outputs: An array of classifier's outputs

    Returns:
        tuple: Format, number of classes

    >>> binary_confidence = np.array([0.2, 0.3, 0.0, 0.9])
    >>> _infer_output_format(binary_confidence)
    ('binary_confidence', 2)
    >>> multiclass_confidence = np.array([[0.3, 0.4, 0.3], [0.1, 0.1, 0.8]])
    >>> _infer_output_format(multiclass_confidence)
    ('multiclass_confidence', 3)
    >>> binary_confidence_redundant = np.array([[0.3, 0.6], [0.1, 0.9]])
    >>> _infer_output_format(binary_confidence_redundant)
    ('multiclass_confidence', 2)
    >>> binary_decisions = np.array([1, 0, 0, 1, 0])
    >>> _infer_output_format(binary_decisions)
    ('binary_decisions', 2)
    >>> multiclass_decisions = np.array([1, 2, 7, 10, 5])
    >>> _infer_output_format(multiclass_decisions)
    ('multiclass_decisions', 10)
    """

    shape = outputs.shape
    if len(shape) >= 2:
        return "multiclass_confidence", shape[1]

    unique_elements = np.unique(outputs)
    if len(unique_elements) == 2:
        return "binary_decisions", 2

    is_ints = all([int(elem) == elem for elem in unique_elements])
    if is_ints:
        return "multiclass_decisions", outputs.max()
    else:
        return "binary_confidence", 2


def _binary_class_downsample(X, y, target_y, tolerance=10e-5, seed=1):
    rng = np.random.RandomState(seed=seed)
    target_prop = target_y.mean()
    current_prop = y.mean()
    downsampled_X = X.copy()
    downsampled_y = y.copy()
    while abs(current_prop - target_prop) > tolerance and len(downsampled_X) > 1.:
        overrepresented_class = int(current_prop > target_prop)
        class_indices, = np.where(y == overrepresented_class)
        index_to_remove = rng.choice(class_indices, size=1)
        downsampled_X = np.delete(downsampled_X, index_to_remove, axis=0)
        downsampled_y = np.delete(downsampled_y, index_to_remove, axis=0)
        current_prop = downsampled_y.mean()

    return downsampled_X, downsampled_y
