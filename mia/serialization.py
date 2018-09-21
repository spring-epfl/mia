import os
import abc


class BaseModelSerializer(object):
    """
    ABC class for a model serializer.

    :param model_fn: Function that builds a new model
    :param prefix: Path to the directory where models will be saved.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, model_fn, prefix=".", *args, **kwargs):
        self.model_fn = model_fn
        self.models_path = prefix
        os.makedirs(self.models_path, exist_ok=True)

    def get_model_path(self, model_id):
        """Get the path to the model with given ID."""
        return os.path.join(self.models_path, model_id)

    @abc.abstractmethod
    def load(self, model_id):
        pass

    @abc.abstractmethod
    def save(self, model, model_id):
        pass
