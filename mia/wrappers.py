import copy

import torch
import time
import tqdm
import numpy as np
import os

from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from .serialization import BaseModelSerializer


class ExpLrScheduler(object):
    """Decay learning rate by a factor every `lr_decay_every_epochs`.

    Based on https://discuss.pytorch.org/t/fine-tuning-squeezenet/3855/7
    """

    def __init__(
        self, init_lr=0.001, decay_factor=0.1, lr_decay_every_epochs=7, verbose=False
    ):
        self.init_lr = init_lr
        self.decay_factor = decay_factor
        self.lr_decay_every_epochs = lr_decay_every_epochs
        self.verbose = verbose

    def __call__(self, optimizer, epoch):
        lr = self.init_lr * (self.decay_factor ** (epoch // self.lr_decay_every_epochs))

        if self.verbose and (epoch % self.lr_decay_every_epochs == 0):
            print("LR is set to {}".format(lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return optimizer


def _numpy_to_dataloader(X, y=None, *args, **kwargs):
    X = torch.from_numpy(X)
    if y is not None:
        y = torch.from_numpy(y)
    return _torch_to_dataloader(X, y, *args, **kwargs)


def _torch_to_dataloader(X, y=None, *args, **kwargs):
    tensors = [X]
    if y is not None:
        tensors.append(y)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, *args, **kwargs)


def _input_to_dataloader(X, y=None, offset=None, max_examples=None, *args, **kwargs):
    if offset is None:
        offset = 0
    if max_examples is not None:
        max_examples = len(X)
    X_slice = X[offset:max_examples]
    y_slice = y[offset:max_examples] if y is not None else None
    if isinstance(X, np.ndarray):
        return _numpy_to_dataloader(X_slice, y_slice, *args, **kwargs)
    elif isinstance(X, torch.Tensor):
        return _torch_to_dataloader(X_slice, y_slice, *args, **kwargs)
    else:
        raise NotImplementedError()


class TorchWrapperSerializer(BaseModelSerializer):
    """Torch wrapper serializer."""

    def __init__(self, model_fn, prefix, verbose=False):
        super().__init__(model_fn, prefix)
        self.verbose = verbose

    def _get_val_data_path(self, model_id):
        return os.path.join(self._get_model_path(model_id, "__val_data"))

    def save(self, model_id, model):
        print("Saved %s" % model_id)
        torch.save(model.module_.state_dict(), self._get_model_path(model_id))
        torch.save(model.val_loader_, self._get_val_data_path(model_id))

    def load(self, model_id):
        model = self.model_fn()
        model.module_.load_state_dict(torch.load(self._get_model_path(model_id)))
        if self.verbose:
            print("Loaded %s" % model_id)
        return model


class TorchWrapper(object):
    """
    Simplified Keras/sklearn-like wrapper for a torch module.

    We know there's skorch, but it was a pain to debug.

    :param module: Torch module class
    :param criterion: Criterion class
    :param optimizer: Optimizer class
    :param dict module_params: Parameters to pass to the module
            on initialization.
    :param dict optimizer_params: Parameters to pass to the optimizer
            on initialization.
    :param lr_scheduler: Learning rate scheduler
    :param enable_cude: Whether to use CUDA
    :param ModelSerializer serializer: Model serializer to save the best model.
    """

    STATS_FMT = "[{:>5s}] loss: {:+.4f}, acc: {:.4f}"
    BEST_MODEL_ID = "__fit_best"

    def __init__(
        self,
        module,
        criterion,
        optimizer,
        module_params=None,
        optimizer_params=None,
        lr_scheduler=None,
        enable_cuda=True,
        serializer=None,
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.enable_cuda = enable_cuda
        self.serializer = serializer

        self.criterion_ = criterion()

        if module_params is None:
            module_params = {}
        self.module_ = module(**module_params)
        if enable_cuda:
            self.module_ = self.module_.cuda()

        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_ = optimizer(self.module_.parameters(), **optimizer_params)

        self.lr_scheduler = lr_scheduler
        if lr_scheduler is None:
            self.lr_scheduler = lambda opt, *args, **kwargs: opt

    def fit_step(self, batch, phase="train"):
        """
        Run a single training step.

        :param batch: A tuple of numpy batch examples and labels
        :param phase: Phase. One of ['train', 'val']. If in val, does not
                      update the model parameters.
        """
        self.module_.train(phase == "train")
        inputs, labels = batch
        batch_size = len(inputs)

        # Wrap them in Variable
        if self.enable_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Zero the parameter gradients.
        self.optimizer_.zero_grad()

        # Forward pass.
        outputs = self.module_(inputs)
        _, preds = torch.max(outputs.data, 1)

        # Calculating the loss.
        loss = self.criterion_(outputs, labels)

        # Backward + optimize only if in training phase.
        if phase == "train":
            loss.backward()
            self.optimizer_.step()

        # Batch statistics.
        batch_loss = float(loss)
        num_correct_preds = float(torch.sum(preds == labels.data))

        return batch_loss, batch_size, num_correct_preds

    def fit(
        self,
        X,
        y=None,
        batch_size=32,
        epochs=20,
        shuffle=True,
        validation_split=None,
        validation_data=None,
        verbose=False,
    ):
        """
        Fit a torch classifier.

        :param X: Dataset
        :type X: ``numpy.ndarray`` or ``torch.Tensor``.
        :param y: Labels
        :param batch_size: Batch size
        :param epochs: Number of epochs to run the training
        :param shuffle: Whether to shuffle the dataset
        :param validation_split: Ratio of data to use for training. E.g., 0.7
        :param validation_data: If ``validation_split`` is not specified,
                the explicit validation dataset.
        :param verbose: Whether to output the progress report.

        TODO: Add custom metrics.
        """
        max_train_examples = None
        val_offset = None

        if validation_split is not None:
            max_train_examples = int((1 - validation_split) * len(X))
            val_offset = int(validation_split * len(X))
            validation_data = (X, y)

        train_loader = _input_to_dataloader(
            X,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            max_examples=max_train_examples,
        )

        phases = ["train"]
        if validation_data is not None:
            self.val_loader_ = val_loader = _input_to_dataloader(
                *validation_data,
                batch_size=batch_size,
                shuffle=shuffle,
                offset=val_offset
            )
            phases.append("val")
            best_val_loss = 0.0

        since = time.time()

        for epoch in range(epochs):
            if verbose:
                print("Epoch %d/%d" % (epoch + 1, epochs))

            # Each epoch has a training and validation phase.
            for phase in phases:
                if phase == "train":
                    self.optimizer_ = self.lr_scheduler(self.optimizer_, epoch)
                    data_iter = train_loader
                else:
                    data_iter = val_loader

                if verbose and phase == "train":
                    prog_bar = tqdm.tqdm(total=len(data_iter.dataset))

                epoch_dataset_size = 0
                running_loss = 0.0
                running_num_correct_preds = 0

                # Run through the data in minibatches.
                for data in data_iter:
                    batch_loss, batch_size, num_correct_preds = self.fit_step(
                        data, phase=phase
                    )

                    # Batch statistics.
                    epoch_dataset_size += batch_size
                    running_loss += batch_loss
                    running_num_correct_preds += num_correct_preds

                    if verbose and phase == "train":
                        batch_stats_str = TorchWrapper.STATS_FMT.format(
                            phase,
                            batch_loss / batch_size,
                            num_correct_preds / batch_size,
                        )
                        prog_bar.set_description(batch_stats_str)
                        prog_bar.update(batch_size)

                # Epoch statistics.
                epoch_loss = running_loss / epoch_dataset_size
                epoch_acc = running_num_correct_preds / epoch_dataset_size

                if verbose:
                    prog_bar.close()
                    epoch_stats_str = TorchWrapper.STATS_FMT.format(
                        phase, epoch_loss, epoch_acc
                    )
                    print(epoch_stats_str)

                # Determine if model is the best.
                if phase == "val" and self.serializer is not None:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        # TODO: Needs more work. Check the temperature
                        #       scaling class. Neural Network should change.
                        self.serializer.save(TorchWrapper.BEST_MODEL_ID)
                        if verbose:
                            print("New best accuracy: %.4f" % best_loss)

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        # Load the best model.
        if "val" in phases and self.serializer is not None:
            model_wrapper = self.serializer.load(TorchWrapper.BEST_MODEL_ID)
            self.module_ = model_wrapper.module_
            self.val_loader_ = model_wrapper.val_loader_

        return self.module_

    def predict_proba(self, X, batch_size=32):
        """Get the confidence vector for an evaluation of a trained model.

        :param X: Data
        :param batch_size: Batch size

        TODO: Fix in case this is not one-hot.

        """
        data_iter = _input_to_dataloader(X, batch_size=batch_size)

        batch_outputs = []
        for batch in data_iter:
            x = batch[0]
            if self.enable_cuda:
                x = x.cuda()
            outputs = self.module_(x).data
            batch_outputs.append(outputs)
        return np.vstack(batch_outputs)

    def predict(self, X, batch_size=32):
        """Get the confidence vector for an evaluation of a trained model.

        :param X: Data
        :param batch_size: Batch size
        """
        return np.argmax(self.predict_proba(X, batch_size), axis=1)
