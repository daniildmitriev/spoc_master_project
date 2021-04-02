import numpy as np
import torch
from copy import copy
from torch.utils.data import TensorDataset


def model(data, weights):
    """
    let:
    N be the number of samples
    P be the number of features
    M be the number of neurons in the hidden layer
    params:
    data: input data, P x N
    weights: M x P
    """
    return torch.mean(torch.matmul(weights, data) ** 2, axis=0)


def loss(y_pred, y_true):
    return torch.sum((y_pred - y_true) ** 2)


def error(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def check_success_sgd(
    conf, train_loader, test_loader, optimizer, weights, verbose=False
):
    train_errors = []
    test_errors = []
    max_gradient = None
    for epoch in range(int(conf.n_epochs * np.log2(conf.n_features + 1))):
        train_error = 0
        train_n_batches = 0

        # Train 
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.T
            batch_labels = batch_labels.T
            optimizer.zero_grad()
            y_pred = model(batch_data, weights)
            loss(y_pred, batch_labels).backward()
            optimizer.step()
            train_error += error(y_pred, batch_labels).item()
            train_n_batches += 1
            cur_max_gradient = torch.max(torch.abs(weights.grad))
            if max_gradient is None:
                max_gradient = cur_max_gradient
            # perform exponential averaging of the max gradient value
            max_gradient = 0.9 * max_gradient + 0.1 * cur_max_gradient

        # Test
        test_error = 0
        test_n_batches = 0
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.T
            batch_labels = batch_labels.T
            y_pred = model(batch_data, weights)
            test_error += error(y_pred, batch_labels).item()
            test_n_batches += 1
        test_error /= test_n_batches
        train_error /= train_n_batches
        if epoch % conf.verbose_freq == 0:
            to_log = f"#E: {epoch} \t"
            to_log += f"Train E: {train_error:.5f} \t Test E: {test_error:.5f}"
            to_log += f"\t Avg Max Grad: {max_gradient:.8f}"
            conf.logger.info(to_log)
        train_errors.append(train_error)
        test_errors.append(test_error)
        if (
            train_error < conf.train_threshold
            or test_error < conf.test_threshold
            or max_gradient < conf.gradient_threshold
        ):
            return train_errors, test_errors
    return train_errors, test_errors
