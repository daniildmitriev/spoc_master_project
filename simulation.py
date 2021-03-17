import numpy as np
import torch
from copy import copy
from torch.utils.data import TensorDataset

def create_dataset(n_train, n_test, n_features, activation='quadratic'):
    """
    params:
    n_train: number of samples in the train dataset
    n_test: number of samples in the test dataset
    n_features: number of features
    returns:
    train_data
    train_labels
    test_data
    test_labels
    """
    teacher_weights = torch.randn(n_features)
    train_data = torch.normal(0, std=1/np.sqrt(n_features), size=(n_features, n_train))
    test_data = torch.normal(0, std=1/np.sqrt(n_features), size=(n_features, n_test))
    if activation == 'quadratic':
        train_labels = teacher_weights.matmul(train_data) ** 2
        test_labels = teacher_weights.matmul(test_data) ** 2
    elif activation == 'absolute':
        train_labels = torch.abs(teacher_weights.matmul(train_data))
        test_labels = torch.abs(teacher_weights.matmul(test_data))    
    return train_data, train_labels, test_data, test_labels

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
    return torch.mean(torch.matmul(weights, data)**2, axis=0)

def loss(y_pred, y_true):
    return torch.sum((y_pred - y_true)**2)

def error(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def check_success_sgd(conf,
                      train_loader,
                      test_loader,
                      optimizer,
                      weights,
                      verbose=False):
    train_losses = []
    test_losses = []
    max_gradient = None
    for epoch in range(int(conf.n_epochs * np.log2(conf.n_features + 1))):
        train_loss = 0
        train_error = 0 
        train_n_batches = 0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.T
            batch_labels = batch_labels.T
            optimizer.zero_grad()
            y_pred = model(batch_data, weights)
            cur_loss = loss(y_pred, batch_labels)
            cur_loss.backward()
            optimizer.step()
            train_loss += cur_loss.item()
            train_error += error(y_pred, batch_labels).item()
            train_n_batches += 1
            cur_max_gradient = torch.max(torch.abs(weights.grad))
            if max_gradient is None:
                max_gradient = cur_max_gradient
            # perform exponential averaging of the max gradient value
            max_gradient = 0.8 * max_gradient + 0.2 * cur_max_gradient
        if epoch % conf.verbose_freq == 0:
            test_loss = 0
            test_error = 0
            test_n_batches = 0
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.T
                batch_labels = batch_labels.T
                y_pred = model(batch_data, weights)
                test_loss += loss(y_pred, batch_labels).item()
                test_error += error(y_pred, batch_labels).item()
                test_n_batches += 1
            test_loss /= test_n_batches
            train_loss /= train_n_batches
            test_error /= test_n_batches
            train_error /= train_n_batches
            to_log = f"#E: {epoch} \t"
            to_log += f"Train E: {train_error:.5f} \t Test E: {test_error:.5f}"
            to_log += f"\t Avg Max Grad: {max_gradient:.3f} \t Cur Max Grad: {cur_max_gradient:.3f}"
            conf.logger.info(to_log)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if train_error < conf.threshold:
                return train_losses, test_losses
            if test_error < conf.test_threshold:
                return train_losses, test_losses
            if max_gradient < conf.gradient_threshold:
                return train_losses, test_losses
    return train_losses, test_losses
