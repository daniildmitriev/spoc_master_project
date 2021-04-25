import numpy as np
import torch
from copy import copy
from torch.utils.data import TensorDataset


def model(data, weights, activation='quadratic'):
    """
    let:
    N be the number of samples
    P be the number of features
    M be the number of neurons in the hidden layer
    params:
    data: input data, P x N
    weights: M x P
    """
    if activation == 'quadratic':
        return torch.mean(torch.matmul(weights, data) ** 2, axis=0)
    elif activation == 'absolute':
        return torch.mean(torch.abs(torch.matmul(weights, data)), axis=0)
    
    return torch.mean(torch.matmul(weights, data), axis=0)


def loss(conf, y_pred, y_true):
    if conf.loss == 'mse':
        return torch.sum((y_pred - y_true) ** 2)
    elif conf.loss == 'square':
        activation_eps = torch.sqrt(y_pred ** 2 + conf.loss_eps) - conf.symmetric_door_channel_K
        return 0.5 * torch.sum((y_true - activation_eps) ** 2)
    elif conf.loss == 'logloss':
        activation_eps = torch.sqrt(y_pred ** 2 + conf.loss_eps) - conf.symmetric_door_channel_K
        return torch.sum(torch.log(1 + torch.exp(-y_true * activation_eps)))
    


def error(conf, y_pred, y_true):
    if conf.loss == 'mse':
        return torch.mean((y_pred - y_true) ** 2)
    activation_eps = torch.abs(y_pred) - conf.symmetric_door_channel_K
    return 0.5 * torch.mean((torch.sign(activation_eps) - y_true)**2)


def check_success_sgd(
    conf, train_loader, test_loader, optimizer, weights, verbose=False
):
    train_losses = []
    train_errors = []
    test_errors = []
    max_gradient = None
    best_loss = None
    best_loss_e = None
    for epoch in range(int(conf.n_epochs * np.log2(conf.n_features + 1))):
        train_error = 0
        train_loss = 0
        train_n_batches = 0

        # Train 
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.T
            batch_labels = batch_labels.T
            optimizer.zero_grad()
            y_pred = model(batch_data, weights, conf.activation)
            cur_loss = loss(conf, y_pred, batch_labels)
            cur_loss.backward()
            train_loss += cur_loss.item()
            optimizer.step()
            train_error += error(conf, y_pred, batch_labels).item()
            train_n_batches += 1
            cur_max_gradient = torch.max(torch.abs(weights.grad))
            if max_gradient is None:
                max_gradient = cur_max_gradient
            # perform exponential averaging of the max gradient value
            max_gradient = 0.9 * max_gradient + 0.1 * cur_max_gradient
            
            # projecting on sphere
            if conf.project_on_sphere:
                weights_norm = torch.linalg.norm(weights, axis=1)
                weights = torch.div(weights, torch.unsqueeze(weights_norm, 1))
                weights *= np.sqrt(conf.n_features)
        # Test
        test_error = 0
        test_n_batches = 0
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.T
            batch_labels = batch_labels.T
            y_pred = model(batch_data, weights, conf.activation)
            test_error += error(conf, y_pred, batch_labels).item()
            test_n_batches += 1
        test_error /= test_n_batches
        train_error /= train_n_batches
        train_loss /= train_n_batches
        if epoch % conf.verbose_freq == 0:
            to_log = f"#E: {epoch} \t Train L: {train_loss:.5f} \t"
            to_log += f"Train E: {train_error:.3f} \t Test E: {test_error:.3f} \t"
            to_log += f"Grad: {max_gradient:.6f}"
            conf.logger.info(to_log)
        train_losses.append(train_loss)
        train_errors.append(train_error)
        test_errors.append(test_error)
        if best_loss is None or train_loss < best_loss:
            best_loss = train_loss
            best_loss_e = epoch
        elif (conf.early_stopping_epochs > 0 and 
              epoch - best_loss_e > conf.early_stopping_epochs):
            return train_losses, train_errors, test_errors
        if (
            train_error < conf.train_threshold
            or test_error < conf.test_threshold
            or max_gradient < conf.gradient_threshold
        ):
            return train_losses, train_errors, test_errors
    return train_losses, train_errors, test_errors
