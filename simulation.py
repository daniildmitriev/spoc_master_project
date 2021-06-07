import numpy as np
import torch
from copy import copy
from torch.utils.data import TensorDataset
from torch.autograd.functional import hessian

def model(conf, data, weights, train=True):
    """
    let:
    N be the number of samples
    P be the number of features
    M be the number of neurons in the hidden layer
    params:
    data: input data, P x N
    weights: M x P
    """
    
    # first layer
    first_layer_output = torch.matmul(weights, data)
    if conf.activation == 'quadratic':
        first_layer_output = first_layer_output ** 2
    elif conf.activation == 'absolute':
        if train:
            first_layer_output = torch.sqrt(first_layer_output ** 2 + conf.loss_eps)
        else:
            first_layer_output = torch.abs(first_layer_output)
    elif conf.activation == 'relu':
        first_layer_output = torch.maximum(first_layer_output, 
                                           torch.zeros_like(first_layer_output))
    first_layer_output = torch.mean(first_layer_output, axis=0) # + bias <- to learn
   

    # second layer
    if conf.second_layer_activation is None or conf.second_layer_activation == 'none':
        return first_layer_output
    elif conf.second_layer_activation == 'quadratic': # not sure if needed
        return first_layer_output ** 2
    elif conf.second_layer_activation == 'absolute': # not sure if needed
        if train:
            return torch.sqrt(first_layer_output ** 2 + conf.loss_eps)
        else:
            return torch.abs(first_layer_output)
    elif conf.second_layer_activation[:14] == 'symmetric-door':
        activation = first_layer_output
        return activation - conf.symmetric_door_channel_K
#         if conf.second_layer_activation[-8:] == 'absolute':
#             if train:
#                 activation = torch.sqrt(activation ** 2 + conf.loss_eps)
#             else:
#                 activation = torch.abs(activation)
#         # ask if K is known!
#         return 2 / (1 + torch.exp(-activation + conf.symmetric_door_channel_K)) - 1


def loss(conf, y_pred, y_true):
    if conf.loss == 'mse':
        return 0.5 * torch.sum((y_pred - y_true) ** 2)
    elif conf.loss == 'logloss':
        return torch.sum(torch.log(1 + torch.exp(-y_true * y_pred)))

def error(conf, y_pred, y_true):
    if conf.labels == 'symmetric-door':
        return 0.5 * torch.mean((torch.sign(y_pred) - y_true)**2)
    return 0.5 * torch.mean((y_pred - y_true) ** 2)

def compute_hessian(conf, model, weights, data_loader):
    """
    Computes Hessian of the loss at the given point.
    """
    
    # using for-loop, but only need first iteration
    for batch_data, batch_labels in data_loader:
        batch_data = batch_data.T
        batch_labels = batch_labels.T
        loss_func = lambda weights: loss(conf, 
                                         model(conf, batch_data, weights, train=True), 
                                         batch_labels)
        return hessian(loss_func, weights)

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
            y_pred = model(conf, batch_data, weights, train=True)
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
            
                # projecting on spherel
            if conf.project_on_sphere:
                with torch.no_grad():
                    w = weights.data
                    w_norm = torch.linalg.norm(w, axis=1)
                    w.div_(torch.unsqueeze(w_norm, 1) / np.sqrt(conf.n_features))
        # Test
        test_error = 0
        test_n_batches = 0
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.T
            batch_labels = batch_labels.T
            y_pred = model(conf, batch_data, weights, train=False)
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
            break
        if (
            train_error < conf.train_threshold
            or test_error < conf.test_threshold
            or max_gradient < conf.gradient_threshold
        ):
            break
        if conf.compute_hessian and (epoch % conf.compute_hessian_freq == 0):
            hessian_matrix = compute_hessian(conf, model, weights, train_loader)
            conf.logger.save_tensor(hessian_matrix, f"hessian_{epoch}e.pt")
    if conf.compute_hessian:
        hessian_matrix = compute_hessian(conf, model, weights, train_loader)
        conf.logger.save_tensor(hessian_matrix, f"hessian_{epoch}e.pt")
    return train_losses, train_errors, test_errors
