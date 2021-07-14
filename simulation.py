import numpy as np
import torch
from copy import deepcopy
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
    elif conf.activation == 'relu-quadratic':
        first_layer_output = torch.maximum(first_layer_output ** 2 * \
                                           torch.sign(first_layer_output), 
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
    grad_difs = []
    max_gradient = None
    best_loss = None
    best_loss_e = None
    cur_iter = 0
    saved_weights = []
    prev_grad = None
    grads = []
    momentums = []
    if conf.compute_hessian and conf.save_eigenvalues:
        eigenvalues = {'train': [], 'test': []}
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
            batch_loss = loss(conf, y_pred, batch_labels)
            batch_loss.backward()
            train_loss += batch_loss.item()
            if conf.optimizer == "langevin":
                if cur_iter >= len(conf.noise_std):
                    optimizer.step(0)
                else:
                    optimizer.step(conf.noise_std[cur_iter])
                cur_iter += 1
            else:
                optimizer.step()
            train_error += error(conf, y_pred, batch_labels).item()
            train_n_batches += 1
            cur_max_gradient = torch.max(torch.abs(weights.grad))
            if max_gradient is None:
                max_gradient = cur_max_gradient
            # perform exponential averaging of the max gradient value
            max_gradient = 0.9 * max_gradient + 0.1 * cur_max_gradient
            if conf.save_grads:
                batch_grad = deepcopy(weights.grad.data)
                grads.append(batch_grad)
            if conf.save_momentum:
                for _, param_state in optimizer.state.items():
                    momentums.append(deepcopy(param_state["momentum_buffer"]))
            # computing difference between true grad and batch grad
            if conf.compute_grad_dif:
                batch_grad = deepcopy(weights.grad.data)
                if prev_grad is None:
                    grad_difs.append(0)
                else:
                    grad_dif = torch.linalg.norm(batch_grad - prev_grad).item()
                    grad_difs.append(grad_dif)

                prev_grad = deepcopy(batch_grad)
            
            # projecting on sphere
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
            to_log = f"#E: {epoch} \t Train L: {train_loss:.7f} \t"
            to_log += f"Train E: {train_error:.7f} \t Test E: {test_error:.7f} \t"
            to_log += f"Grad: {max_gradient:.7f}"
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
            hessian_matrix_train = compute_hessian(conf, model, weights, train_loader)
            hessian_matrix_test = compute_hessian(conf, model, weights, test_loader)
            if conf.save_eigenvalues:
                n_params = conf.n_features * conf.n_hidden
                hessian_matrix_train = hessian_matrix_train.reshape((n_params, n_params))
                hessian_matrix_test = hessian_matrix_test.reshape((n_params, n_params))
                eigvals_train = np.linalg.eigh(hessian_matrix_train)[0]
                eigvals_test = np.linalg.eigh(hessian_matrix_test)[0]
                eigenvalues['train'].append(eigvals_train)
                eigenvalues['test'].append(eigvals_test)
            else:
                conf.logger.save_tensor(hessian_matrix_train, f"hessian_train_seed_{conf.cur_seed}", epoch)
                conf.logger.save_tensor(hessian_matrix_test, f"hessian_test_seed_{conf.cur_seed}", epoch)
    if conf.compute_hessian:
        hessian_matrix_train = compute_hessian(conf, model, weights, train_loader)
        hessian_matrix_test = compute_hessian(conf, model, weights, test_loader)
        if conf.save_eigenvalues:
            n_params = conf.n_features * conf.n_hidden
            hessian_matrix_train = hessian_matrix_train.reshape((n_params, n_params))
            hessian_matrix_test = hessian_matrix_test.reshape((n_params, n_params))
            eigvals_train = np.linalg.eigh(hessian_matrix_train)[0]
            eigvals_test = np.linalg.eigh(hessian_matrix_test)[0]
            eigenvalues['train'].append(eigvals_train)
            eigenvalues['test'].append(eigvals_test)
            conf.logger.save_pickle(eigenvalues, f"eigenvalues_seed_{conf.cur_seed}")
        else:
            conf.logger.save_tensor(hessian_matrix_train, f"hessian_train_seed_{conf.cur_seed}", epoch)
            conf.logger.save_tensor(hessian_matrix_test, f"hessian_test_seed_{conf.cur_seed}", epoch)
        conf.logger.save_tensor(weights, f"weights_seed_{conf.cur_seed}", epoch)
    if conf.compute_grad_dif:
        conf.logger.save_pickle(grad_difs, f"grad_difs_seed_{conf.cur_seed}")
    if conf.save_grads:
        conf.logger.save_pickle(grads, f"grads_seed_{conf.cur_seed}")
    if conf.save_momentum:
        conf.logger.save_pickle(momentums, f"momentums_seed_{conf.cur_seed}")
    return train_losses, train_errors, test_errors
