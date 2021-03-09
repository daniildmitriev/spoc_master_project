import numpy as np
from copy import copy

def loss(w, data_, labels_):
    return np.sum((labels_**2 - w.dot(data_)**2)**2)
def loss_grad(w, data_, labels_):
    return -data_.dot((labels_**2 - w.dot(data_)**2) * w.dot(data_))

def check_success_sgd(data, 
                      labels,
                      init_weights=None,
                      _type='vanilla', 
                      n_epoch=1500, 
                      batch_size=64, 
                      lr=20.0, 
                      momentum_gamma=0.9,
                      verbose=False):
    losses = []
    p, n = data.shape
    if init_weights is None:
        cur_weights = np.random.normal(size=p)
        cur_weights /= np.linalg.norm(cur_weights) / np.sqrt(p)
    else:
        cur_weights = copy(init_weights)
    momentum = np.zeros_like(cur_weights)
    for epoch in range(int(n_epoch * np.log2(p + 1))):
        indices = np.arange(n)
        np.random.shuffle(indices)
        for batch_indices in np.array_split(indices, len(indices) // batch_size):
            batch_data = data[:, batch_indices]
            batch_labels = labels[batch_indices]
            if _type == 'vanilla':
                momentum = lr * loss_grad(cur_weights, batch_data, batch_labels)
            elif _type == 'momentum':
                momentum *= momentum_gamma
                momentum += lr * loss_grad(cur_weights, batch_data, batch_labels)
            elif _type == 'nesterov':
                momentum *= momentum_gamma
                momentum += lr * loss_grad(cur_weights - momentum, batch_data, batch_labels)
            else:
                raise ValueError('Bad type')
            cur_weights -= momentum
        if epoch % 10 == 0:
            cur_loss = loss(cur_weights, data, labels)
            if verbose:
                print(f"Epoch: {epoch}, loss: {cur_loss}")
            losses.append(cur_loss)
            if cur_loss < 1e-8:
                return 1, losses
    final_loss = loss(cur_weights, data, labels)
    losses.append(final_loss)
    if final_loss < 1e-8:
        return 1, losses
    return 0, losses