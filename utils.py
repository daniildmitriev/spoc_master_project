import torch
import numpy as np
import scipy.stats as sps
from torch.utils.data import Sampler, DataLoader, TensorDataset


class PoisSampler(Sampler[int]):
    r"""Samples elements as Poisson random process. Used in Persistent-SGD

    Args:
        data_source (Dataset): dataset to sample from
        expected_batch_size (int): how many elements on average should be sampled
        eta (floor): learning rate
        tau (float): persistence time
        num_iters (int): how many iterations to perform
    """

    def __init__(self, data_source, expected_batch_size, eta, tau=1, num_iters=None):
        self.data_source = data_source
        self.n = len(data_source)
        if num_iters is None:
            self.num_iters = int(np.ceil(self.n / expected_batch_size))
        else:
            self.num_iters = num_iters
        self.b = expected_batch_size / self.n
        self.eta = eta
        self.tau = tau
        self.cur_samples = sps.bernoulli.rvs(p=self.b, size=self.n)

    def __iter__(self):
        for _ in range(self.num_iters):
            yield np.nonzero(self.cur_samples)[0]
            from_zero_to_one = sps.bernoulli.rvs(p=self.eta / self.tau, size=self.n)
            from_one_to_zero = sps.bernoulli.rvs(
                p=(1 - self.b) * self.eta / (self.b * self.tau),
                size=self.n
            )
            self.cur_samples = (
                self.cur_samples * (1 - from_one_to_zero)
                + (1 - self.cur_samples) * from_zero_to_one
            )

    def __len__(self):
        return self.num_iters

def create_dataset(conf, teacher_weights=None):
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
    if teacher_weights is None:
        if conf.labels == "symmetric-door":
            teacher_weights = 2 * torch.randint(low=0, 
                                                high=2, 
                                                size=(conf.n_features,), 
                                                dtype=torch.float) - 1
        else:
            teacher_weights = torch.randn(conf.n_features)
            if conf.reverse_mult_by_sqrt:
                teacher_weights /= np.sqrt(conf.n_features)
    if conf.reverse_mult_by_sqrt:
        train_data = torch.normal(size=(conf.n_features, conf.n_train))
        test_data = torch.normal(size=(conf.n_features, conf.n_test))
    else:
        train_data = torch.normal(0, 
                                  std=1 / np.sqrt(conf.n_features), 
                                  size=(conf.n_features, conf.n_train))
        test_data = torch.normal(0, 
                                 std=1 / np.sqrt(conf.n_features), 
                                 size=(conf.n_features, conf.n_test))
    train_mult = teacher_weights.matmul(train_data)
    test_mult = teacher_weights.matmul(test_data)
    if conf.labels == "quadratic":
        train_labels = train_mult ** 2
        test_labels = test_mult ** 2
    elif conf.labels == "absolute":
        train_labels = torch.abs(train_mult)
        test_labels = torch.abs(test_mult)
    elif conf.labels == "symmetric-door":
        train_labels = torch.sign(torch.abs(train_mult) - conf.symmetric_door_channel_K)
        test_labels = torch.sign(torch.abs(test_mult) - conf.symmetric_door_channel_K)
    return train_data, train_labels, test_data, test_labels

def from_data_to_dataloader(conf, train_data, train_labels, test_data, test_labels):
    if conf.optimizer == "p-sgd":
        pois_sampler = PoisSampler(
            train_data.T, conf.batch_size, conf.lr, conf.persistence_time
        )
        train_loader = DataLoader(
            TensorDataset(train_data.T, train_labels.T), batch_sampler=pois_sampler
        )
    else:
        train_loader = DataLoader(
            TensorDataset(train_data.T, train_labels.T), 
            batch_size=conf.batch_size,
            shuffle=True,
            drop_last=True # to avoid rounding errors
        )
    test_loader = DataLoader(
        TensorDataset(test_data.T, test_labels.T), batch_size=conf.n_test
    )
    return train_loader, test_loader

def create_dataloaders(conf, teacher_weights=None):
    train_data, train_labels, test_data, test_labels = create_dataset(conf, teacher_weights)
    return from_data_to_dataloader(conf, train_data, train_labels, test_data, test_labels)
