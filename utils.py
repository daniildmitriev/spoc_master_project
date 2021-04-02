import torch
import numpy as np
import scipy.stats as sps
from torch.utils.data import Sampler


def create_dataset(n_train, n_test, n_features, activation="quadratic"):
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
    train_data = torch.normal(
        0, std=1 / np.sqrt(n_features), size=(n_features, n_train)
    )
    test_data = torch.normal(0, std=1 / np.sqrt(n_features), size=(n_features, n_test))
    if activation == "quadratic":
        train_labels = teacher_weights.matmul(train_data) ** 2
        test_labels = teacher_weights.matmul(test_data) ** 2
    elif activation == "absolute":
        train_labels = torch.abs(teacher_weights.matmul(train_data))
        test_labels = torch.abs(teacher_weights.matmul(test_data))
    return train_data, train_labels, test_data, test_labels

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
            yield np.nonzero(self.cur_samples)
            from_zero_to_one = sps.bernoulli.rvs(p=self.eta / self.tau, size=self.n)
            from_one_to_zero = sps.bernoulli.rvs(
                p=(1 - self.b) * self.eta / (self.b * self.tau), size=self.n
            )
            self.cur_samples = (
                self.cur_samples * (1 - from_one_to_zero)
                + (1 - self.cur_samples) * from_zero_to_one
            )

    def __len__(self):
        return self.num_iters
