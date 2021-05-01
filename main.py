import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import parameters
from create_conf import create_conf
from simulation import check_success_sgd
from utils import create_dataset, PoisSampler

if __name__ == "__main__":
    conf = parameters.get_args()
    conf = create_conf(conf)
    if conf.optimizer == "gd":
        assert conf.batch_size == conf.n_train
    np.random.seed(conf.start_seed)
    torch.manual_seed(conf.start_seed)
    train_data, train_labels, test_data, test_labels = create_dataset(
        conf.n_train, 
        conf.n_test, 
        conf.n_features,
        data_type=conf.activation,
        K=conf.symmetric_door_channel_K
    )
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
            shuffle=True
        )
    test_loader = DataLoader(
        TensorDataset(test_data.T, test_labels.T), batch_size=conf.n_test
    )

    for seed in range(conf.start_seed + 1, conf.start_seed + conf.n_runs + 1):
        np.random.seed(seed)
        torch.manual_seed(seed)
        weights = torch.randn(conf.n_hidden, conf.n_features, requires_grad=True)
        conf.logger.info(f"initial weights: {weights[0]}")
        optimizer = torch.optim.SGD(
            [weights],
            lr=conf.lr,
            momentum=conf.momentum_factor,
            weight_decay=conf.weight_decay,
            nesterov=conf.use_nesterov
        )
        train_losses, train_errors, test_errors = check_success_sgd(
            conf, train_loader, test_loader, optimizer, weights, verbose=False
        )
        conf.logger.save_csv(
            {
                "seed": seed,
                "loss_eps": conf.loss_eps,
                "tau": conf.persistence_time,
                "symmetric_door_k": conf.symmetric_door_channel_K,
                "project_on_sphere": conf.project_on_sphere,
                "early_stopping_epochs": conf.early_stopping_epochs,
                "weightdecay": conf.weight_decay,
                "epoch": len(train_errors),
                "train loss": train_losses[-1],
                "train error": train_errors[-1],
                "test error": test_errors[-1],
            }
        )
        conf.logger.info(f"Run #{seed - conf.start_seed} completed")
        conf.logger.info(
            f"train error: {train_errors[-1]:.10f} \t test error: {test_errors[-1]:.10f}"
        )
