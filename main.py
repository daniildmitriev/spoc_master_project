import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import parameters
from create_conf import create_conf
from simulation import create_dataset
from simulation import check_success_sgd

if __name__ == "__main__":
    conf = parameters.get_args()
    conf = create_conf(conf)
    if conf.optimizer == "gd":
        assert conf.batch_size == conf.n_train
    np.random.seed(conf.start_seed)
    torch.manual_seed(conf.start_seed)
    train_data, train_labels, test_data, test_labels = create_dataset(conf.n_train, 
                                                                      conf.n_test, 
                                                                      conf.n_features)

    train_loader = DataLoader(TensorDataset(train_data.T, 
                                            train_labels.T), 
                              batch_size=conf.batch_size)
    test_loader = DataLoader(TensorDataset(test_data.T, 
                                           test_labels.T), 
                             batch_size=conf.batch_size)

    for seed in range(conf.start_seed + 1, conf.start_seed + conf.n_runs + 1):
        np.random.seed(seed)
        torch.manual_seed(seed)
        weights = torch.randn(conf.n_hidden, 
                              conf.n_features, 
                              requires_grad=True)
        conf.logger.info(f"initial weights: {weights[:5]}")
        optimizer = torch.optim.SGD([weights], 
                                    lr=conf.lr, 
                                    momentum=conf.momentum_factor, 
                                    weight_decay=conf.weight_decay)
        train_losses, test_losses = check_success_sgd(conf,
                                                      train_loader,
                                                      test_loader,
                                                      optimizer,
                                                      weights,
                                                      verbose=False)
        conf.logger.save_csv({'seed': seed, 
                              'epoch': (len(train_losses) - 1) * conf.verbose_freq
                              'train loss': train_losses[-1], 
                              'test loss': test_losses[-1]})
        conf.logger.info(f"Run #{seed - conf.start_seed} completed")
        conf.logger.info(f"train loss: {train_losses[-1]:.10f} \t test loss: {test_losses[-1]:.10f}")
        
        