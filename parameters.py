import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, default="default", help="root directory for logging"
    )

    parser.add_argument("--start_seed", type=int, default=5, help="random seed")
    parser.add_argument("--n_runs", type=int, default=20, help="number of simulations")
    parser.add_argument(
        "--train_threshold",
        type=float,
        default=1e-5,
        help="stop when train error is below threshold",
    )
    parser.add_argument(
        "--gradient_threshold",
        type=float,
        default=1e-3,
        help="stop when max absolute gradient component is below threshold",
    )
    parser.add_argument(
        "--test_threshold",
        type=float,
        default=1e-3,
        help="stop when test error is below threshold",
    )

    # simulation scheme
    parser.add_argument(
        "--n_train", type=int, default=None, help="number of samples in train dataset"
    )
    parser.add_argument(
        "--n_test", type=int, default=None, help="number of samples in test dataset"
    )
    parser.add_argument(
        "--sample_complexity",
        type=float,
        default=3.0,
        help="ratio of number of samples to number of features",
    )
    parser.add_argument("--n_features", type=int, help="number of features")
    parser.add_argument("--n_hidden", type=int, help="number of hidden units")

    parser.add_argument(
        "--n_epochs",
        default=10000,
        type=int,
        help="number of epochs (will be multiplied by log(n_features))",
    )
    parser.add_argument("--batch_size", default=32, type=int)

    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.0001)

    # momentum scheme
    parser.add_argument("--momentum_factor", default=0, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    parser.add_argument(
        "--weight_decay", default=0, type=float, help="weight decay (default: 1e-4)"
    )

    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="sgd", 
        help='"adam" for Adam, "sgd" for SGD, "p-sgd" for Persistent-SGD'
    )

    parser.add_argument(
        "--verbose_freq", type=int, default=100, help="How often print logs"
    )
    
    # persistent SGD param
    parser.add_argument(
        "--persistence_time", default=1.0, type=float, help="persistence time"
    )

    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
