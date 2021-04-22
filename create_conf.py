import torch
import time
import logging
import os
import json


def is_zero_file(fpath):
    return not os.path.isfile(fpath) or os.path.getsize(fpath) == 0


def get_checkpoint_folder_name(conf):
    # concat them together.
    directory = f"{conf.root_dir}/{time.time():.0f}_optim_{conf.optimizer}_"
    directory += f"activation_{conf.activation}_loss_{conf.loss}_"
    directory += f"samplecomplexity_{conf.sample_complexity}_"
    directory += f"nfeatures_{conf.n_features}_nhidden_{conf.n_hidden}_"
    directory += f"batchsize_{conf.batch_size}_lr_{conf.lr}_"
    directory += f"momentum_{conf.momentum_factor}_nesterov_{conf.use_nesterov}_"
    directory += f"nepochs_{conf.n_epochs}_"
    directory += f"ntest_{conf.n_test}_startseed_{conf.start_seed}_"
    directory += f"nruns_{conf.n_runs}"
    return directory


def build_dirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(" encounter error: {}".format(e))


class Logger:
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, file_folder):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        logging.basicConfig(filename=file_folder + "/log.log", level=logging.INFO)
        self.file_folder = file_folder
        self.file_json = os.path.join(file_folder, "log-1.json")
        self.file_txt = os.path.join(file_folder, "log.txt")
        self.file_csv = os.path.join(file_folder, "log.csv")
        self.values = []

    def log(self, *messages):
        print(*messages)
        logging.info(*messages)

    def info(self, *messages):
        print(*messages)
        logging.info(*messages)

    def log_metric(self, name, values, tags, display=False):
        """
        Store a scalar metric
        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})
        if display:
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def log_manual(self, value, display=True):
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        print(content)
        self.save_txt(content)

    def save_json(self):
        """
        Save the internal memory to a file
        """
        with open(self.file_json, "w") as fp:
            json.dump(self.values, fp, indent=" ")

        # reset 'values' and redirect the json file to other name.
        if self.meet_cache_limit():
            self.values = []
            self.redirect_new_json()

    def save_txt(self, value):
        utils.write_txt(value + "\n", self.file_txt, type="a")

    def save_csv(self, to_report):
        files = open(self.file_csv, "a")
        if is_zero_file(self.file_csv):
            files.write(",".join(to_report.keys()) + "\n")
        files.write(",".join([str(x) for x in to_report.values()]) + "\n")
        files.close()

    def redirect_new_json(self):
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file for file in os.listdir(self.file_folder) if "json" in file
        ]
        self.file_json = os.path.join(
            self.file_folder, "log-{}.json".format(len(existing_json_files) + 1)
        )

    def meet_cache_limit(self):
        return True if len(self.values) > 1e4 else False


def create_conf(conf):
    conf.n_train = int(conf.sample_complexity * conf.n_features)
    if conf.n_test is None:
        conf.n_test = conf.n_train
    if conf.optimizer == "gd":
        conf.batch_size = conf.n_train
    if (conf.optimizer in ["p-sgd", "sgd"]) and conf.psgd_adaptive_bs:
        conf.batch_size = int(conf.batch_size * conf.sample_complexity)

    conf.directory = get_checkpoint_folder_name(conf)
    build_dirs(conf.directory)
    conf.logger = Logger(conf.directory)
    if conf.optimizer == "p-sgd":l
        lr_over_tau = conf.lr / conf.persistence_time
        conf.logger.info("Persistent SGD switch probabilities:")
        conf.logger.info(f"from 0 to 1: {lr_over_tau}")
        conf.logger.info(f"from 1 to 0: {(conf.n_train - conf.batch_size) / conf.batch_size * lr_over_tau}")
        assert lr_over_tau <= 1.0
        assert lr_over_tau * (1 - conf.batch_size) / conf.batch_size  <= 1.0

    conf.logger.info(f"Loss function: {conf.loss}, with epsilong: {conf.loss_eps}")
    return conf
