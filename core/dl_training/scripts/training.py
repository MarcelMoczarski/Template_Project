import sys
from pathlib import Path

import click
import toml
from core.dl_framework.data import Dataset, get_dataset
from core.dl_framework.learner import Learner
from core.dl_framework.utils import read_config


@click.command()
@click.argument("config_path", default="./configs/default_train_config.toml", type=click.Path(exists=True))
def main(config_path):
    setup_config = read_config(toml.load(config_path))
    x_train, y_train, x_test, y_test = get_dataset(
        setup_config["s_source"], setup_config["s_set"], setup_config["p_tmp_data_path"], CNN=True)

    train_ds, test_ds = Dataset(x_train, y_train), Dataset(x_test, y_test)

    learn = Learner([train_ds, test_ds], setup_config)

    # learn = Learner(train_ds, setup_config)
    
    learn.fit(setup_config["g_num_epochs"])


if __name__ == "__main__":
    main()


# calc runtime
# average runtime
# add tensorboard support
# add telegramlogger
# check pytest
# tensorflow
# transferlearning
# gridsearch
# model arch via blocks
# test c ext for faster calc
