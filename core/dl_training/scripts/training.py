from core.dl_framework.utils import read_config
from core.dl_framework.data import get_dataset, Dataset
from core.dl_framework.learner import Learner
import click
import toml
from pathlib import Path
import sys

@click.command()
@click.argument("config_path", default="./configs/default_train_config.toml", type=click.Path(exists=True))
def main(config_path):
    setup_config = read_config(toml.load(config_path))
    x_train, y_train, x_test, y_test = get_dataset(
        setup_config["s_source"], setup_config["s_set"], setup_config["p_tmp_data_path"])

    train_ds, test_ds = Dataset(x_train, y_train), Dataset(x_test, y_test)

    learn = Learner(train_ds, setup_config)
    learn.fit(1000)

if __name__ == "__main__":
    main()


# calc runtime
# average runtime
# toml file
# add paths in setup for start training
# add mechanism to safe model to gdrive. with option to resume training
# add save best model
# add tensorboard support
# add telegramlogger
# check pytest
