import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import torch

"""all checkpoints should be included at the moment

implemented callbacks:
    Recorder: Tracking train/valid loss and setattr[yb, out, loss] for all child classes
    CudaCallback: Manages devices and where data is send to
    Monitor: Tracks 
"""


class Callback():
    """parent class for all Callbacks
    implements dummy methods
    """
    def __init__(self, learn): 
        self.learn = learn

    def on_train_begin(self, learn, epochs):
        self.learn = learn
        self.epochs

    def on_train_end(self): pass

    def on_epoch_begin(self, epoch, *args):
        self.epoch = epoch

    def on_epoch_end(self): pass
    def on_batch_begin(self, *args): pass
    def on_batch_end(self): pass
    def on_loss_begin(self): pass

    def on_loss_end(self, loss, out, yb):
        self.loss = loss

    def on_step_begin(self): pass
    def on_step_end(self): pass
    def on_validate_begin(self): pass
    def on_validate_end(self, *args): pass


class CallbackHandler():
    def __init__(self, cbs):
        self.cbs = cbs
        for cb in self.cbs:
            setattr(self, type(cb).__name__, cb)

    def on_train_begin(self, learn, epochs):
        self.learn = learn
        for cb in self.cbs:
            cb.on_train_begin(self.learn, epochs)
        # self.train_mode = True

    def on_epoch_begin(self, epoch):
        self.learn.model.train()
        for cb in self.cbs:
            cb.on_epoch_begin(epoch)

    def on_epoch_end(self):
        for cb in self.cbs:
            cb.on_epoch_end()

    def on_batch_begin(self, batch):
        for cb in self.cbs:
            cb.on_batch_begin(batch)

    def on_batch_end(self):
        for cb in self.cbs:
            cb.on_batch_end()

    def on_loss_end(self, loss, out, yb):
        for cb in self.cbs:
            cb.on_loss_end(loss, out, yb)
        return self.learn.model.training

    def on_validate_begin(self):
        self.learn.model.eval()
        for cb in self.cbs:
            cb.on_validate_begin()

    def on_validate_end(self, loss):
        for cb in self.cbs:
            cb.on_validate_end(loss)


class Recorder(Callback):
    """ Tracking of  train/valid loss and setattr[yb, out, loss] to be available in all child classes

    Args:
        Callback (self): Implements alls methods
    """
    # todo: set self.history_raw on epoch end, in case that Monitor is not included
    #  * using numpy arrays for summming vals is much faster than lists

    def __init__(self, learn):
        super().__init__(learn)
        self.learn = learn
        if self.learn.resume:
            _, _, self.learn.history_raw, _ = load_checkpoint(
                self.learn.resume, self.learn.model, self.learn.opt)
        
    def on_train_begin(self, learn, epochs):
        # self.learn = learn
        self.epochs = epochs
        self.batch_vals = {"train_loss": [], "valid_loss": [],
                           "train_pred": [], "valid_pred": []}

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        self.batch_vals = {"train_loss": [], "valid_loss": [],
                           "train_pred": [], "valid_pred": []}

    def on_batch_begin(self, batch):
        self.batch = batch

    def on_loss_end(self, loss, out, yb):
        self.yb = yb
        self.loss = loss
        self.out = out
        if self.learn.model.training:
            self.batch_vals["train_loss"].append(loss.item())
        else:
            self.batch_vals["valid_loss"].append(loss.item())


class CudaCallback(Callback):
    """Manages if data/ model is send to gpu or cpu

    Args:
        Callback (self): Implements alls methods
    """

    def __init__(self, *args): pass

    def on_train_begin(self, learn, *args):
        self.learn = learn
        if self.learn.gpu:
            # todo: error message if no gpu available
            learn.model = learn.model.to(learn.device)

    def on_batch_begin(self, batch):
        self.xb, self.yb = batch[0], batch[1]
        # todo: uncomment when CNN
        self.xb = self.xb.unsqueeze(1)
        if self.learn.gpu:
            self.xb = self.xb.to(self.learn.device)
            self.yb = self.yb.to(self.learn.device)
        self.batch = (self.xb, self.yb)


class Monitor(Recorder):
    """Monitors custom metrics

    Args:
        Recorder (_type_): _description_
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.history = {"epochs": []}
        self.best_values = {}

    def on_train_begin(self, learn, epochs):
        self.learn = learn
        self.epochs = epochs

        self.batch_vals = {"train_loss": [], "valid_loss": [],
                           "train_pred": [], "valid_pred": []}
        for mon in self.monitor:
            self.history[mon] = []
        if not self.learn.resume:
            setattr(self.learn, "history_raw", self.history)
        else:
            self.history = self.learn.history_raw
    def on_batch_end(self):
        _, batch_pred = torch.max(self.out.data, 1)
        batch_correct = (batch_pred == self.yb).sum().item() / len(self.yb)
        if self.learn.model.training:
            self.batch_vals["train_pred"].append(batch_correct)
        else:
            self.batch_vals["valid_pred"].append(batch_correct)

    def on_epoch_end(self):
        for mon in self.monitor:
            self.history[mon].append(getattr(self, mon)())
            self.best_values[mon] = max(self.history[mon])
        self.history["epochs"].append(int(self.epoch+1))

        if self.verbose == True:
            self._print_console()

        setattr(self.learn, "history_raw", self.history)


    def valid_acc(self):
        return sum(self.batch_vals["valid_pred"]) / len(self.batch_vals["valid_pred"])

    def valid_loss(self):
        return sum(self.batch_vals["valid_loss"]) / len(self.batch_vals["valid_loss"])

    def train_loss(self):
        return sum(self.batch_vals["train_loss"]) / len(self.batch_vals["train_loss"])

    def _print_console(self):
        out_string = f""
        out_string += f"epoch: {int(self.epoch)+1}/{self.epochs}\t["
        for key, val in self.history.items():
            if key != "epochs":
                out_string += f"{key}: {val[-1]:.4f}\t"
        print(out_string[:-1] + "]")


class TrackValues(Callback):
    """Class tracks best values for train/ valid loss/acc

    Args:
        Callback: inherits self.learn.history_raw attribute from Callback class. self.learn.history_raw stores all implemented
                  metric values for each epoch

    Attr:
        track_best_vals: Saves best values from self.learn.history_raw
    """

    def __init__(self, learn):
        self.track_best_vals = {}
        self.learn = learn
        if self.learn.resume:
            _, _, _, self.track_best_vals = load_checkpoint(
                self.learn.resume, self.learn.model, self.learn.opt)
            
        
    def on_epoch_end(self):
        for monitor, values in self.learn.history_raw.items():
            if ("loss" in monitor) and (self.epoch == 0):
                self.track_best_vals[monitor] = [np.less, values[0]]
            if ("loss" in monitor) and (self.epoch > 0):
                comp = self.track_best_vals[monitor][0]
                best_val = self.track_best_vals[monitor][1]
                new_val = self.learn.history_raw[monitor][-1]
                if comp(new_val, best_val):
                    self.track_best_vals[monitor][1] = new_val
            if ("acc" in monitor) and (self.epoch == 0):
                self.track_best_vals[monitor] = [np.greater, values[0]]
            if ("acc" in monitor) and (self.epoch > 0):
                comp = self.track_best_vals[monitor][0]
                best_val = self.track_best_vals[monitor][1]
                new_val = self.learn.history_raw[monitor][-1]
                if comp(new_val, best_val):
                    self.track_best_vals[monitor][1] = new_val


class EarlyStopping(TrackValues):
    def __init__(self, learn):
        super().__init__(learn)
        self.monitor = "train_loss"
        self.patience = 20
        self.counter = 0
        self.delta = 1e-1

    def on_train_begin(self, learn, *args):
        self.learn = learn
        if "loss" in self.monitor:
            self.comp = np.less
            self.best_val = np.inf
        if "acc" in self.monitor:
            self.comp = np.greater
            self.best_val = -np.inf

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        if epoch > 0:
            diff = np.abs(self.best_val -
                          self.track_best_vals[self.monitor][1])
            if self.comp(self.best_val, self.track_best_vals[self.monitor][1]):
                self.counter += 1
            else:
                if diff > self.delta:
                    self.counter = 0
                    self.best_val = self.track_best_vals[self.monitor][1]
                else:
                    self.counter += 1
            if self.counter == self.patience:
                self.learn.do_stop = True


class Checkpoints(TrackValues):
    # todo: finish docstring
    """Saves history and Pytorch models during training

    Args:
        TrackValues: Parent class tracks best values for train/ valid loss/acc

    Attr:
        VarAttr:
            monitor(str): quantity to be monitored
            history_format(fileformat): fileformat for panda.DataFrame.to_*'s method. [parquet for high compression and fast reading]
            delta(float): min change in monitored quantity to qualify as improvement 
            ckp_path(str): path to save checkpoints
            no_time_path(str): if not specified in toml the current date is used as checkpoint foldername
            use_last_run(bool): if true, no new run directory is created. all files are saved in last run directory
            detailed_name(bool): if true, file names are saved with arch/bs/monitor information
            debug_timestamp(bool): if true, each run is saved in the last run folder, but with timeformat h/m/s
            resume(str): if specified in toml: history and model is loaded, before resuming training 
        FuncAttr:
            on_train_begin: creates folder struct for new run and initiates function for comparison of best and new value
            on_epoch_begin: checks for new best value -> saves history and model statedict
            create_checkpoint_path: creates checkpoint directory structure

    Deps:
        Modules: on_train_begin -> datetime/ pytz/ np
        Functions: on_train_begin -> self.create_checkpoint_path
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.monitor = "train_loss"
        self.history_format = "csv"
        self.delta = 1e-1

    def on_train_begin(self, learn, *args):
        self.save_path = self.create_checkpoint_path()
        self.learn = learn

        if self.detailed_name:
            self.save_name = f"_Arch-{self.learn.arch}_bs-{self.learn.bs}_{self.monitor}"
        else:
            self.save_name = f""

        if type(self.debug_timestamp) is bool:
            if self.debug_timestamp:
                timezone = pytz.timezone("Europe/Berlin")
                time = datetime.now()
                time = timezone.localize(time).strftime("%Y-%m-%dT%H_%M_%ST%z")
                self.save_name += f"_{time}"
        else:
            self.save_name += f"_{self.debug_timestamp}"

        if "loss" in self.monitor:
            self.comp = np.less
            self.best_val = np.inf
        if "acc" in self.monitor:
            self.comp = np.greater
            self.best_val = -np.inf
            
        if self.learn.resume:
            self.learn.model, self.learn.opt, self.learn.history_raw, self.track_best_vals = load_checkpoint(
            self.learn.resume, self.learn.model, self.learn.opt)
            self.comp, self.best_val = self.track_best_vals[self.monitor][0], self.track_best_vals[self.monitor][1]

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        if self.epoch > 0:
            diff = np.abs(self.best_val -
                          self.track_best_vals[self.monitor][1])

            if self.save_model:
                checkpoint = {
                    "best_history": self.track_best_vals,
                    "history": self.learn.history_raw,
                    "state_dict": self.learn.model.state_dict(),
                    "optimizer": self.learn.opt.state_dict()
                }

                save_checkpoint(checkpoint, False, Path(
                    self.save_path / f"model_{self.save_name}"))

                if not self.comp(self.best_val, self.track_best_vals[self.monitor][1]):
                    if diff > self.delta:
                        print(f"best checkpoint: {self.monitor}: {self.best_val}")
                        self.best_val = self.track_best_vals[self.monitor][1]
                        save_checkpoint(checkpoint, True, Path(
                            self.save_path / f"model_{self.save_name}"))

            if self.save_history:
                df = pd.DataFrame(self.learn.history_raw).set_index("epochs")
                to_func = getattr(df, "to_" + self.history_format)
                to_func(
                    Path(self.save_path / f"history{self.save_name}.{self.history_format}"))

    def create_checkpoint_path(self):
        """Creating Checkpoint directory structure according to attributs set in setup.toml

        Returns:
            run_path(str): path to created checkpoint directory
        """
        if hasattr(self, "no_time_path"):
            datetime_now = self.no_time_path
        else:
            datetime_now = datetime.now().strftime("%Y-%m-%d")

        curr_path = Path(self.ckp_path + "/" + datetime_now)
        curr_path.mkdir(parents=True, exist_ok=True)

        run_dirs = []
        for path in curr_path.iterdir():
            if path.is_dir():
                run_dirs.append(path.name)

        if (self.use_last_run and run_dirs):
            run_path = curr_path / run_dirs[-1]
        elif not self.use_last_run and run_dirs:
            run_num = int(run_dirs[-1][-3:]) + 1
            for i in range(3-len(str(run_num))):
                run_num = "0" + str(run_num)
            run_path = curr_path / Path("run_" + run_num)
            run_path.mkdir(parents=True, exist_ok=True)
        # * not sure if this is needed: (use_last_run and not run_dirs) or (not use_last_run and not run_dirs):
        else:
            run_path = curr_path / Path("run_001")
            run_path.mkdir(parents=True, exist_ok=True)
        return run_path


def save_checkpoint(state, is_best, checkpoint_path):
    """save best and last pytorch model in checkpoint_path

    Args:
        state(dict): checkpoint to save
        is_best(bool): getting bool from metric function
        checkpoint_path(str): path to save checkpoint
    """
    save_path = checkpoint_path.as_posix() + ".pt"
    torch.save(state, save_path)
    if is_best:
        best_path = checkpoint_path.as_posix() + "_best.pt"
        shutil.copyfile(save_path, best_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    """loads pytorch model and history

    Args:
        checkpoint_path(str): path to save checkpoint
        model(nn.Model): model to load checkpoint parameters into
        optimizer(torch.optim): in previous training defined optimizer
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    history = checkpoint["history"]
    best_history = checkpoint["best_history"]
    return model, optimizer, history, best_history


def get_callbacks(setup_config, learn):
    """Loading Callback classes according to setup.toml

    Args:
        setup_config (dict): contains information of callbacks and attributes

    Returns:
        list: returns list of all callback classes
    """
    implemented_cbs = {"m": Monitor(learn),
                       "e": EarlyStopping(learn),
                       "c": Checkpoints(learn)}

    cb_list = [c for c in setup_config if c[:2] == "c_"]
    cb_list
    cb_args = {}
    for i in cb_list:
        cb = i.split("_", 2)[:]
        if cb[1] not in cb_args:
            cb_args[cb[1]] = {cb[2]: setup_config[i]}
        else:
            cb_args[cb[1]][cb[2]] = setup_config[i]
    cbs = []
    for _cb, cb_list in cb_args.items():
        # important, that classes get instantiated here
        cb = implemented_cbs[_cb]
        for attr, val in cb_list.items():
            setattr(cb, attr, val)
        cbs.append(cb)
    return cbs


def get_callbackhandler(setup_config, learn):
    """Creates Callbackhandler with Callbacks from setup.toml

    Args:
        setup_config (dict): contains information of callbacks and attributes
    Deps:
        get_callbacks(dict): loading and returning callback classes according to setup.toml as list 
    Returns:
        CallbackHandler: returns CallbackHandler class with all callbacks

    Adds:
        Recorder and CudaCallback are added automatically in case of no Callbacks in setup.toml
    """
    if any([c for c in setup_config.keys() if c[:2] == "c_"]):
        cbs = [Recorder(learn), CudaCallback(learn)]
        cbs.extend(get_callbacks(setup_config, learn))
    else:
        cbs = [Recorder(learn), CudaCallback(learn)]
    return CallbackHandler(cbs)
