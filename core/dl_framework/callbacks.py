import torch
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# from datetime import date

class Callback():
    def __init__(): pass

    def on_train_begin(self, learn):
        self.learn = learn

    def on_train_end(self): pass
    def on_epoch_begin(self, *args): pass
    def on_epoch_end(self): pass

    def on_batch_begin(self): pass

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

    def on_train_begin(self, learn):
        self.learn = learn
        for cb in self.cbs:
            cb.on_train_begin(self.learn)
        # self.learn = learn
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
    # using numpy arrays for summming vals is much faster than lists
    def __init__(self):
        self.epoch_vals = {"epoch": [], "train": [], "valid": []}
        self.batch_vals = {"train_loss": [], "valid_loss": [], "train_out": [], "valid_out": []}

    def on_epoch_begin(self, epoch):
        self.epoch_vals["epoch"].append(epoch + 1)
        self.batch_vals["train_loss"] = []
        self.batch_vals["valid_loss"] = []

    def on_batch_begin(self, batch):
        self.batch = batch

    def on_loss_end(self, loss, out, yb):
        self.yb = yb
        self.loss = loss
        self.out = out
        if self.learn.model.training:
            self.batch_vals["train_loss"].append(loss.item())
            self.batch_vals["train_out"].append(torch.max(out.data, 1))
        else:
            self.batch_vals["valid_loss"].append(loss.item())
            self.batch_vals["valid_out"].append(torch.max(out.data, 1))


    def history(self):
        pass


class CudaCallback(Callback):
    def __init__(self): pass

    def on_train_begin(self, learn):
        self.learn = learn
        if self.learn.gpu:
            # error message if no gpu available
            learn.model = learn.model.to(learn.device)

    def on_batch_begin(self, batch):
        self.xb, self.yb = batch[0], batch[1]
        if self.learn.gpu:
            self.xb = self.xb.to(self.learn.device)
            self.yb = self.yb.to(self.learn.device)
        self.batch = (self.xb, self.yb)


class Monitor(Recorder):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_train_begin(self, learn):
        self.learn = learn
        for mon in self.monitor:
            self.history[mon] = []

    def on_epoch_end(self):
        for mon in self.monitor:
            print(mon)
            self.history[mon].append(getattr(self, mon)())
            print(mon)

    def valid_acc(self):
        _, predicted = torch.max(self.out.data, 1)
        correct = (predicted == self.yb).sum().item() / self.learn.data.valid_dl.bs
        return correct
#         _, predicted = torch.max(kwargs["out"].data, 1)
#         correct = (predicted == kwargs["yb"]).sum(
#         ).item() / self.learn.data.valid_dl.bs
#         self.batch_vals["valid_acc"].append(correct)

    def valid_loss(self):
        return sum(self.batch_vals["valid_loss"]) / len(self.batch_vals["valid_loss"])

    def train_loss(self):
        return sum(self.batch_vals["train_loss"]) / len(self.batch_vals["train_loss"])


class EarlyStopping(Recorder):
    def __init__(self):
        super().__init__()


# class Callback():
#     def __init__(): pass

#     def on_train_begin(self, learn, epochs):
#         self.learn = learn
#         self.epochs = epochs

#     def on_train_end(self): pass
#     def on_epoch_begin(self, *args): pass
#     def on_epoch_end(self, *args): pass

#     def on_batch_begin(self, xb, yb):
#         self.xb = xb
#         self.yb = yb

#     def on_batch_end(self): pass
#     def on_loss_begin(self): pass

#     def on_loss_end(self, loss):
#         self.loss = loss

#     def on_step_begin(self): pass
#     def on_step_end(self): pass
#     def on_validate_begin(self): pass
#     def on_validate_end(self): pass


# class CallbackHandler():
#     def __init__(self, cbs):
#         self.cbs = cbs
#         for cb in cbs:
#             setattr(self, type(cb).__name__, cb)

#     def on_train_begin(self, learn, epochs):
#         self.learn = learn
#         self.epochs = epochs
#         for cb in self.cbs:
#             cb.on_train_begin(learn, self.epochs)

#     def on_train_end(self):
#         for cb in self.cbs:
#             cb.on_train_end()

#     def on_epoch_begin(self, epoch):
#         self.learn.model.train()
#         self.epoch = epoch + 1
#         for cb in self.cbs:
#             cb.on_epoch_begin(self.epoch)

#     def on_epoch_end(self):
#         for cb in self.cbs:
#             if type(cb).__name__ == "Monitor_Cb":
#                 cb.on_epoch_end()
#                 self.learn.recorder = cb.history
#             else:
#                 cb.on_epoch_end()

#     def on_batch_begin(self, xb, yb):
#         for cb in self.cbs:
#             cb.on_batch_begin(xb, yb)

#     def on_batch_end(self):
#         for cb in self.cbs:
#             cb.on_batch_end()

#     def on_loss_end(self, loss):
#         self.loss = loss
#         for cb in self.cbs:
#             cb.on_loss_end(loss)

#     def on_validate_begin(self):
#         self.learn.model.eval()
#         for cb in self.cbs:
#             cb.on_validate_begin()

#     def on_validate_end(self):
#         for cb in self.cbs:
#             cb.on_validate_end()

# class Recorder_Cb(Callback):
#     def __init__(self, ):
#         self.string = string

# class Monitor_Cb(Recorder_Cb):
#     def __init__(self, monitor, print2console=True):
#         super().__init__("test")
#         self.monitor = monitor
#         self.print2console = print2console
#         self.batch_vals = {}
#         self.epoch_vals = {}
#         self.__reset_dict("epoch_vals")
#         self.implemented_metrics = {
#             "valid_acc": self.__val_acc,
#             "valid_loss": self.__val_loss,
#             "loss": self.__loss
#         }

#     def on_epoch_begin(self, epoch):
#         self.epoch = epoch
#         self.__reset_dict("batch_vals")

#     def on_epoch_end(self, *args):
#         for mon in self.batch_vals:
#             avg_val = sum(self.batch_vals[mon]) / len(self.batch_vals[mon])
#             self.epoch_vals[mon].append(avg_val)
#         if self.print2console:
#             self.__print_to_console()

#     def on_batch_end(self):
#         for mon in self.monitor:
#             if mon[:5] != "valid":
#                 self.implemented_metrics[mon]()

#     def on_validate_begin(self):
#         empty_string = "validate".rjust(
#             len(f"epoch: {self.epoch} / {self.epochs}"))
#         with torch.no_grad():
#             pbar = tqdm(self.learn.data.valid_dl,
#                         total=len(self.learn.data.valid_dl))
#             for data in pbar:
#                 pbar.set_description(empty_string, refresh=False)
#                 xb, yb = data
#                 out = self.learn.model(xb)
#                 loss = self.learn.loss_func(out, yb)
#                 for mon in self.monitor:
#                     if mon[:5] == "valid":
#                         self.implemented_metrics[mon](
#                             out=out, loss=loss, xb=xb, yb=yb)

#     def __val_acc(self, **kwargs):
#         _, predicted = torch.max(kwargs["out"].data, 1)
#         correct = (predicted == kwargs["yb"]).sum(
#         ).item() / self.learn.data.valid_dl.bs
#         self.batch_vals["valid_acc"].append(correct)

#     def __val_loss(self, **kwargs):
#         self.batch_vals["valid_loss"].append(kwargs["loss"].item())

#     def __loss(self):
#         self.batch_vals["loss"].append(self.loss.item())

#     def __reset_dict(self, dict):
#         if dict == "batch_vals":
#             for mon in self.monitor:
#                 self.batch_vals[mon] = []
#         if dict == "epoch_vals":
#             for mon in self.monitor:
#                 self.epoch_vals[mon] = []

#     def __print_to_console(self):
#         print_string = f""
#         if self.epoch == self.epochs:
#             print_string += "\n"
#         print_string += f"metrics: ".rjust(
#             len(f"epoch: {self.epoch} / {self.epochs}")+2)

#         for mon in self.epoch_vals.items():
#             print_string += f"{mon[0]}: {mon[1][-1]:.6f},  "
#         print(print_string)

#     @property
#     def history(self):
#         history = pd.DataFrame(self.epoch_vals)
#         l = len(history)
#         history.insert(0, column="epoch", value=range(1, l+1))
#         return history

# # class Tracker_Cb(Callback):
# #     def __init__(self, monitor="valid_loss", comp=np.less):
# #         super().__init__()
# #         self.monitor = monitor
# #         self.comp = comp
# #     def get_monitor_value(self):
# #         return self.history # [self.monitor][-1]


# class EarlyStopping_Cb(Recorder_Cb):
#     def __init__(self, monitor="valid_loss", comp=np.less, patience=1):
#         self.monitor = monitor
#         self.patience = patience
#         self.comp = np.less

#     def on_train_begin(self, learn, *args):
#         self.learn = learn
#         self.wait = 0

#     def on_epoch_end(self):
#         minimum = self.learn.recorder[self.monitor][:-1].min()
#         last_val = self.learn.recorder[self.monitor].iloc[-1]
#         if minimum < last_val:
#             self.wait += 1
#         else:
#             self.wait = 0
#         if self.wait >= self.patience:
#             self.learn._stop = True


# class SaveBestModel_Cb(Callback):
#     def __init__(self, path, monitor="valid_loss", date=date.today(), run=0):
#         self.path = path
#         self.monitor = monitor
#         self.date_time = date
#         self.run = run

#     def on_epoch_end(self, *args):
#         minimum = self.learn.recorder[self.monitor][:-1].min()
#         last_val = self.learn.recorder[self.monitor].iloc[-1]
#         if last_val < minimum:
#             print(self.learn.cbh.Monitor_Cb)

# # run_variable which increases with each fit
# # make folder_structure -> date -> check if there is a run folder -> safe best model to run_folder
# # tracker_class that calcs the best value

def get_callbacks(setup_config):
    implemented_cbs = {"m": Monitor(),
                       "e": EarlyStopping()}

    cb_list = [c for c in setup_config if c[:2] == "c_"]
    cb_args = {}
    for i in cb_list:
        cb = i.split("_")[1:]
        if cb[0] not in cb_args:
            cb_args[cb[0]] = {cb[1]: setup_config[i]}
        else:
            cb_args[cb[0]][cb[1]] = setup_config[i]

    cbs = []
    for _cb, cb_list in cb_args.items():
        # importent, that classes get instantiated here
        cb = implemented_cbs[_cb]
        for attr, val in cb_list.items():
            setattr(cb, attr, val)
        cbs.append(cb)
    return cbs


def get_callbackhandler(setup_config):
    if any([c for c in setup_config.keys() if c[:2] == "c_"]):
        cbs = [Recorder(), CudaCallback()]
        cbs.extend(get_callbacks(setup_config))
    else:
        cbs = [Recorder(), CudaCallback()]
    return CallbackHandler(cbs)
