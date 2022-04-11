import torch
# from tqdm import tqdm
import numpy as np
# import pandas as pd
# from datetime import date

class Callback():
    def __init__(): pass
    def on_train_begin(self, learn, epochs):
        self.learn = learn
        self.epochs
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
    # using numpy arrays for summming vals is much faster than lists
    def __init__(self): pass
  
    def on_train_begin(self, learn, epochs):
        self.learn = learn
        self.epochs = epochs
        # self.epoch_vals = {"epoch": np.zeros(epochs), "train": np.zeros(epochs), "valid": np.zeros(epochs)}
        self.batch_vals = {"train_loss": np.zeros(epochs), "valid_loss": np.zeros(epochs), "train_pred": np.zeros(epochs), "valid_pred": np.zeros(epochs)}


    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        # self.epoch_vals["epoch"][epoch] = epoch + 1
        self.batch_vals["train_loss"] = np.zeros(self.epochs)
        self.batch_vals["valid_loss"] = np.zeros(self.epochs)
        # self.epoch_vals["epoch"].append(epoch + 1)
        # self.batch_vals["train_loss"] = []
        # self.batch_vals["valid_loss"] = []
        
    def on_batch_begin(self, batch):
        self.batch = batch
    def on_loss_end(self, loss, out, yb):
        self.yb = yb
        self.loss = loss
        self.out = out
        if self.learn.model.training:
            self.batch_vals["train_loss"][self.epoch] = loss.item()
            # self.batch_vals["train_loss"].append(loss.item())
        else:
            self.batch_vals["valid_loss"][self.epoch] = loss.item()
            # self.batch_vals["valid_loss"].append(loss.item())
    #     self.best_value = 

    # def _best_value(self):

class CudaCallback(Callback):
    def __init__(self): pass

    def on_train_begin(self, learn, *args):
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
        self.history = {"epochs": []}


    def on_train_begin(self, learn, epochs):
        self.learn = learn
        self.epochs = epochs
        self.batch_vals = {"train_loss": np.zeros(epochs), "valid_loss": np.zeros(epochs), "train_pred": np.zeros(epochs), "valid_pred": np.zeros(epochs)}
        for mon in self.monitor:
            self.history[mon] = []
        

    def on_batch_end(self):
        _, batch_pred = torch.max(self.out.data, 1)
        batch_correct = (batch_pred == self.yb).sum().item() / len(self.yb)
        if self.learn.model.training:
            self.batch_vals["train_pred"][self.epoch] = batch_correct
        else:
            self.batch_vals["valid_pred"][self.epoch] = batch_correct

    def on_epoch_end(self):
        for mon in self.monitor:
            self.history[mon].append(getattr(self, mon)())
        self.history["epochs"].append(int(self.epoch + 1))
        if self.verbose == True:
            self._print_console()

        setattr(self.learn, "history", self.history)

    def valid_acc(self):
        return sum(self.batch_vals["valid_pred"]) / len(self.batch_vals["valid_pred"])

    def valid_loss(self):
        return sum(self.batch_vals["valid_loss"]) / len(self.batch_vals["valid_loss"])
        
    def train_loss(self):
        return sum(self.batch_vals["train_loss"]) / len(self.batch_vals["train_loss"])
    

    def _print_console(self):
        out_string = f""
        out_string += f"epoch: {int(self.epoch)+1}/{self.epochs}\t"
        for key, val in self.history.items():
            if key != "epochs":
                out_string += f"{key}: {val[-1]:.4f}\t"
        print(out_string)


class EarlyStopping(Recorder):
    def __init__(self):
        super().__init__()
        self.monitor = "train_loss"
        self.patience = 20

        # def on_epoch_end(self, ):

# here function which calcs best val


#     def on_train_begin(self, learn, epochs):
#         self.learn = learn
#         self.epochs = epochs
#         for cb in self.cbs:
#             cb.on_train_begin(learn, self.epochs)

#     def on_train_end(self):
#         for cb in self.cbs:
#             cb.on_train_end()


#     def on_epoch_end(self, *args):
#         for mon in self.batch_vals:
#             avg_val = sum(self.batch_vals[mon]) / len(self.batch_vals[mon])
#             self.epoch_vals[mon].append(avg_val)
#         if self.print2console:
#             self.__print_to_console()


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



#     def __print_to_console(self):
#         print_string = f""
#         if self.epoch == self.epochs:
#             print_string += "\n"
#         print_string += f"metrics: ".rjust(
#             len(f"epoch: {self.epoch} / {self.epochs}")+2)

#         for mon in self.epoch_vals.items():
#             print_string += f"{mon[0]}: {mon[1][-1]:.6f},  "
#         print(print_string)




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
