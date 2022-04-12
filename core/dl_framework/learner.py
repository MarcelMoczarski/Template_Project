from tqdm import tqdm
import pandas as pd
import torch
from core.dl_framework.callbacks import get_callbackhandler
from core.dl_framework.data import get_dls, DataBunch, Dataset, DataLoader, split_data, get_databunch
from core.dl_framework.model import get_model
from core.dl_framework import loss_functions

class Container():
    def __init__(self, data, setup_config):
        self.opt = setup_config["g_optimizer"]
        self.loss_func = getattr(loss_functions, setup_config["g_loss_func"])
        self.bs = setup_config["h_batch_size"]
        self.arch = setup_config["g_arch"]
        self.c = setup_config["g_num_classes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = setup_config["h_lr"]
        self.gpu = setup_config["m_gpu"]
        self.data = get_databunch(data, self.bs, setup_config["g_valid_split"], self.c)
        self.model, self.opt = get_model(self.data, self.arch, self.lr, self.c, self.opt)

        self.do_stop = False
        # self._setup_config = setup_config
        

class Learner():
    def __init__(self, data, setup_config):
        self.learn = Container(data, setup_config)
        self.cbh = get_callbackhandler(setup_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, epochs):
        self.cbh.on_train_begin(self.learn, epochs)
        for epoch in range(epochs):
            self.cbh.on_epoch_begin(epoch)
            if self.learn.do_stop: break 
            self.all_batches(self.learn.data.train_dl)

            self.cbh.on_validate_begin()
            with torch.no_grad():
                self.all_batches(self.learn.data.valid_dl)

            self.cbh.on_epoch_end()

    def all_batches(self, data): 
        pbar = tqdm(data, total=len(data))
        for batch in pbar:
            self.one_batch(batch)
            self.cbh.on_batch_end()
        # pbar.set_description(f"self.learn.history")

    def one_batch(self, batch):
        self.cbh.on_batch_begin(batch)
        xb, yb = self.cbh.CudaCallback.batch
        out = self.learn.model(xb)
        loss = self.learn.loss_func(out, yb)
        if not self.cbh.on_loss_end(loss, out, yb): return
        loss.backward()
        self.learn.opt.step()
        self.learn.opt.zero_grad()
    
    @property
    def history(self):
        return pd.DataFrame(self.learn.history_raw).set_index("epochs")
        
    #     def __one_batch(self, xb, yb):
#         out = self.model(xb)
#         loss = self.loss_func(out, yb)
#         self.cbh.on_loss_end(loss)
#         loss.backward()
#         self.opt.step()
#         self.opt.zero_grad()
 

# class Learner():
#           self.recorder = self.cbh.Monitor_Cb.history
#         #helper vars
#         self._stop = False
#         self._best_model = True
#         self.epoch = 0
#         self._start_time = 0

#     def fit(self, epochs):
#         self.cbh.on_train_begin(self, epochs)
#         for epoch in range(epochs):
#             self.cbh.on_epoch_begin(epoch)
#             self.__all_batches()
#             self.cbh.on_validate_begin()
#             self.cbh.on_validate_end()
#             self.cbh.on_epoch_end()
#             if self._stop == True:
#                 break
#         self.cbh.on_train_end()

#     def __all_batches(self):
#         pbar = tqdm(self.data.train_dl, total=len(self.data.train_dl))
#         for data in pbar:
#             pbar.set_description(f"epoch: {self.cbh.epoch} / {self.cbh.epochs}", refresh=False)
#             xb, yb = data
#             self.cbh.on_batch_begin(xb, yb)
#             self.__one_batch(xb, yb)
#             self.cbh.on_batch_end()

#     def __one_batch(self, xb, yb):
#         out = self.model(xb)
#         loss = self.loss_func(out, yb)
#         self.cbh.on_loss_end(loss)
#         loss.backward()
#         self.opt.step()
#         self.opt.zero_grad()

#     def save(self, path, run, **kwargs):
#         state = {
#             "epoch": self.epoch + 1,
#             "state_dict": self.model.state_dict(),
#             "optimizer": self.opt.state_dict()
#         }
#         path = path / "checkpoint_model.pt"
#         torch.save(state, path)


# class Learner():
#     def __init__(self, model, opt, loss_func, data, cbh):
#         self.model = model
#         self.opt = opt
#         self.loss_func = loss_func
#         self.data = data
#         self.cbh = cbh
#         self.recorder = self.cbh.Monitor_Cb.history

#         #helper vars
#         self._stop = False
#         self._best_model = True
#         self.epoch = 0
#         self._start_time = 0

#     def fit(self, epochs):
#         self.cbh.on_train_begin(self, epochs)
#         for epoch in range(epochs):
#             self.cbh.on_epoch_begin(epoch)
#             self.__all_batches()
#             self.cbh.on_validate_begin()
#             self.cbh.on_validate_end()
#             self.cbh.on_epoch_end()
#             if self._stop == True:
#                 break
#         self.cbh.on_train_end()

#     def __all_batches(self):
#         pbar = tqdm(self.data.train_dl, total=len(self.data.train_dl))
#         for data in pbar:
#             pbar.set_description(f"epoch: {self.cbh.epoch} / {self.cbh.epochs}", refresh=False)
#             xb, yb = data
#             self.cbh.on_batch_begin(xb, yb)
#             self.__one_batch(xb, yb)
#             self.cbh.on_batch_end()

#     def __one_batch(self, xb, yb):
#         out = self.model(xb)
#         loss = self.loss_func(out, yb)
#         self.cbh.on_loss_end(loss)
#         loss.backward()
#         self.opt.step()
#         self.opt.zero_grad()

#     def save(self, path, run, **kwargs):
#         state = {
#             "epoch": self.epoch + 1,
#             "state_dict": self.model.state_dict(),
#             "optimizer": self.opt.state_dict()
#         }
#         path = path / "checkpoint_model.pt"
#         torch.save(state, path)