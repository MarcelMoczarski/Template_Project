from tqdm import tqdm
import torch
from core.dl_framework.callbacks import get_callbacks, Recorder_Cb
from core.dl_framework.data import get_dls, DataBunch, Dataset, DataLoader, split_data


class Learner():
    def __init__(self, data, setup_config):
        self.model = setup_config["g_arch"]
        self.opt = setup_config["g_optimizer"]
        self.loss = setup_config["g_loss_func"]
        self.bs = setup_config["h_batch_size"]
        self.data = self._get_databunch(data, self.bs, setup_config["g_valid_split"])
       
        self._setup_config = setup_config

        if any([c for c in setup_config.keys() if c[:2] == "c_"]):
            self.cbh = get_callbacks(setup_config)
        else:
            self.cbh = []

        self.recorder = Recorder_Cb

    def _get_databunch(self, data, bs, split_size):
        if type(data) != list:
            data = [data]
        if len(data) < 2:
            data = split_data(data, split_size)
        else:
            if any(dl for dl in data if type(dl) == DataLoader):
                data = [data[0].dataset, data[1].dataset]
        data = DataBunch(*get_dls(data[0], data[1], self.bs))
        return data

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
