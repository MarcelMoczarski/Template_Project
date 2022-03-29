from tqdm import tqdm
import torch 

class Learner():
    def __init__(self, model, opt, loss_func, data, cbh):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.data = data
        self.cbh = cbh
        self.recorder = self.cbh.Monitor_Cb.history

        #helper vars
        self._stop = False
        self._best_model = True
        self.epoch = 0
        self._start_time = 0

    def fit(self, epochs):
        self.cbh.on_train_begin(self, epochs)
        for epoch in range(epochs):
            self.cbh.on_epoch_begin(epoch)
            self.__all_batches()
            self.cbh.on_validate_begin()
            self.cbh.on_validate_end()
            self.cbh.on_epoch_end()
            if self._stop == True:
                break
        self.cbh.on_train_end()

    def __all_batches(self):
        pbar = tqdm(self.data.train_dl, total=len(self.data.train_dl))
        for data in pbar:
            pbar.set_description(f"epoch: {self.cbh.epoch} / {self.cbh.epochs}", refresh=False)
            xb, yb = data
            self.cbh.on_batch_begin(xb, yb)
            self.__one_batch(xb, yb)
            self.cbh.on_batch_end()
            
    def __one_batch(self, xb, yb):
        out = self.model(xb)
        loss = self.loss_func(out, yb)
        self.cbh.on_loss_end(loss)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

    def save(self, path, run, **kwargs):
        state = {
            "epoch": self.epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.opt.state_dict()
        }
        path = path / "checkpoint_model.pt"
        torch.save(state, path)
