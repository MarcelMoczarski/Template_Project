import torchvision
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset(source, dataset, save_path):
    if source == "torchvision":
        # pass
        if dataset == "MNIST":
            mnist_trainset = torchvision.datasets.MNIST(save_path, train=True, download=True)
            mnist_testset = torchvision.datasets.MNIST(save_path, train=False, download=True)
            x_train, y_train = mnist_trainset.data / 255, mnist_trainset.targets
            x_test, y_test = mnist_testset.data / 255, mnist_testset.targets
            x_train = x_train.reshape((len(x_train), -1))
            x_test = x_test.reshape((len(x_test), -1))
            return x_train, y_train, x_test, y_test

class Dataset():
    #transforms?
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class DataLoader():
    #shuffle? num_workers?
    def __init__(self, ds, bs, drop_last=True):
        self.dataset = ds
        self.bs = bs
        self.drop_last = drop_last
        self._drop_length = len(self.dataset) - bs * int(len(self.dataset) // bs)
        
    def __len__(self):
        if self.drop_last:
            length = np.floor(len(self.dataset) / self.bs)
        else:
            length = np.ceil(len(self.dataset) / self.bs)
        return int(length)

    def __iter__(self):
        if self.drop_last:
            length = len(self.dataset) - self._drop_length
        else:
            length = len(self.dataset)
        for i in range(0, length, self.bs):
            yield self.dataset[i:i+self.bs]

class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property #setter getter etc. -> instead of DataBunch.train_ds() you can use DataBunch.train_ds to get Dataset
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset

def get_dls(train_ds, valid_ds, test_ds, bs):
    return DataLoader(train_ds, bs), DataLoader(valid_ds, bs), DataLoader(test_ds, bs)

def split_trainset(x_data, y_data, valid_split, stratify=False):
    if stratify == False:
        train_indices, valid_indices, _, _ = train_test_split(range(len(x_data)), y_data, test_size=valid_split)
    else:
        train_indices, valid_indices, _, _ = train_test_split(range(len(x_data)), y_data, test_size=valid_split, stratify=y_data)
    x_train, y_train = x_data[train_indices], y_data[train_indices]
    x_valid, y_valid =  x_data[valid_indices], y_data[valid_indices]
    return x_train, y_train, x_valid, y_valid

    #use of subsets here, to safe  ram, if needed

