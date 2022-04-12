import torch
from torch import nn

class Model_1(nn.Module):
    def __init__(self, n_in, n_out, nh=50):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, nh),
            nn.ReLU(),
            nn.Linear(nh, n_out)
            # nn.Linear(n_in, 128),
            # nn.ReLU(),
            # # nn.Dropout(p=0.2),

            # nn.Linear(128, 64),
            # nn.ReLU(),
            # # nn.Dropout(p=0.2),

            # nn.Linear(64, 32),
            # nn.ReLU(),
            # # nn.Dropout(p=0.2),

            # nn.Linear(32, n_out)
        )
    def forward(self, x):
        return self.model(x)


class Model_2(nn.Module):
    def __init__(self, n_in, n_out, nh=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(32, n_out)
        )
    def forward(self, x):
        return self.model(x)


def get_model(data, arch, lr, c, opt):
    input_shape = data.train_ds.x.shape[1]
    net = globals()[arch](input_shape, c)
    optim = getattr(torch.optim, opt)
    return net, optim(net.parameters(), lr = lr)