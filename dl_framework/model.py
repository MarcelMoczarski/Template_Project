import torch
from torch import nn

class Model_1(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, nh),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(nh, n_out)
        )
    def forward(self, x):
        return self.model(x)


class Model_2(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        nh = 512
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

def get_model(data, model_class, lr=0.5, nh=50, optim=torch.optim.SGD):
    m = data.train_ds.x.shape[1]
    model = model_class(m, nh, data.c)
    return model, optim(model.parameters(), lr=lr)