import torch
from torch import nn




class Model_1(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__() #? doesn't work without ? maybe is calling nn.Module.__init__() first ?
        self.model = nn.Sequential(
            nn.Linear(n_in, nh),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(nh, n_out)
        )
    def forward(self, x):
        return self.model(x)

def get_model(data, model_class, lr=0.5, nh=50, optim=torch.optim.SGD):
    m = data.train_ds.x.shape[1]
    model = model_class(m, nh, data.c)
    return model, optim(model.parameters(), lr=lr)