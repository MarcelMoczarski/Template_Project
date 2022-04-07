import torch.nn.functional as F


def cross_entropy(x, y):
    cross_entropy = F.cross_entropy