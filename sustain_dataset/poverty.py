import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader



class Poverty_Dataset(Dataset):
    def __init__(self, train=True, normalize=False, random_permute=True, permutation=None, doc="sustain_dataset/2009-11_uganda/data.pth", partition=0.3):
        super(Poverty_Dataset, self).__init__()

        data_torch = torch.load(doc)
        if random_permute:
            data_torch = data_torch[torch.randperm(data_torch.shape[0])]
        else:
            data_torch = data_torch[permutation]

        data_x = data_torch[:, :-1]
        data_y = data_torch[:, -1]

        # Normalize data_y
        if normalize:
            data_y = (torch.argsort(torch.argsort(data_y)).type(torch.float) / data_y.shape[0]).view(-1, 1)
        else:
            data_y = data_y.view(-1, 1)

        p = int(partition * data_x.shape[0])

        self.mean = torch.mean(data_x[p:], dim=0, keepdim=True)
        self.std = torch.std(data_x[p:], dim=0, keepdim=True)

        self.mean_y = torch.mean(data_y[p:], dim=0, keepdim=True)
        self.std_y = torch.std(data_y[p:], dim=0, keepdim=True)
        self.total_data = data_x.shape[0]

        if train:
            self.data_x = data_x[p:]
            self.data_y = data_y[p:]
        else:
            self.data_x = data_x[:p]
            self.data_y = data_y[:p]

        if not normalize:
            self.data_y = (self.data_y - self.mean_y) / self.std_y

        self.x_dim = data_x.shape[-1]
        self.data_x = (self.data_x - self.mean) / self.std

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]