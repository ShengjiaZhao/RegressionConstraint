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

import pdb

class GDP_Dataset(Dataset):
    def __init__(self, train=True, normalize=False, random_permute=True, permutation=None, doc="sustain_dataset/gdp_data/CHN_no2_train.pkl", partition=0.3):
        super(GDP_Dataset, self).__init__()
        pd_data = pd.read_pickle(doc)

        data = pd_data[[
            'avg_lights_x_pct_max',
            'avg_lights_x_pct_mean', 'avg_lights_x_pct_stdDev',
            'avg_lights_x_pct_sum', 'avg_vis_max', 'avg_vis_mean', 'avg_vis_stdDev',
            'avg_vis_sum', 'cf_cvg_max', 'cf_cvg_mean', 'cf_cvg_stdDev',
            'cf_cvg_sum', 'stable_lights_max', 'stable_lights_mean',
            'stable_lights_stdDev', 'stable_lights_sum', 'cvg_diff',
            'light_sum_diff',
            'GDP_cap']]

        data = data.replace('?', np.nan)
        data = data.replace([-np.inf, np.inf], np.nan)
        data = data.dropna(axis=1)

        data_torch = torch.from_numpy(data.to_numpy()).type(torch.float32)
        if random_permute:
            data_torch = data_torch[torch.randperm(data_torch.shape[0])]
        else:
            data_torch = data_torch[permutation]

        data_x = data_torch[:, :-1]
        data_y = data_torch[:, -1]

        # pdb.set_trace()
        # Normalize data_y
        if normalize:
            data_y = (torch.argsort(torch.argsort(data_y)).type(torch.float) / data_y.shape[0]).view(-1, 1)
        else:
            data_y = data_y.view(-1, 1)

        partition = int((1 - partition) * data_x.shape[0])
        self.mean = torch.mean(data_x[:partition], dim=0, keepdim=True)
        self.std = torch.std(data_x[:partition], dim=0, keepdim=True)

        self.mean_y = torch.mean(data_y[:partition], dim=0, keepdim=True)
        self.std_y = torch.std(data_y[:partition], dim=0, keepdim=True)
        self.total_data = data_x.shape[0]

        if train:
            self.data_x = data_x[:partition]
            self.data_y = data_y[:partition]
        else:
            self.data_x = data_x[partition:]
            self.data_y = data_y[partition:]

        # self.data_x = data_x
        # self.data_y = data_y

        if not normalize:
            self.data_y = (self.data_y - self.mean_y) / self.std_y

        self.data_x = (self.data_x - self.mean) / self.std
        self.x_dim = data_x.shape[-1]

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]