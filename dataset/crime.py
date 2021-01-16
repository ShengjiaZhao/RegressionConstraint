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
import os, sys, shutil, copy, time
from torch.utils.data import Dataset, DataLoader

class CrimeDataset(Dataset):
    def __init__(self, train=True):
        super(CrimeDataset, self).__init__()
        
        attrib = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/attributes.csv'), delim_whitespace = True)
        data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/communities.data'), names = attrib['attributes'])
        
        # Drop useless features
        data = data.drop(columns=['state','county', 'community','communityname', 'fold'], axis=1)
        
        data = data.replace('?', np.nan)
        data = data.dropna(axis=1)
        data_torch = torch.from_numpy(data.to_numpy()).type(torch.float32)
        data_torch = data_torch[torch.randperm(data_torch.shape[0])]
        data_x = data_torch[:, :99]
        data_y = data_torch[:, 99]
        
        # Normalize data_y
        data_y = (torch.argsort(torch.argsort(data_y)).type(torch.float) / data_y.shape[0]).view(-1, 1)
        if train:
            self.data_x = data_x[:1500]
            self.data_y = data_y[:1500]
        else:
            self.data_x = data_x[1500:]
            self.data_y = data_y[1500:]
        
        self.x_dim = 99
        
    def __len__(self):
        return self.data_x.shape[0]
    
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    