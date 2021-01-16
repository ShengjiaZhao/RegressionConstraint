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

class ToyDataset(Dataset):
    def __init__(self, train=True):
        super(ToyDataset, self).__init__()
        
        if train:
            self.size = 100
        else:
            self.size = 500
            
        self.data_x = torch.rand(self.size)

        self.func = lambda x: torch.sin(1.0 / (x+0.1))
        
        self.data_y = self.func(self.data_x).view(-1, 1)
        self.data_x = self.data_x.view(-1, 1)
        self.x_dim = 1
        
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
        
    def __len__(self):
        return self.size
