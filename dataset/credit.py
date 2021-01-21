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

class CreditDataset(Dataset):
    def __init__(self, split='train', normalize=True, seed=0):
        super(CreditDataset, self).__init__()
        df = pd.concat([pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/CreditScore_train.csv')), 
                        pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/CreditScore_test.csv'))])

        percent = (df.isnull().sum() / df.isnull().count() * 100)
        m_per = percent[percent > 10]
        df = df.drop(columns=m_per.index, axis=1)

        for i in df.columns:
            df[i].fillna(df[i].mean(), inplace=True)
        if split == 'train':
            df = df.iloc[:-20000]
        elif split == 'train_val':
            df = df.iloc[:-10000]
        elif split == 'val':
            df = df.iloc[-20000:-10000]
        else:
            assert split == 'test'
            df = df.iloc[-10000:]
        self.data_x = df.drop("y", axis=1).to_numpy()
        self.data_y = np.reshape(df["y"].to_numpy(), [-1, 1])
    
        if normalize:
            self.data_x = np.argsort(np.argsort(self.data_x, axis=0), axis=0) / self.data_x.shape[0]
            self.data_y = np.argsort(np.argsort(self.data_y, axis=0), axis=0) / self.data_y.shape[0]
        self.x_dim = self.data_x.shape[1]
        self.data_x = torch.from_numpy(self.data_x).type(torch.float32)
        self.data_y = torch.from_numpy(self.data_y).type(torch.float32)
        
    def __len__(self):
        return self.data_x.shape[0]
    
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
    


