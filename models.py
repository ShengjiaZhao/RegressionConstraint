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



class ModelLinear(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelLinear, self).__init__()
        self.fc1 = nn.Linear(x_dim, out_dim, bias=True)
        self.recalibrator = None 
        
    def forward(self, x):
        out = self.fc1(x)
        if not self.training and self.recalibrator is not None:
            return self.recalibrator.adjust(out)
        return out
    
class ModelSmall(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelSmall, self).__init__()
        self.fc1 = nn.Linear(x_dim, 20)
        self.fc2 = nn.Linear(20, out_dim)
        self.recalibrator = None 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        if not self.training and self.recalibrator is not None:
            return self.recalibrator.adjust(out)
        return out

class ModelBig(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelBig, self).__init__()
        self.fc1 = nn.Linear(x_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_dim)
        self.recalibrator = None 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        if not self.training and self.recalibrator is not None:
            return self.recalibrator.adjust(out)
        return out

class ModelBigg(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelBigg, self).__init__()
        self.fc1 = nn.Linear(x_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, out_dim)
        self.recalibrator = None
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = self.fc3(x)
        
        if not self.training and self.recalibrator is not None:
            return self.recalibrator.adjust(out)
        return out
    
model_list = {'linear': ModelLinear, 'small': ModelSmall, 'big': ModelBig, 'bigg': ModelBigg}
