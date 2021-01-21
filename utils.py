from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random


class Recalibrator:
    def __init__(self, model, data, args):
        self.args = args
        self.model = model
        
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
        
        labels = torch.sort(labels.flatten())[0].cpu().numpy()
        outputs = torch.sort(outputs.flatten())[0].cpu().numpy()
#         plt.scatter(outputs, labels)
#         plt.show()
        
#         plt.hist(outputs, bins=30, alpha=0.5, color='r')
#         plt.hist(labels, bins=30, alpha=0.5, color='g')
#         plt.show()
        # print(labels.shape, outputs.shape)
        self.iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
        self.iso = self.iso.fit(outputs, labels)

        
    def adjust(self, original_y):
        original_shape = original_y.shape
        return torch.from_numpy(self.iso.predict(original_y.cpu().flatten())).view(original_shape).to(self.args.device)


