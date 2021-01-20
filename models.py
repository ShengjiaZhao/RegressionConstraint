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
    
    def forward(self, x):
        out = self.fc1(x)
        return out
    
class ModelSmall(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelSmall, self).__init__()
        self.fc1 = nn.Linear(x_dim, 20)
        self.fc2 = nn.Linear(20, out_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

class ModelBig(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelBig, self).__init__()
        self.fc1 = nn.Linear(x_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class ModelBigg(nn.Module):
    def __init__(self, x_dim, out_dim=1):
        super(ModelBigg, self).__init__()
        self.fc1 = nn.Linear(x_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, out_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
model_list = {'linear': ModelLinear, 'small': ModelSmall, 'big': ModelBig, 'bigg': ModelBigg}

# Input a regression model and a pair of data, output the total error and binned error
# If axis=label computes the label conditional bias, if axis=prediction computes the prediction conditional bias
def eval_bias(model, data, args, axis='label'):
    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
        
    # Define the bins
    vals = torch.linspace(0.0, 1.0, args.num_bins+1)
    errs = []
    err_total = 0.0

    outputs = model(inputs)
    for i in range(args.num_bins):
        # Compute the index of elements that fall into each bin
        if axis == 'label':
            bi = (labels.flatten() > vals[i]) & (labels.flatten() < vals[i+1])
        else:
            assert axis == 'prediction'
            bi = (outputs.flatten() > vals[i]) & (outputs.flatten() < vals[i+1])
            
        # Remove any bin that contain less than 10 element
        if (bi.type(torch.int).sum() < 10):
            errs.append(0)
            continue
            
        # Extract the elements in a bin and compute the error
        err = outputs[bi].mean() - labels[bi].mean()
        err_total += err.pow(2) * bi.type(torch.float32).mean()
        errs.append(err)
    return err_total, errs

def eval_bias_knn(model, data, args, axis='label', k=100):
    assert k % 2 == 0

    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device).flatten()
    outputs = model(inputs).flatten()

    if axis == 'label':
        ranking = torch.argsort(labels)
    else:
        assert axis == 'prediction'
        ranking = torch.argsort(outputs)

    sorted_labels = labels[ranking]
    sorted_outputs = outputs[ranking]

    smoothed_outputs = F.conv1d(sorted_outputs.view(1, 1, -1), 
                                weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False))).flatten()
    smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1), 
                               weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False))).flatten()
    loss_bias = smoothed_labels - smoothed_outputs
    return loss_bias.pow(2).mean(), loss_bias

def eval_cons(model, data, args, axis='label', alpha=0.5):
    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)

    # Define the bins
    vals = torch.linspace(0.0, 1.0, args.num_bins+1)
    errs = []
    err_total = 0.0

    outputs = model(inputs)
    for i in range(args.num_bins):
        # Compute the index of elements that fall into each bin
        if axis == 'label':
            bi = (labels.flatten() > vals[i]) & (labels.flatten() < vals[i+1])
        else:
            assert axis == 'prediction'
            bi = (outputs.flatten() > vals[i]) & (outputs.flatten() < vals[i+1])
            
        # Remove any bin that contain less than 5 element
        if (bi.type(torch.int).sum() < 5):
            errs.append(0)
            continue
            
        # Extract the elements in a bin and compute the error
        # This is non-zero if less than alpha proportion of outputs are below labels
        err = (alpha - (outputs[bi] <= labels[bi]).type(torch.float).mean()).detach()
        if args.two_sided:
            err_total += F.relu(err) * outputs[bi].mean() * bi.type(torch.float32).mean() - \
                F.relu(-err) * outputs[bi].mean() * bi.type(torch.float32).mean()
        else:
            err_total += F.relu(err) * outputs[bi].mean() * bi.type(torch.float32).mean()
        # Use hinge loss for training

        errs.append(err)
    return err_total * 200.0, errs

# et, err = eval_cons(model, next(train_bb_iter), args, alpha=0.8)
# plt.bar(np.linspace(0, 1, 20), np.array(err), width=0.05)
# plt.ylim([-0.2, 0.2])
# plt.show()
# et, err = eval_cons(model, test_dataset[:], args, alpha=0.8)
# plt.bar(np.linspace(0, 1, 20), np.array(err), width=0.05)
# plt.ylim([-0.2, 0.2])
# plt.show()


def eval_calibration(model, data, args):
    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
    
    outputs = model(inputs)
    
    labels, _ = torch.sort(labels.flatten())
    outputs, _ = torch.sort(outputs.flatten())
    
    loss_calib = (labels - outputs).pow(2).mean()
    return loss_calib, None

def eval_decisions(model, data, args, thresholds):
    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
    
    outputs = model(inputs)
    
    labels = labels.repeat(1, thresholds.shape[0])
    outputs = outputs.repeat(1, thresholds.shape[0])
    thresholds = thresholds.view([1, -1]).repeat(labels.shape[0], 1)

    fn = ((outputs < thresholds) & (labels >= thresholds)).type(torch.float32).mean(axis=0)
    fp = ((outputs >= thresholds) & (labels < thresholds)).type(torch.float32).mean(axis=0)
    return fn, fp 
    
    
def eval_l2(model, data, args):
    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
        
    outputs = model(inputs)
    loss = (outputs - labels).pow(2)
    return loss

def make_plot(model, data, args, file_name='train', do_plot=True, alpha=0.5):
    err_total_bias, errs_bias = eval_bias(model, data, args)
    err_total_cons, errs_cons = eval_cons(model, data, args, alpha=alpha)
    
    if do_plot:
        errs_bias = np.array(errs_bias)
        fig = plt.figure()
        plt.bar(np.linspace(0, 1, args.num_bins+1)[:-1], errs_bias, align='edge', width=0.05)
        plt.ylim([-0.3, 0.1])
        plt.savefig(os.path.join(args.log_dir, file_name % 'bias'))
        plt.close()

        errs_cons = np.array(errs_cons)
        fig = plt.figure()
        plt.bar(np.linspace(0, 1, args.num_bins+1)[:-1], errs_cons, align='edge', width=0.05)
        plt.ylim([-0.3, 0.1])
        plt.savefig(os.path.join(args.log_dir, file_name % 'cons'))
        plt.close()
    return err_total_bias, err_total_cons
