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
from models import NafFlow


class Recalibrator:
    # This class is untested 
    def __init__(self, model, data, args, re_calib=False, re_bias_f=False, re_bias_y=False, verbose=False):
        self.args = args
        self.model = model  # regression model        
        self.flow = NafFlow(feature_size=40).to(args.device) # This flow model is too simple, might need more layers and latents?
        flow_optim = optim.Adam(self.flow.parameters(), lr=1e-3)
        flow_scheduler = torch.optim.lr_scheduler.StepLR(flow_optim, step_size=20, gamma=0.9)
        
        k = args.knn
        assert k % 2 == 0
        assert re_calib or re_bias_f or re_bias_y 
        
        inputs, labels = data
        inputs = inputs.to(self.args.device)
        labels = labels.to(self.args.device).flatten()
        
        for iteration in range(5000):
            flow_optim.zero_grad()
            loss_all = 0.0
            outputs = model(inputs).flatten()
            
            for objective in range(2):
                if objective == 0 and self.re_bias_f:
                    ranking = torch.argsort(outputs)
                elif objective == 1 and self.re_bias_y:
                    ranking = torch.argsort(labels)
                else:
                    continue
                sorted_labels = labels[ranking]
                sorted_outputs = outputs[ranking]

                smoothed_outputs = F.conv1d(sorted_outputs.view(1, 1, -1), 
                                            weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False)),
                                            padding=k // 2).flatten()
                smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1), 
                                           weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False)),
                                           padding=k // 2).flatten()
                adjusted_outputs, _ = self.flow(smoothed_outputs.view(-1, 1))
                loss_bias = (smoothed_labels - adjusted_outputs.view(-1)).pow(2).mean()
                loss_all += loss_bias 

            if re_calib:
                labels = torch.sort(labels.flatten())[0]
                outputs = torch.sort(outputs.flatten())[0]
                adjusted_outputs, _ = self.flow(outputs.view(-1, 1))
                loss_bias = (labels - adjusted_outputs.view(-1)).pow(2).mean()
                loss_all += loss_bias

            loss_all.backward()
            flow_optim.step()
            flow_scheduler.step()
            if verbose and iteration % 100 == 0:
                print("Iteration %d, loss_bias=%.5f" % (iteration, loss_bias))
    
    def adjust(self, original_y):
        original_shape = original_y.shape
        adjusted_output, _ = self.flow(original_y.view(-1, 1))
        return adjusted_output.view(original_shape)

class RecalibratorOnline:
    def __init__(self, model, args, re_calib=False, re_bias_f=False, re_bias_y=False, verbose=False):
        self.args = args
        self.model = model  # regression model        
        self.flow = NafFlow(feature_size=40).to(args.device) # This flow model is too simple, might need more layers and latents?
        self.flow_optim = optim.Adam(self.flow.parameters(), lr=1e-3)
        self.flow_scheduler = torch.optim.lr_scheduler.StepLR(self.flow_optim, step_size=args.num_iter // (args.flow_skip * 20), gamma=0.9)
        self.re_calib = re_calib
        self.re_bias_f = re_bias_f 
        self.re_bias_y = re_bias_y
    
        assert self.args.knn % 2 == 0
        assert re_calib or re_bias_f or re_bias_y 
        
    def train_step(self, data):
        k = self.args.knn
        inputs, labels = data
        inputs = inputs.to(self.args.device)
        labels = labels.to(self.args.device).flatten()
        
        self.flow_optim.zero_grad()
        loss_all = 0.0
        outputs = self.model(inputs).flatten()

        for objective in range(2):
            if objective == 0 and self.re_bias_f:
                ranking = torch.argsort(outputs)
            elif objective == 1 and self.re_bias_y:
                ranking = torch.argsort(labels)
            else:
                continue
            sorted_labels = labels[ranking]
            sorted_outputs = outputs[ranking]

            smoothed_outputs = F.conv1d(sorted_outputs.view(1, 1, -1), 
                                        weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=self.args.device, requires_grad=False)),
                                        padding=k // 2).flatten()
            smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1), 
                                       weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=self.args.device, requires_grad=False)),
                                       padding=k // 2).flatten()
            adjusted_outputs, _ = self.flow(smoothed_outputs.view(-1, 1))
            loss_bias = (smoothed_labels - adjusted_outputs.view(-1)).pow(2).mean()
            loss_all += loss_bias 

        if self.re_calib:
            labels = torch.sort(labels.flatten())[0]
            outputs = torch.sort(outputs.flatten())[0]
            adjusted_outputs, _ = self.flow(outputs.view(-1, 1))
            loss_bias = (labels - adjusted_outputs.view(-1)).pow(2).mean()
            loss_all += loss_bias
        
        loss_all.backward()
        self.flow_optim.step()
        self.flow_scheduler.step()
        return loss_all
            
    def adjust(self, original_y):
        original_shape = original_y.shape
        adjusted_output, _ = self.flow(original_y.view(-1, 1))
        return adjusted_output.view(original_shape)
    

class Recalibrator_flow:
    def __init__(self, model, flow, args):
        self.args = args
        self.model = model  # regression model
        self.flow = flow # replace IsotonicRegression with 1d flow

    def compute_loss(self, data):
        inputs, labels = data[0].to(self.args.device), data[1].to(self.args.device)
        # with torch.no_grad():  # Question: shall we use no_grad()?
        outputs = self.model(inputs)

        labels = torch.sort(labels.flatten())[0]
        outputs = torch.sort(outputs.flatten())[0]
        adjusted_outputs, _ = self.flow(outputs.view(-1, 1))
        loss_bias = (labels - adjusted_outputs.view(-1)).pow(2).mean()
        return loss_bias

    def adjust(self, original_y):
        output = self.flow(original_y.reshape(-1, 1).to(self.args.device))[0]
        return output.reshape(original_y.shape)


class RecalibratorBias_flow:
    def __init__(self, model, flow, args, axis='label'):
        self.axis = axis
        self.args = args
        self.model = model  # regression model
        self.flow = flow

        k = args.knn
        assert k % 2 == 0

    def compute_loss(self, data):
        k = self.args.knn
        inputs, labels = data
        inputs = inputs.to(self.args.device)
        labels = labels.to(self.args.device).flatten()

        outputs = self.model(inputs).flatten() # Question: shall we use no_grad()?
        if self.axis == 'label':
            ranking = torch.argsort(labels)
        else:
            assert self.axis == 'prediction'
            ranking = torch.argsort(outputs)

        sorted_labels = labels[ranking]
        sorted_outputs = outputs[ranking]

        smoothed_outputs = F.conv1d(sorted_outputs.view(1, 1, -1),
                                    weight=(1. / (k + 1) * torch.ones(1, 1, k + 1, device=self.args.device,
                                                                      requires_grad=False)),
                                    padding=k // 2).flatten()
        smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1),
                                   weight=(1. / (k + 1) * torch.ones(1, 1, k + 1, device=self.args.device,
                                                                     requires_grad=False)),
                                   padding=k // 2).flatten()

        if self.axis == 'label':
            adjusted_labels, _ = self.flow(smoothed_labels.view(-1, 1))
            # adjusted_outputs = self.flow.invert(smoothed_outputs.view(-1, 1))
            loss_bias = (adjusted_labels.view(-1) - smoothed_outputs).pow(2).mean()
        elif self.axis == 'prediction':
            adjusted_outputs, _ = self.flow(smoothed_outputs.view(-1, 1))
            loss_bias = (smoothed_labels - adjusted_outputs.view(-1)).pow(2).mean()
        return loss_bias

    def adjust(self, original_y):
        original_shape = original_y.shape
        if self.axis == 'label':
            adjusted_output = self.flow.invert(original_y.view(-1, 1))
        else:
            adjusted_output, _ = self.flow(original_y.view(-1, 1))
        return adjusted_output.view(original_shape)


class RecalibratorBias:
    def __init__(self, model, data, args, axis='label', verbose=False):
        self.axis = axis
        self.flow = NafFlow().to(args.device) # This flow model is too simple, might need more layers and latents?
        flow_optim = optim.Adam(self.flow.parameters(), lr=1e-3)
        
        k = args.knn
        assert k % 2 == 0
        
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device).flatten()
        
        for iteration in range(5000):
            flow_optim.zero_grad()
            outputs = model(inputs).flatten()

            if axis == 'label':
                ranking = torch.argsort(labels)
            else:
                assert axis == 'prediction'
                ranking = torch.argsort(outputs)

            sorted_labels = labels[ranking]
            sorted_outputs = outputs[ranking]

            smoothed_outputs = F.conv1d(sorted_outputs.view(1, 1, -1), 
                                        weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False)),
                                        padding=k // 2).flatten()
            smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1), 
                                       weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False)),
                                       padding=k // 2).flatten()
        #     print(smoothed_outputs.shape)
        #     print(smoothed_labels.shape)
        #     loss_bias = smoothed_labels - smoothed_outputs

            if axis == 'label':
                adjusted_labels, _ = self.flow(smoothed_labels.view(-1, 1))
                adjusted_outputs = self.flow.invert(smoothed_outputs.view(-1, 1))
                loss_bias = (adjusted_labels.view(-1) - smoothed_outputs).pow(2).mean()
            elif axis == 'prediction':
                adjusted_outputs, _ = self.flow(smoothed_outputs.view(-1, 1))
                loss_bias = (smoothed_labels - adjusted_outputs.view(-1)).pow(2).mean()
            loss_bias.backward()
            flow_optim.step()
            
            if verbose and iteration % 100 == 0:
                print("Iteration %d, loss_bias=%.5f" % (iteration, loss_bias))
    
    def adjust(self, original_y):
        original_shape = original_y.shape
        if self.axis == 'label':
            adjusted_output = self.flow.invert(original_y.view(-1, 1))
        else:
            adjusted_output, _ = self.flow(original_y.view(-1, 1))
        return adjusted_output.view(original_shape)
    

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


def eval_bias_knn(model, data, args, axis='label'):
    k = args.knn
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
                                weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False)),
                                padding=k // 2).flatten()
    smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1), 
                               weight=(1./ (k+1) * torch.ones(1, 1, k+1, device=args.device, requires_grad=False)),
                               padding=k // 2).flatten()
    loss_bias = smoothed_labels - smoothed_outputs
    return loss_bias.pow(2).mean(), loss_bias

# def eval_cons(model, data, args, axis='label', alpha=0.5):
#     inputs, labels = data
#     inputs = inputs.to(args.device)
#     labels = labels.to(args.device)

#     # Define the bins
#     vals = torch.linspace(0.0, 1.0, args.num_bins+1)
#     errs = []
#     err_total = 0.0

#     outputs = model(inputs)
#     for i in range(args.num_bins):
#         # Compute the index of elements that fall into each bin
#         if axis == 'label':
#             bi = (labels.flatten() > vals[i]) & (labels.flatten() < vals[i+1])
#         else:
#             assert axis == 'prediction'
#             bi = (outputs.flatten() > vals[i]) & (outputs.flatten() < vals[i+1])
            
#         # Remove any bin that contain less than 5 element
#         if (bi.type(torch.int).sum() < 5):
#             errs.append(0)
#             continue
            
#         # Extract the elements in a bin and compute the error
#         # This is non-zero if less than alpha proportion of outputs are below labels
#         err = (alpha - (outputs[bi] <= labels[bi]).type(torch.float).mean()).detach()
#         if args.two_sided:
#             err_total += F.relu(err) * outputs[bi].mean() * bi.type(torch.float32).mean() - \
#                 F.relu(-err) * outputs[bi].mean() * bi.type(torch.float32).mean()
#         else:
#             err_total += F.relu(err) * outputs[bi].mean() * bi.type(torch.float32).mean()
#         # Use hinge loss for training

#         errs.append(err)
#     return err_total * 200.0, errs

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


def eval_SmoothL1Loss(model, data, args):
    inputs, labels = data
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)

    outputs = model(inputs)
    loss = torch.nn.SmoothL1Loss(reduction="none")(outputs.reshape(-1), labels.reshape(-1))
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



