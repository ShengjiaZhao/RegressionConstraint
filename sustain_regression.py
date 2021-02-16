import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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

from sustain_dataset import *
from models import *
from utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="gdp")
parser.add_argument('--log_root', type=str, default="./runs/unbiased")

parser.add_argument('--train_bias_y', action='store_true')
parser.add_argument('--train_bias_f', action='store_true')
parser.add_argument('--train_cons', action='store_true')
parser.add_argument('--train_calib', action='store_true')
parser.add_argument('--re_calib', action='store_true')
parser.add_argument('--re_bias_f', action='store_true')
parser.add_argument('--re_bias_y', action='store_true')
parser.add_argument('--require_val', action='store_true')

# Modeling parameters
parser.add_argument('--model', type=str, default="bigg")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_bins', type=int, default=0)
parser.add_argument('--knn', type=int, default=100)

# Run related parameters
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--num_run', type=int, default=10)
parser.add_argument('--run_label', type=int, default=0)

# parser.add_argument('--doc', type=str, default="2009-11_uganda")

args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu)
args.device = device
start_time = time.time()

knn = ''
if args.num_bins == 0:
    eval_bias = eval_bias_knn
    assert args.knn > 10 and args.knn % 2 == 0
    knn = '_knn'


class RecalibratorBias:
    def __init__(self, model, data, args, axis='label', verbose=False):
        self.axis = axis
        #         self.flow = deeper_flow(layer_num=5, feature_size=20).to(args.device)
        self.flow = NafFlow().to(args.device)  # This flow model is too simple, might need more layers and latents?
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
                                        weight=(1. / (k + 1) * torch.ones(1, 1, k + 1, device=args.device,
                                                                          requires_grad=False)),
                                        padding=0).flatten()
            smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1),
                                       weight=(1. / (k + 1) * torch.ones(1, 1, k + 1, device=args.device,
                                                                         requires_grad=False)),
                                       padding=0).flatten()

            # Generate some pseudo datapoints
            #             max_val = smoothed_outputs.max()
            #             pseudo_outputs_max = max_val + torch.linspace(0, 1, len(inputs) // 10, device=args.device)
            #             pseudo_labels_max = max_val + torch.linspace(0, 1, len(inputs) // 10, device=args.device)  * \
            #                 (smoothed_labels[-10:].mean() - smoothed_labels[-20:-10].mean()) / (smoothed_outputs[-10:].mean() - smoothed_outputs[-20:-10].mean())

            #             smoothed_outputs = torch.cat([smoothed_outputs, pseudo_outputs_max])
            #             smoothed_labels = torch.cat([])

            #             pseudo_outputs = torch.linspace(max_val, )
            #             print(smoothed_outputs.shape)
            #             print(smoothed_labels.shape)
            #             loss_bias = smoothed_labels - smoothed_outputs

            #             if axis == 'label':
            #                 adjusted_labels, _ = self.flow(smoothed_labels.view(-1, 1))
            # #                 adjusted_outputs = self.flow.invert(smoothed_outputs.view(-1, 1))
            #                 loss_bias = (adjusted_labels.view(-1) - smoothed_outputs).pow(2).mean()
            #             elif axis == 'prediction':
            adjusted_outputs, _ = self.flow(smoothed_outputs.view(-1, 1))
            loss_bias = (smoothed_labels - adjusted_outputs.view(-1)).pow(2).mean()
            loss_bias.backward()
            flow_optim.step()

            if verbose and iteration % 100 == 0:
                print("Iteration %d, loss_bias=%.5f" % (iteration, loss_bias))

    def adjust(self, original_y):
        original_shape = original_y.shape
        #        if self.axis == 'label':
        #             adjusted_output = self.flow.invert(original_y.view(-1, 1))
        #         else:
        adjusted_output, _ = self.flow(original_y.view(-1, 1))
        return adjusted_output.view(original_shape)


class Recalibrator:
    # This class is untested
    def __init__(self, model, data, args, re_calib=False, re_bias_f=False, re_bias_y=False, verbose=False):
        self.args = args
        self.re_calib = re_calib
        self.re_bias_f = re_bias_f
        self.re_bias_y = re_bias_y
        self.model = model  # regression model
        self.flow = NafFlow(feature_size=40).to(
            args.device)  # This flow model is too simple, might need more layers and latents?
        flow_optim = optim.Adam(self.flow.parameters(), lr=1e-3)
        # flow_scheduler = torch.optim.lr_scheduler.StepLR(flow_optim, step_size=100, gamma=0.9)

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
                                            weight=(1. / (k + 1) * torch.ones(1, 1, k + 1, device=args.device,
                                                                              requires_grad=False)),
                                            padding=k // 2).flatten()
                smoothed_labels = F.conv1d(sorted_labels.view(1, 1, -1),
                                           weight=(1. / (k + 1) * torch.ones(1, 1, k + 1, device=args.device,
                                                                             requires_grad=False)),
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
            # flow_scheduler.step()
            if verbose and iteration % 100 == 0:
                print("Iteration %d, loss_bias=%.5f" % (iteration, loss_bias))

    def adjust(self, original_y):
        original_shape = original_y.shape
        adjusted_output, _ = self.flow(original_y.view(-1, 1))
        return adjusted_output.view(original_shape)

for runs in range(args.num_run):
    while True:
        # args.name = '%s%s/new_model=%s-%r-%r-%r-%r-bs=%d-run=%d' % \
        #             (args.dataset, knn, args.model, args.train_bias_y, args.train_bias_f, args.train_cons,
        #              args.train_calib, args.batch_size, args.run_label)
        args.name = '%s%s/recalibration_model=%s-%r-%r-%r-%r-%r-%r-%r-bs=%d-bin=%d-%d-run=%d' % \
                    (args.dataset, knn, args.model,
                     args.train_bias_y, args.train_bias_f, args.train_cons, args.train_calib, args.re_calib,
                     args.re_bias_f, args.re_bias_y,
                     args.batch_size, args.num_bins, args.knn, args.run_label)
        args.log_dir = os.path.join(args.log_root, args.name)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
            break
        args.run_label += 1
    print("Run number = %d" % args.run_label)
    writer = SummaryWriter(args.log_dir)
    log_writer = open(os.path.join(args.log_dir, 'results.txt'), 'w')

    global_iteration = 0
    random.seed(args.run_label)  # Set a different random seed for different run labels
    torch.manual_seed(args.run_label)

    def log_scalar(name, value, epoch):
        writer.add_scalar(name, value, epoch)
        log_writer.write('%f ' % value)

    # Define dataset and dataset loader
    Dataset = dataset_list[args.dataset]
    total_data = Dataset().total_data
    permutation = torch.randperm(total_data)

    if args.re_calib or args.re_bias_y or args.re_bias_f or args.require_val:
        train_dataset = Dataset(train=True, val=False, random_permute=False, val_partition=0.2, permutation=permutation)
        val_dataset = Dataset(train=False, val=True, random_permute=False, val_partition=0.2, permutation=permutation)
        test_dataset = Dataset(train=False, val=False, random_permute=False, val_partition=0.2, permutation=permutation)
    else:
        train_dataset = Dataset(train=True, random_permute=False, permutation=permutation)
        test_dataset = Dataset(train=False, random_permute=False, permutation=permutation) # could have overlapping

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    train_bb_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Define model and optimizer
    model = model_list[args.model](train_dataset.x_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    # flow_bias_y = deeper_flow(layer_num=5, feature_size=20).to(device)
    # flow_bias_f = deeper_flow(layer_num=5, feature_size=20).to(device)
    # flow_calib = deeper_flow(layer_num=5, feature_size=20).to(device)
    # flow = deeper_flow(layer_num=1, feature_size=20).to(device)  # one joint flow
    # flow_optimizer = optim.Adam(itertools.chain(flow_bias_y.parameters(), flow_bias_f.parameters(), flow_calib.parameters()),
    #                             lr=args.learning_rate) # shall we train flows and regression model jointly and share the optimizers?
    # flow_optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
    # flow_scheduler = torch.optim.lr_scheduler.StepLR(flow_optimizer, step_size=args.num_epoch // 20, gamma=0.9)

    # train_bb_iter = itertools.cycle(train_bb_loader)

    bb = iter(train_bb_loader).next()
    bb_counter = 0  # Only refresh bb every 100 steps to save computation

    for epoch in range(args.num_epoch):
        train_l2_all = []
        model.train()
        for i, data in enumerate(train_loader):
            # Minimize L2
            optimizer.zero_grad()
            loss_l2 = eval_l2(model, data, args)
            # loss_l2 = eval_SmoothL1Loss(model, data, args)
            train_l2_all.append(loss_l2.detach())
            loss_l2.mean().backward()
            optimizer.step()

            # Minimize any of the special objectives
            optimizer.zero_grad()

            if args.train_bias_y:
                loss_bias, _ = eval_bias(model, bb, args, axis='label')
                writer.add_scalar('bias_loss_y', loss_bias, global_iteration)
                loss_bias.backward()

            if args.train_bias_f:
                loss_bias, _ = eval_bias(model, bb, args, axis='prediction')
                writer.add_scalar('bias_loss_f', loss_bias, global_iteration)
                loss_bias.backward()

            if args.train_cons:
                loss_cons, _ = eval_cons(model, bb, args, alpha=alpha)
                writer.add_scalar('cons_loss', loss_cons, global_iteration)
                loss_cons.backward()

            if args.train_calib:
                loss_calib, _ = eval_calibration(model, bb, args)
                writer.add_scalar('calib_loss', loss_calib, global_iteration)
                loss_calib.backward()
            optimizer.step()

            global_iteration += 1

            bb_counter += 1
            if bb_counter > 100:
                bb = iter(train_bb_loader).next()
                bb_counter = 0

        # Performance evaluation
        model.eval()
        with torch.no_grad():
            # Log the train and test l2
            train_l2_all = torch.cat(train_l2_all).mean()
            log_scalar('train_l2', train_l2_all.item(), global_iteration)

            test_l2_all = eval_l2(model, test_dataset[:], args).mean()
            log_scalar('test_l2', test_l2_all.item(), global_iteration)

            #             train_bias_err, train_cons_err = make_plot(model, train_dataset[:], args, ('train-%d' % epoch) + '-%s.png',
            #                                                        do_plot=(epoch % 100 == 0), alpha=alpha)
            #             test_bias_err, test_cons_err = make_plot(model, test_dataset[:], args, ('test-%d' % epoch) + '-%s.png',
            #                                                     do_plot=(epoch % 100 == 0), alpha=alpha)
            # train_calib_err, _ = eval_calibration(model, test_dataset[:], args)
            test_bias_y, _ = eval_bias(model, test_dataset[:], args, axis='label')
            test_bias_f, _ = eval_bias(model, test_dataset[:], args, axis='prediction')
            test_calib_err, _ = eval_calibration(model, test_dataset[:], args)

            #             log_scalar('train_bias_loss', train_bias_err, global_iteration)
            #             log_scalar('train_cons_loss', train_cons_err, global_iteration)
            log_scalar('test_bias_y', test_bias_y, global_iteration)
            log_scalar('test_bias_f', test_bias_f, global_iteration)
            log_scalar('test_calib_loss', test_calib_err, global_iteration)

            thresholds = torch.linspace(0.1, 0.9, 8).to(device)
            fn, fp = eval_decisions(model, test_dataset[:], args, thresholds)
            for ti in range(8):
                log_scalar('fn_%d' % ti, fn[ti], global_iteration)
                log_scalar('fp_%d' % ti, fp[ti], global_iteration)
            log_scalar('fp+fn_all', fn.mean() + fp.mean(), global_iteration)

        log_writer.write('\n')
        log_writer.flush()

        print('global_iteration %d, time %.2f, %s' % (global_iteration, time.time() - start_time, args.name))
        scheduler.step()

        if epoch % 100 == 0:
            print('epoch %d, global_iteration %d, time %.2f, %s' % (
            epoch, global_iteration, time.time() - start_time, args.name))

        states = [
            model.state_dict(),
            train_dataset,
            test_dataset,
            val_dataset,
            epoch,
        ]
        torch.save(states, os.path.join(args.log_dir, 'ckpt.pth'))

        model.eval()
        with torch.no_grad():
            inputs, labels = test_dataset[:]
            inputs, labels = inputs.to(device), labels
            pred = model(inputs).cpu()
            from scipy import stats
            r, p_value = stats.pearsonr(labels, pred)
            plt.figure(figsize=(6, 6))
            plt.scatter(labels, pred, alpha=0.5, linewidth=0)
            plt.xlabel("Ground truth", fontsize=26)
            plt.ylabel("Predicted", fontsize=26)
            plt.title(r"$R^2$ %.2f" % (r**2))
            # plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linewidth=3, color="orange", alpha=0.7)
            plt.savefig(os.path.join(args.log_dir, "prediction.png"))
            plt.close()

    args.knn = 200
    recalibrator_bias_f = RecalibratorBias(model, val_dataset[:], args, axis='prediction', verbose=True)
    recalibrator_bias_y = RecalibratorBias(model, val_dataset[:], args, axis='label', verbose=True)
    recalibrator_calib = Recalibrator(model, val_dataset[:], args, re_calib=True, verbose=True)

    states = [
        model.state_dict(),
        train_dataset,
        test_dataset,
        val_dataset,
        epoch,
        recalibrator_bias_f.flow.state_dict(),
        recalibrator_bias_y.flow.state_dict(),
        recalibrator_calib.flow.state_dict(),
    ]
    torch.save(states, os.path.join(args.log_dir, 'ckpt.pth'))
