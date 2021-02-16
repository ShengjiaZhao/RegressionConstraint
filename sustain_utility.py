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
# from utils import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="gdp")
parser.add_argument('--log_root', type=str, default="./runs/unbiased")

parser.add_argument('--train_bias_y', action='store_true')
parser.add_argument('--train_bias_f', action='store_true')
parser.add_argument('--train_cons', action='store_true')
parser.add_argument('--train_calib', action='store_true')

parser.add_argument('--model', type=str, default="bigg")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_bins', type=int, default=20)

parser.add_argument('--run_label', type=int, default=0)

# parser.add_argument('--doc', type=str, default="2009-11_uganda")

def tax_utility(difference, beta=1.):
    # difference: y_0 - pred
    return beta * difference


def compute_utility(model, test_dataset, a=torch.relu, r=torch.log, y_0=0.3):
    inputs, labels = test_dataset[:]
    inputs, labels = inputs.to(device), labels.to(device)
    pred = model(inputs).reshape(-1)
    labels = labels.reshape(-1)
    finacial_aid = a(y_0 - pred)
    after_finacial_aid = labels + finacial_aid
    # utility = r(2.+after_finacial_aid) # worked well
    utility = r(3.+after_finacial_aid)
    utility = utility.mean(dim=0)
    return utility.data.item(), finacial_aid.sum().data.item()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

# original
def plot_errbar_original(ax, x, y, c=None, label=None):
    mean = np.mean(y, axis=1)
    std = np.std(y, axis=1) / np.sqrt(y.shape[1])
    ax.plot(x, mean, label=label, c=c, linewidth=2.)
    ax.fill_between(x, mean-std, mean+std, color=c, alpha=0.01)

from scipy.interpolate import interp1d
def plot_errbar(ax, x, y, x1=1000, x2=1200, c=None, label=None):
    # import pdb
    # pdb.set_trace()
    for i in range(y.shape[1]):
        f2 = interp1d(x[:, i], y[:, i], kind='linear')
        y[:, i] = f2(np.linspace(x1, x2, len(x)))

    mean = np.mean(y, axis=1)
    std = np.std(y, axis=1) / np.sqrt(y.shape[1])
    x = np.linspace(x1, x2, len(mean))
    ax.plot(x, mean, label=label, c=c, linewidth=3.)
    # ax.fill_between(x, mean-std, mean+std, color=c, alpha=0.01)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda:%d' % args.gpu)
    args.device = device
    start_time = time.time()
    run_labels = range(0, 10, 1)
    utility_points = 100
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    colors = plt.cm.get_cmap('Set3').colors
    # colors = ['r', 'g', 'c', 'm', 'k', 'orange', 'r', 'black']

    for i, train_bias_y in enumerate([True, False]):
        for j, train_bias_f in enumerate([True, False]):
            for k, train_cons in enumerate([True, False]):
                for t, train_calib in enumerate([True, False]):

                    # test_utility = np.zeros((utility_points + 18, len(run_labels)))
                    test_utility = np.zeros((utility_points, len(run_labels)))
                    test_labels = np.zeros((utility_points, len(run_labels)))

                    labels_bias = None
                    for run_label in run_labels:
                        args.name = '%s_knn/model=%s-%r-%r-%r-%r-bs=%d-run=%d' % \
                                    (args.dataset, args.model, train_bias_y, train_bias_f, train_cons,
                                     train_calib, args.batch_size, run_label)

                        args.log_dir = os.path.join(args.log_root, args.name)
                        if not os.path.isdir(args.log_dir):
                            # print("dir not exist {}".format(args.name))
                            continue

                        ckpt = torch.load(os.path.join(args.log_dir, "ckpt.pth"))
                        train_dataset = ckpt[1]
                        test_dataset = ckpt[2]

                        # Define model and optimizer
                        model = model_list[args.model](train_dataset.x_dim).to(device)
                        # model = model_list[args.model](train_dataset.x_dim, act=F.tanh).to(device)
                        model.load_state_dict(ckpt[0])

                        # Performance evaluation
                        with torch.no_grad():
                            u_array_bias = []
                            labels_bias = []

                            for y0 in np.linspace(-1.0, 2, utility_points):
                                u, a = compute_utility(model, test_dataset, y_0=y0)
                                # u, a = compute_utility(model, test_dataset, r=(lambda x: torch.pow(x, 1/2.)), y_0=y0)
                                # u = compute_utility(model, test_dataset, a=tax_utility, y_0=y0)  # .data.item()
                                #     print(u)
                                u_array_bias.append(u)
                                labels_bias.append(a)
#                                 labels_bias.append(y0)

                        # plt.plot(labels_bias, u_array_bias, label="%r-%r-%r-%r"%(train_bias_y, train_bias_f, train_cons,
                        #          train_calib))
                        test_utility[:utility_points, run_label] = np.array(u_array_bias) #smooth(np.array(u_array_bias), 18)
                        test_labels[:utility_points, run_label] = np.array(labels_bias)
                        # ax1.plot(labels_bias, u_array_bias, label='bias_y=%r-bias_f=%r-calib=%r' % (train_bias_y, train_bias_f, train_calib),
                        #         c=colors[int(k * 8 + i * 4 + j * 2 + t) % 12], linewidth=1.)
                    if isinstance(labels_bias, type(None)):
                        continue
                    print(int(k * 8 + i * 4 + j * 2 + t))
                    plot_errbar(ax1, test_labels[:utility_points], test_utility[:utility_points],
                                label='bias_y=%r-bias_f=%r-calib=%r' % (train_bias_y, train_bias_f, train_calib),
                                c=colors[int(k * 8 + i * 4 + j * 2 + t)%12])


    # ax1.set_ylim([0.6, 1.5])
    # ax1.set_ylim([1., 1.5])
    fontsize = 36
    if args.dataset == "gdp":
        ax1.set_title("China GDP per capita prediction", fontsize=fontsize)
    else:
        ax1.set_title("Uganda poverty prediction", fontsize=fontsize)

    ax1.legend(fontsize=fontsize)
    ax1.set_xlabel(r"$y_0$", fontsize=fontsize)
    ax1.set_ylabel(r"$u(\epsilon)$", fontsize=fontsize)

    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('plots/result_{}_knn.png'.format(args.dataset))
    plt.close()

    # a = torch.relu
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    palette = plt.cm.get_cmap('Set3').colors
    for i, train_bias_y in enumerate([True, False]):
        for j, train_bias_f in enumerate([True, False]):
            for k, train_cons in enumerate([True, False]):
                for t, train_calib in enumerate([True, False]):
                    labels_bias = None
                    predicted_array = np.zeros((utility_points, len(run_labels)))
                    true_array = np.zeros((utility_points, len(run_labels)))

                    for run_label in run_labels:
                        args.name = '%s_knn/model=%s-%r-%r-%r-%r-bs=%d-run=%d' % \
                                    (args.dataset, args.model, train_bias_y, train_bias_f, train_cons,
                                     train_calib, args.batch_size, run_label)

                        args.log_dir = os.path.join(args.log_root, args.name)
                        if not os.path.isdir(args.log_dir):
                            print("dir not exist {}".format(args.name))
                            continue

                        ckpt = torch.load(os.path.join(args.log_dir, "ckpt.pth"))
                        train_dataset = ckpt[1]
                        test_dataset = ckpt[2]

                        # Define model and optimizer
                        model = model_list[args.model](train_dataset.x_dim).to(device)
                        # model = model_list[args.model](train_dataset.x_dim, act=F.tanh).to(device)
                        model.load_state_dict(ckpt[0])

                        # Performance evaluation
                        with torch.no_grad():
                            inputs, labels = test_dataset[:]
                            total = inputs.shape[0]
                            value = -0.8
                            mask = (labels < value).squeeze()  # all the data with gdp labels<-0.5

                            # value = 0.8 #-0.5
                            # mask = (labels > value).squeeze()  # all the data with gdp labels<-0.5

                            labels = labels[mask].to(device)
                            inputs = inputs[mask].to(device)
                            # print(inputs.shape, labels.shape)
                            new_total = inputs.shape[0]
                            # palette = plt.cm.get_cmap('Set3').colors

                            predicted_ = []
                            true_ = []
                            labels_bias = []
                            # import pdb
                            # pdb.set_trace()
                            a = torch.relu
                            for y_0 in np.linspace(-0.1, 0.6, utility_points):
                                labels_bias.append(y_0)
                                pred = model(inputs).reshape(-1)
                                finacial_aid = a(y_0 - pred)
                                true_aid = a(y_0 - labels)
                                predicted_.append(finacial_aid.mean().data.item())
                                # print(finacial_aid.mean().data.item())
                                true_.append(true_aid.mean().data.item())

                        predicted_array[:utility_points, run_label] = np.array(predicted_)  # smooth(np.array(u_array_bias), 18)
                        true_array[:utility_points, run_label] = np.array(true_)

                    if isinstance(labels_bias, type(None)):
                        continue
                    print(int(k * 8 + i * 4 + j * 2 + t))
                    plot_errbar_original(ax1, labels_bias, predicted_array[:utility_points],
                                label='bias_y=%r-bias_f=%r-calib=%r' % (
                                train_bias_y, train_bias_f, train_calib),
                                                                c=colors[int(k * 8 + i * 4 + j * 2 + t)%12])
    plot_errbar_original(ax1, labels_bias, true_array[:utility_points],
                label='true', c="green")

    # ax1.set_ylim([0.6, 1.5])
    # ax1.set_ylim([1., 1.5])
    fontsize = 36
    if args.dataset == "gdp":
        ax1.set_title("China GDP per capita prediction", fontsize=fontsize)
    else:
        ax1.set_title("Uganda poverty prediction", fontsize=fontsize)

    ax1.legend(fontsize=fontsize)
    ax1.set_xlabel(r"$y_0$", fontsize=fontsize)
    ax1.set_ylabel("Averaged aid", fontsize=fontsize)

    plt.title("GDP less than {} ({}/{})".format(value, new_total, total), fontsize=fontsize)
    # plt.title("GDP greater than {} ({}/{})".format(value, new_total, total), fontsize=fontsize)
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('plots/aid_result_{}_knn.png'.format(args.dataset))


    # plt.xlabel(r"$y_0$", fontsize=16)
    # plt.ylabel(r"$u(\epsilon)$", fontsize=16)
    # plt.title(r"China GDP per capita prediction", fontsize=16)
    # plt.legend(fontsize=16)
    # plt.savefig(os.path.join("utility.png"))
    # plt.close()


    # model.eval()
    # with torch.no_grad():
    #     inputs, labels = test_dataset[:]
    #     inputs, labels = inputs.to(device), labels
    #     pred = model(inputs).cpu()
    #     from scipy import stats
    #     r, p_value = stats.pearsonr(labels, pred)
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(labels, pred, alpha=0.5, linewidth=0)
    #     plt.xlabel("Ground truth", fontsize=26)
    #     plt.ylabel("Predicted", fontsize=26)
    #     plt.title(r"$R^2$ %.2f" % (r**2))
    #     # plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linewidth=3, color="orange", alpha=0.7)
    #     plt.savefig(os.path.join(args.log_dir, "eval_prediction.png"))
    #     plt.close()