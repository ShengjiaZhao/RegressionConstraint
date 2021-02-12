import pandas as pd
import numpy as np
import itertools
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
from dataset import *
from models import *
from utils import *
from data_loaders import get_uci_datasets

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_root', type=str, default='int_log')
parser.add_argument('--dataset', type=str, default='crime')   # Available options are crime, naval, 

parser.add_argument('--train_bias_y', action='store_true')
parser.add_argument('--train_bias_f', action='store_true')
parser.add_argument('--train_cons', action='store_true')
parser.add_argument('--train_calib', action='store_true')
parser.add_argument('--re_calib', action='store_true')
parser.add_argument('--re_bias_f', action='store_true')
parser.add_argument('--re_bias_y', action='store_true')

# Modeling parameters
parser.add_argument('--model', type=str, default='bigg')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_bins', type=int, default=0)
parser.add_argument('--knn', type=int, default=100)

# Run related parameters
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--run_label', type=int, default=0)
parser.add_argument('--num_run', type=int, default=10)
args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu)
args.device = device

start_time = time.time()

if args.num_bins == 0:
    eval_bias = eval_bias_knn
    assert args.knn > 10 and args.knn % 2 == 0
    
for runs in range(args.num_run):
    while True:
        args.name = '%s/model=%s-%r-%r-%r-%r-%r-%r-%r-bs=%d-bin=%d-%d-run=%d' % \
            (args.dataset, args.model, 
             args.train_bias_y, args.train_bias_f, args.train_cons, args.train_calib, args.re_calib, args.re_bias_f, args.re_bias_y,
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

    if args.re_calib or args.re_bias_f or args.re_bias_y:
        train_dataset, val_dataset, test_dataset, x_dim, y_dim, _= get_uci_datasets(args.dataset, split_seed=args.run_label, val_fraction=0.2, test_fraction=0.2) 
    else:
        train_dataset, _, test_dataset, x_dim, y_dim, _ = get_uci_datasets(args.dataset, split_seed=args.run_label, val_fraction=0.0, test_fraction=0.2)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    train_bb_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    
    # Define model and optimizer
    model = model_list[args.model](x_dim[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epoch // 20, gamma=0.9) 
    # train_bb_iter = itertools.cycle(train_bb_loader)

    bb = iter(train_bb_loader).next()
    bb_counter = 0   # Only refresh bb every 100 steps to save computation
    
    prev_iteration = 0
    while global_iteration < 200000:
        model.train()
        train_l2_all = []
        for i, data in enumerate(train_loader):
            # Minimize L2
            optimizer.zero_grad()
            loss_l2 = eval_l2(model, data, args)
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
        
        if global_iteration - prev_iteration > 20000:
            prev_iteration = global_iteration
            
            if args.re_calib:
                model.recalibrator = Recalibrator(model, val_dataset[:], args)
            if args.re_bias_y:
                model.recalibrator = RecalibratorBias(model, val_dataset[:], args, axis='label')
            if args.re_bias_f:
                model.recalibrator = RecalibratorBias(model, val_dataset[:], args, axis='prediction')
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

            if global_iteration % 1000 == 0:
                print('global_iteration %d, time %.2f, %s' % (global_iteration, time.time() - start_time, args.name))
            scheduler.step()
       