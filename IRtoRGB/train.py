# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import os
import shutil
import time
import copy
from enum import Enum

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.nn.functional as F

import pretrainedmodels
from torchvision import transforms

from models import *

import rgb_nir_loader

class Params:    
    workers = 1
    epochs = 10000
    start_epoch = 0
    batch_size = 8  # might want to make smaller
    lr = 1e-4
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 20
    
    feats_in = 3
    feats_out = 6
    feats_max = 32   
    
    sz = 128
    num_layers = 5

    multi_gpu = False

    data_root_path = 'E:/Research/Images/RGB-NIR/nirscene1'
    models_root_path = './'
    #resume = []
    resume = './model_best.pth_nirscene1_128.tar'

    train_file = data_root_path + '/train.csv'
    val_file = data_root_path + '/val.csv'
    data_root = data_root_path

def main():
    global args
    args = Params()

    best_prec = 1e10

    # data loading code
    train_dataset = rgb_nir_loader.RGBNIR(args.data_root, args.train_file, args.sz,
                     is_train=True)

    val_dataset = rgb_nir_loader.RGBNIR(args.data_root, args.val_file, args.sz,
                     is_train=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    # build model
    if (args.resume):
        model = IRtoRGB(args.resume, (args.sz, args.sz), args.num_layers, args.feats_in, args.feats_out, args.feats_max, True)
    else:
        model = IRtoRGB('', (args.sz, args.sz), args.num_layers, args.feats_in, args.feats_out, args.feats_max, True)

    #model.apply(weight_init)

    ct = 0
    for child in model.children():
        print("Child %d %s" % (ct, child))
        #for param in child.parameters():
        #    param.requires_grad = True
        ct += 1

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    print("training %d params" % len(trainable_params))

    optimizer = torch.optim.Adam(trainable_params, args.lr)

    # load pretrained model
    #if (args.resume and not isinstance(args.resume, list)):    
    #    best_prec, args.start_epoch = model.loadOptimizer(optimizer)

    if (args.multi_gpu):
        model = nn.DataParallel(model)

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        #for subepoch in range(0, 10):
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on remaining 10% validation set
        prec = validate(val_loader, model, criterion, False)

        # remember best prec and save checkpoint

        is_best = prec < best_prec

        print('new %f best %f loss' % (prec, best_prec))

        best_prec = min(prec, best_prec)

        save_model({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer' : optimizer.state_dict(),                                    
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss')
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        loss = criterion(output, target_var)        

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(args.batch_size/ (time.time() - end) + 1e-6)
        end = time.time()

        if i % args.print_freq == 0:

            show_images(input, output, target)

            print('[{0}/{1}]\t'
                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                '{loss.val:.3f} ({loss.avg:.3f})'.format(i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_loader, model, criterion, save_preds=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    im_ids = []

    print('Validate:\tTime\t\tLoss')
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(args.batch_size/(time.time() - end) + 1e-6)
        end = time.time()

        if i % args.print_freq == 0:

            show_images(input, output, target)

            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
