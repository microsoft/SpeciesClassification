################
#
# train.py
#
# Main file for training classification models.
#
################

import time
import argparse
import numpy as np
import tqdm
import tensorboardX
import warnings
import datetime
import os
import shutil
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.autograd
import torchvision.utils as vutils

from models import *
import criterions
import data_loader

import apex
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel as DDP


def main():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--data_root', required=True,
                        type=str, metavar='DATASET_ROOT', help='Path to the root directory of the dataset.')
    parser.add_argument('--model_type', default=ModelType.resnext101, 
                        metavar='ARCH', type=ModelType.from_string, choices=list(ModelType),
                        help='model architecture: ' + ' | '.join([m.name for m in ModelType]) +
                        ' (default: resnext101)')
    parser.add_argument('--image_size', default=224, nargs='+',
                        type=int, metavar='RESOLUTION', help='The side length of the CNN input image ' + \
                        '(default: 448). For ensembles, provide one resolution for each network.')
    parser.add_argument('--epochs', default=200, 
                        type=int, metavar='N', help='Number of total epochs to run.')
    parser.add_argument('--start_epoch', default=None, 
                        type=int, metavar='N', help='Override starting epoch, useful on restarts.')
    parser.add_argument('--batch_size', default=32, 
                        type=int, metavar='N', help='mini-batch size (default: 32), which is the number of '+ \
                        'images per GPU in a single forward / backward pass.')
    parser.add_argument('--lr', '--learning-rate', default=0.0045, 
                        type=float, metavar='LR', help='initial learning rate (default: 0.0045). The learning rate ' + \
                        'is scaled linearly with the number of GPUs as the batch size also scales this way.')
    parser.add_argument('--lr_decay', default=0.94, 
                        type=float, metavar='LR_DECAY', help='The factor by which the learning rate is reduced ' + \
                        'every --epoch_decay epochs (default: 0.94)')
    parser.add_argument('--epoch_decay', default=4, 
                        type=int, metavar='EPOCH_DECAY', help='The number of epochs after which the learning rate ' + \
                        'is decayed by a factor of --lr_decay (default: 4)')
    parser.add_argument('--momentum', default=0.9, 
                        type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=1e-4, 
                        type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--label_smoothing', default=0.15,
                        type=float, metavar='SMOOTHING', help='Replaces the hard one-hot target in training by a ' + \
                        'a probability distribution, which has 1-SMOOTHING probability on the ground-truth class ' + \
                        'and a taxonomically aware amount of probability across all other classes (default: 0.15)')
    parser.add_argument('--resume', default=None, nargs='+',
                        type=str, metavar='PATH', help='Path to a checkpoint to resume from (default: none). Can ' + \
                        'be multiple checkpoints when training an ensemble from two different checkpoints.')
    parser.add_argument('--warm_up_iterations', default=1600,
                        type=int, metavar='ITERATIONS', help='Performs this number of iterations in the beginning ' + \
                        'of the training with a very low learning rate in order to avoid accuracy drops when ' + \
                        'resuming from a checkpoint (default: 1600)')
    parser.add_argument('--use_onevsall_loss', action='store_true',
                        help='If set, uses a binary cross-entropy loss for each element instead of multi-class ' + \
                        'cross-entropy. Requires label smoothing > 0.')
    parser.add_argument('--bg_classes', default=None, nargs='+',
                        type=int, metavar='BG_CLASS', help='Allows to provide the class IDs for background classes. ' + \
                        'Requires label smoothing > 0 and one-vs-all loss to be set. ')
    parser.add_argument('--train_logits_only', action='store_true',
                        help='If set, only the last linear layer is trained.')
    parser.add_argument('--reset_classifier', action='store_true',
                        help='If set, reinitializes the classifier of the network before training.')
    parser.add_argument('--workers', default=8,
                        type=int, metavar='N', help='number of data loading workers (default: 8). If 0, the ' + \
                        ' data loading and preprocessing is done in the main thread on demand.')
    parser.add_argument('--print_freq', default=1000,
                        type=int, metavar='N', help='Frequency of printing out stats in console (default: 1000)')
    parser.add_argument('--top_prec', default=1,
                        type=int, metavar='TOPK', help='Uses the TOPK accucary to select the best model (default: 1)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluates the model on validation set')
    parser.add_argument('--multi_crop', action='store_true',
                        help='Whether to use multi-crop during evaluation. We use 12 crops in total.')
    parser.add_argument('--val_bounding_boxes', default=None,
                        type=str, metavar='BBOX_FILE', help='Path to bounding box file generated by our RCNN code ' + \
                        '(default: None). These boxes will be used as first crop during validation. ')
    parser.add_argument('--save_preds', action='store_true',
                        help='If --evaluate and --save_preds is set, we will write out the predictions ' + \
                        'for each validation image in the Kaggle format. For Kaggle submission, you want to ' + \
                        'adjust --val_file to the json containing the test data instead of validation data.')
    parser.add_argument('--save_preds_file', default='test_preds.csv',
                        type=str, metavar='FILEPATH', help='If --evaluate and --save_preds is set, then this is the ' + \
                        'file name to store the predictions to (default: test_preds.csv)')
    parser.add_argument('--save_conf_matrix', action='store_true',
                        help='If --evaluate and --save_conf_matrix is set, we will write out the confusion ' + \
                        'matrix computed from the validation predictions.')
    parser.add_argument('--annotation_format', default='2017',
                        type=str, dest='year', metavar='VERSION', help='Version of the dataset format, 2017 or 2018. ' + \
                        '')
    parser.add_argument('--train_file', default='trainval2017.json',
                        type=str, metavar='TRAIN_FILE', help='Name of the json file containing the training annotation ' + \
                        '(default: trainval2017.json). Should be located within the dataset root directory.')
    parser.add_argument('--val_file', default='minival2017.json',
                        type=str, metavar='VAL_FILE', help='Name of the json file containing the validation annotation ' + \
                        '(default: minival2017.json). Should be located within the dataset root directory.')
    # For multi-GPU and half-precision training
    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only runs 10 iterations in each iteration for testing and profiling.')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of GPUs to use. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument("--local_rank", default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Enables the synchronization of the BN computations across GPUs.')

    global args
    args = parser.parse_args()
    assert args.bg_classes is None or (args.label_smoothing > 0 and args.use_onevsall_loss), \
                   'The use of a background class requires label_smoothing > 0 and --use_onevsall_loss'


    # Prepare logging
    log_dir = './log/{}_{}_gpu{}'.format(args.model_type.name, datetime.datetime.now().strftime('%b%d_%H-%M-%S'), args.local_rank)
    # The summary file will contain most of the print outputs for convenience
    global log_summary_file
    log_summary_file = os.path.join(log_dir, 'summary.txt')
    # The object for logging tensorboard events
    global writer
    writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    # Copy all python files to the log directly
    log_py_dir = os.path.join(log_dir, 'code')
    os.makedirs(log_py_dir)
    for fi in  glob.glob('./*.py'):
        shutil.copyfile(fi, os.path.join(log_py_dir, fi))

    # Preparations for using multiple GPU and half precision, see https://github.com/NVIDIA/apex
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    best_prec1 = 0
    best_prec3 = 0
    best_prec5 = 0

    # data loading code
    train_dataset = data_loader.JSONDataset(args.data_root,
                     os.path.join(args.data_root, args.train_file),
                     args.image_size,
                     is_train=True,
                     dataFormat2017 = (args.year == "2017"),
                     percentUse=0.1 if args.prof else 100,
                     label_smoothing = args.label_smoothing,
                     bg_classes = args.bg_classes)

    # We use balanced sampling of all classes, i.e. each class has the same probability to be present in a batch
    if args.distributed:
        train_sampler = data_loader.DistributedBalancedSampler(train_dataset)
    else:
        train_sampler = data_loader.DistributedBalancedSampler(train_dataset, 1, 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    num_classes = train_dataset.get_num_classes()

    # Write out the list of classnames
    classname_list = [train_dataset.classnames[cid] for cid in range(num_classes)]
    with open(os.path.join(log_dir, 'classnames.txt'), 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(classname_list))

    val_dataset = data_loader.JSONDataset(args.data_root,
                     os.path.join(args.data_root, args.val_file),
                     args.image_size,
                     is_train=False,
                     dataFormat2017 = (args.year == "2017"),
                     percentUse=0.1 if args.prof else 100,
                     multi_crop=args.multi_crop,
                     bbox_predictions=args.val_bounding_boxes if args.val_bounding_boxes else None,
                     label_smoothing = args.label_smoothing)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=8 if args.evaluate else args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    print_log("num classes: %d" % (num_classes))

    # build model
    model = ClassificationModel(args.resume, args.image_size, True, args.model_type, train_dataset.classnames)
    if args.reset_classifier:
        model.model.last_linear = nn.Linear(model.model.last_linear.in_features, num_classes).cuda()

    # define loss function (criterion) and optimizer
    if args.use_onevsall_loss:
        assert args.label_smoothing > 0, 'One-vs-all loss requires label smoothing larger than 0.'
        criterion = criterions.BCEWithLogitsLoss2().cuda()
    elif args.label_smoothing > 0:
        criterion = criterions.KLDivLoss2().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    # define learnable parameters and their learning rate multipliers
    if hasattr(model, 'model') and hasattr(model.model, 'last_linear'):
        # First collect the parameters of the base model and the last linear layer
        base_params = list(set(filter(lambda p: p.requires_grad, model.parameters()))
                           - set(filter(lambda p: p.requires_grad, model.model.last_linear.parameters())))
        classifier_params = list(filter(lambda p: p.requires_grad, model.model.last_linear.parameters()))
        # If we train only the last layer, it is usually safe to use a ten times larger learning rate than usual
        if args.train_logits_only:
            trainable_params = [dict(params=classifier_params, lr_mult=10)]
            print_log("Increasing learning rate of classifier by a factor of 10, because only the classifier is trained.")
        # If we reset the classifier, then increase the learning rate of the last linear layer by 10
        # This is the case if we do not resume training, because then we will start from an pre-trained ImageNet model and
        # automatically reset the classifier to have the appropriate number of output elements
        # This is also the case, when we pass the flag args.reset_classifier
        elif not args.resume or args.reset_classifier:
            trainable_params = [dict(params=base_params, lr_mult=1),
                                dict(params=classifier_params, lr_mult=10)]
            print_log("Increasing learning rate of classifier by a factor of 10, because the classifier is re-initialized.")
        # Otherwise, just use the same learning rate everywhere
        else:
            trainable_params = [dict(params=list(filter(lambda p: p.requires_grad, model.parameters())), lr_mult=1)]
    elif args.train_logits_only:
        raise Exception('Could not fine the final linear layer, hence the parameter --train_logits_only can not be used.')
    else:
        trainable_params = [dict(params=list(filter(lambda p: p.requires_grad, model.parameters())), lr_mult=1)]

    print_log("training %d params" % len(trainable_params))

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)

    # load pretrained model
    ckpt_epoch = None
    # Update tensorboard log with old topk values and restore optimizer
    # This makes only sense if we continue from a single checkpoint file
    # as we will always start a new model when there are more files
    if args.resume and len(args.resume) == 1 and not args.reset_classifier:
        best_prec1, best_prec3, best_prec5, ckpt_epoch = model.loadOptimizer(optimizer)
        writer.add_scalars('validation/topk', {'top1':best_prec1,
                                                 'top3':best_prec3,
                                                 'top5':best_prec5},
                                                len(train_loader) * ckpt_epoch)
    else:
        writer.add_scalars('validation/topk', {'top1':0, 'top3':0, 'top5':0},
                                                len(train_loader) * (args.start_epoch if args.start_epoch is not None else 0))
    # Set starting epoch 
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif ckpt_epoch:
        start_epoch = ckpt_epoch
    else:
        start_epoch = 0

    if args.sync_bn:
        print("Enabling the synchronization of BN across GPUs")
        model = apex.parallel.convert_syncbn_model(model)

    if args.fp16:
        model = network_to_half(model)

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    cudnn.benchmark = True

    if args.evaluate:
        # write predictions to file
        if args.save_preds:
            prec1, prec3, prec5, preds, im_ids = validate(val_loader, model, criterion, 0, True)
            with open(args.op_file_name, 'w') as opfile:
                opfile.write('id,predicted\n')
                for ii in range(len(im_ids)):
                    opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii,:]) + '\n')
        else:
            prec1, prec3, prec5 = validate(val_loader, model, criterion, 0, True)
        if args.save_conf_matrix:
            test_labels = np.array(val_dataset.get_labels())
            unique_labels = np.unique(val_dataset.get_labels() + train_dataset.get_labels())
            # As we turned shuffle off, we can just compare the plain labels
            import sklearn.metrics
            cm = sklearn.metrics.confusion_matrix(test_labels, preds[:,0], labels=unique_labels)
            cm = cm / cm.sum(axis=1, keepdims=True)
            np.savetxt('conf.csv', cm, fmt='%.3f', delimiter=',')
        return

    def to_md(code):
        return str(code).replace('\n','\n\t') #'```python\n' + str(code) + '\n```'
    writer.add_text('args/instance',to_md(args.__dict__), start_epoch*len(train_loader))
    print_log('Arguments / configuration: \n' + to_md(args.__dict__))
    writer.add_text('train.py',to_md(open('train.py','rt').read()), start_epoch*len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        writer.add_scalar('epoch', epoch, len(train_loader)*epoch)
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, None)

        # evaluate on remaining 10% validation set
        prec1, prec3, prec5 = validate(val_loader, model, criterion, epoch, len(train_loader)*(epoch + 1), False)

        # remember best prec and save checkpoint

        if (args.top_prec == 1):
            is_best = prec1 > best_prec1
        elif (args.top_prec == 3):
            is_best = prec3 > best_prec3
        else:
            is_best = prec5 > best_prec5

        best_prec1 = max(prec1, best_prec1)
        best_prec3 = max(prec3, best_prec3)
        best_prec5 = max(prec5, best_prec5)

        if args.local_rank == 0:
            save_model({
                'epoch': epoch + 1,
                'args': args,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec3': best_prec3,
                'best_prec5': best_prec5,
                'optimizer' : optimizer.state_dict(),
                'classnames' : train_dataset.classnames,
                'num_classes' : num_classes,
                'model_type' : args.model_type,
            }, is_best)

# ...def main()


def isinf(tensor):

    return tensor == torch.FloatTensor([float('inf')]).type_as(tensor)


def set_grad(params, params_with_grad, scale=1.0):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size()))
        grad = param_w_grad.grad.data
        if scale is not None:
            grad /= scale
        if torch.isnan(grad).any() or isinf(grad).any():
            return True # invalid grad
        param.grad.data.copy_(grad)
    return False


def train(train_loader, model, criterion, optimizer, epoch, param_copy = None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print_log('Epoch:{0}'.format(epoch))
    print_log('Itr\t\tTime\t\tData\t\tLoss\t\tPrec@1\t\tPrec@3\t\tPrec@5')
    for param_group in optimizer.param_groups:
        writer.add_scalar('training/learning_rate', param_group['lr'] / param_group['lr_mult'],
                             len(train_loader)*epoch)
        break

    # Make lr really low for the first couple iterations
    iterations_processed = 0
    if args.warm_up_iterations:
        print_log('Warming up the training for {} iterations with a very small learning rate'.format(args.warm_up_iterations))
        adjust_learning_rate(optimizer, 10000)

    data_iterator = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, im_id, target) in data_iterator:
        iterations_processed = iterations_processed + 1
        if args.warm_up_iterations and iterations_processed > args.warm_up_iterations:
            args.warm_up_iterations = 0
            print_log('Finished warm-up phase')
            adjust_learning_rate(optimizer, epoch)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3, prec5 = accuracy(output.data, target, topk=(1, 3, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(args.batch_size/ (time.time() - end))
        end = time.time()

        writer.add_scalar('training/loss', losses.val, len(train_loader)*epoch + i)
        writer.add_scalars('training/topk', {'top1':top1.val,
                                                   'top3':top3.val,
                                                   'top5':top5.val}, 
                                                 len(train_loader)*epoch + i)
        if i % args.print_freq == 0:
            x = vutils.make_grid(input[0]/2 + 0.5)
            writer.add_image('Preprocessed training images', x, len(train_loader)*epoch + i)
            #for name, param in model.named_parameters():
            #    writer.add_histogram(name, param.clone().cpu().data.numpy(), i)

            print_log('[{0}/{1}]\t'
                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                '{loss.val:.3f} ({loss.avg:.3f})\t'
                '{top1.val:.2f} ({top1.avg:.2f})\t'
                '{top3.val:.2f} ({top3.avg:.2f})\t'
                '{top5.val:.2f} ({top5.avg:.2f})'.format(i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3, top5=top5))

    args.warm_up_iterations = 0
    print_log(' *** Training summary at epoch {epoch:d}: Prec@1 {top1.avg:.3f} '.format(epoch=epoch, top1=top1) +
              'Prec@3 {top3.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(top3=top3, top5=top5, loss=losses))


def validate(val_loader, model, criterion, epoch, global_step, save_preds=False):

    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        pred = []
        im_ids = []

        print_log('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3\t\tPrec@5')
        for i, (inputIn, im_id, target) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            for j in range(0, len(inputIn)):
                input = inputIn[j].cuda()
                input_var = torch.autograd.Variable(input)
                outputNew = model(input_var)
                # In the first crop
                if j == 0:
                    output = outputNew
                # For all other crops
                else:
                    output = output + outputNew
                    #output = torch.max(output, outputNew)
                output /= len(inputIn)

            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target)
            loss = criterion(output, target_var)

            if save_preds:
                # store the top K classes for the prediction
                im_ids.append(im_id.cpu().numpy().astype(np.int))
                _, pred_inds = output.data.topk(3,1,True,True)
                pred.append(pred_inds.cpu().numpy().astype(np.int))

            # measure accuracy and record loss
            prec1, prec3, prec5 = accuracy(output.data, target, topk=(1, 3, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(input.size(0)/(time.time() - end))
            end = time.time()

            if i % args.print_freq == 0:
                print_log('[{0}/{1}]\t'
                      '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                      '{loss.val:.3f} ({loss.avg:.3f})\t'
                      '{top1.val:.2f} ({top1.avg:.2f})\t'
                      '{top3.val:.2f} ({top3.avg:.2f})\t'
                      '{top5.val:.2f} ({top5.avg:.2f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3, top5=top5))

        writer.add_scalar('validation/loss', losses.avg, global_step)
        writer.add_scalars('validation/topk', {'top1':top1.avg,
                                                  'top3':top3.avg,
                                                  'top5':top5.avg}, global_step)

        print_log(' *** Validation summary at epoch {epoch:d}: Prec@1 {top1.avg:.3f} '.format(epoch=epoch, top1=top1) +
        'Prec@3 {top3.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(top3=top3, top5=top5, loss=losses))

        if save_preds:
            return top1.avg, top3.avg, top5.avg, np.vstack(pred), np.hstack(im_ids)
        else:
            return top1.avg, top3.avg, top5.avg


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
        assert isinstance(val, float) or isinstance(val, int)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lr * (args.lr_decay ** (epoch // args.epoch_decay))
    # The batch size is scaled linearly with the number of GPUs
    # We hence scale the learning rate in the same way
    lr = lr * args.world_size
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
    print_log('Setting LR for this epoch to {} = {} GPUs * {}'.format(
                                  lr, args.world_size, lr/args.world_size))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # if we have soft targets
    if len(target.shape) > 1 and target.size(1) > 1:
        gt = target.topk(1, 1)[1][:,0]
    else:
        gt = target
    correct = pred.eq(gt.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def print_log(text):
    if args.local_rank == 0:
        print(text)
        with open(log_summary_file, 'at') as summaryfile:
            summaryfile.write("{}: {}\n".format(datetime.datetime.now().strftime('%a %Y-%m-%d %H:%M:%S'),text))


if __name__ == '__main__':
    main()
