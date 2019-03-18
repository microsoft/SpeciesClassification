################
#
# trainer.py
#
# Trains a FasterRCNN-based detector for the wildlife classification project.
#
# This is the core of the training that's initiated by train.py .
#
################

from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from torch.autograd import Variable
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
import numpy as np

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """
    Wrapper for conveniently training. returns losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn, n_fg_class=20):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.n_fg_class = n_fg_class
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(n_fg_class+1)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss


    def forward(self, imgs, bboxes, labels, scale):
        """
        Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        #print(bboxes)
        
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        #print(gt_roi_label)
        #print('got region proposals')
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        n_bbox = bbox.shape
        if len(n_bbox) > 0:
            n_bbox = n_bbox[0]
        if n_bbox > 0:
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(bbox),
                anchor,
                img_size)
            #print(gt_rpn_label.shape)
            #print(gt_rpn_label)
            #print(anchor.shape)
            #print(sample_roi.shape)
            #print('got anchor targets')
            gt_rpn_label = at.tovariable(gt_rpn_label).long()
            gt_rpn_loc = at.tovariable(gt_rpn_loc)
            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma)
            #print(rpn_loc_loss)
        else: #if no bboxes, should have no rpn loc loss
            rpn_loc_loss = t.tensor(0.)
            if opt.use_cuda:
                rpn_loc_loss = rpn_loc_loss.cuda()
        #print('got rpn loc loss')
        
        # if no bboxes, all region labels are 0 (background)
  
        if n_bbox == 0:
            gt_rpn_label = t.tensor([0 for i in range(anchor.shape[0])])
        # NOTE: default value of ignore_index is -100 ...
        fg_bg_count = np.unique(gt_rpn_label.detach().cpu(), return_counts=True)[1][1:]
        if opt.reduce_bg_weight:
            # Reweight foreground / background for the case we couldn't sample identical numbers
            rpn_class_weights = 1.0 / fg_bg_count
            rpn_class_weights = t.FloatTensor(rpn_class_weights / np.sum(rpn_class_weights) * 2)
        else:
            rpn_class_weights = None
        if opt.use_cuda:
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1,
                                            weight=rpn_class_weights.cuda() if rpn_class_weights is not None else None)
        else:
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1, weight=rpn_class_weights)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())
        #print('got rpn class loss')

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        #print(n_sample, gt_roi_label.shape, sample_roi.shape)
        if opt.use_cuda:
            roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        else:
            roi_loc = roi_cls_loc[t.arange(0, n_sample).long(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)

        if n_bbox > 0:
            roi_loc_loss = _fast_rcnn_loc_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                self.roi_sigma)
        else: #no roi loc loss if no gt bboxes
            roi_loc_loss = t.tensor(0.)
            if opt.use_cuda:
                roi_loc_loss = roi_loc_loss.cuda()
        #print('got roi loc loss')

        if opt.reduce_bg_weight:
            bg_weight = 1.0 / gt_roi_label.size()[0]
            class_weights = t.FloatTensor(np.hstack([bg_weight, np.ones((self.n_fg_class,))]))
        else:
            class_weights = None

        if opt.use_cuda:
            roi_cls_loss = nn.CrossEntropyLoss(weight=class_weights.cuda() if 
                            class_weights is not None else None)(roi_score, gt_roi_label.cuda())
        else:
            roi_cls_loss = nn.CrossEntropyLoss(weight=class_weights)(roi_score, gt_roi_label)

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())
        #print('got roi class loss')
        
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        #print(losses)
        sum_losses = sum(losses)
        #print(sum_losses.type)
        losses = losses + [sum_losses]

        return LossTuple(*losses)

    # ...def forward(self, imgs, bboxes, labels, scale)


    def train_step(self, imgs, bboxes, labels, scale):
    
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses


    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """
        Serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """

        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        for k_, v_ in kwargs.items():
            save_dict[k_] = v_

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            if 'best_map' in kwargs.keys():
                save_path += '_%s' % kwargs['best_map']

        t.save(save_dict, save_path)
        return save_path


    def load(self, state_dict, load_optimizer=True, parse_opt=False, ):
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return state_dict


    def update_meters(self, losses):

        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key].detach().cpu().numpy())


    def reset_meters(self):

        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()


    def get_meter_data(self):

        return {k: v.value()[0] for k, v in self.meters.items()}

# ...class FasterRCNNTrainer(nn.Module)


def _smooth_l1_loss(x, t, in_weight, sigma):

    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):

    if opt.use_cuda:

        in_weight = t.zeros(gt_loc.shape).cuda()
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation, 
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1

    else:

        in_weight = t.zeros(gt_loc.shape)
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation, 
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).float().sum()  # ignore gt_label==-1 for rpn_loss
    return loc_loss
