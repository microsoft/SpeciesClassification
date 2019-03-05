################
#
# dataset.py
#
# Base classes and utilities for loading training/test data.  The only data set relevant to the
# species classification model is defined in inat_dataset; others are legacy code from the framework
# or other data sets we load for other projects.
#
################

import torch as t
from .inat_dataset import INatBboxDataset
from .voc_dataset import VOCBboxDataset
from .vott_dataset import VottBboxDataset
from .iwildcam_dataset import IWildCamBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from . import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):

    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):

    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):

    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):

    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    #scale1 = min_size / min(H, W)
    scale = max_size / max(H, W)
    scale = min(1, scale) # Downsample, but don't upsample
    #scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        if len(bbox[0]) > 0: #check a bounding box exists for this image
        #if len(bbox) > 0: #check a bounding box exists for this image
            bbox = np.asarray(bbox)
            bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        
        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        if len(bbox[0]) > 0: #check bbox exists for this image
            
            bbox = util.flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])
        
        return img, bbox, label, scale


class Dataset:

    def __init__(self, opt, class_names = []):
        self.opt = opt
        if opt.dataset == 'voc':
            self.db = VOCBboxDataset(opt.voc_data_dir)
        elif opt.dataset in ['inat', 'inat-oneclass']:
            self.db = INatBboxDataset(opt.inat_image_root, opt.inat_train_annotation)
        elif opt.dataset == 'vott':
            self.db = VottBboxDataset(opt.train_image_dir, class_names)
        elif opt.dataset == 'seals':
            self.db = SealsBboxDataset(opt.train_image_dir)
        elif opt.dataset == 'iwildcam':
          self.db = IWildCamBboxDataset(opt.iwildcam_image_root, opt.iwildcam_train_annotation)
        self.tsf = Transform(opt.min_size, opt.max_size)


    def __getitem__(self, idx):

        ori_img, bbox, label, difficult, _ = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        #numpy arrays need to be lists to deal with empty images
        #print(bbox)
        #return img.copy(), bbox.copy().tolist(), label.copy(), scale
        return img.copy(), bbox.copy(), label.copy(), scale


    def __len__(self):

        return len(self.db)
        
    
    def get_class_count(self):

        return self.db.get_class_count()
        
    
    def get_class_names(self):

        return self.db.get_class_names()


class TestDataset:

    def __init__(self, opt, class_names = [], use_difficult=True):

        self.opt = opt
        if opt.dataset == 'voc':
          self.db = VOCBboxDataset(opt.voc_data_dir, split='test', use_difficult=use_difficult)
        elif opt.dataset in ['inat', 'inat-oneclass']:
          self.db = INatBboxDataset(opt.inat_image_root, opt.inat_val_annotation)
        elif opt.dataset == 'vott':
            self.db = VottBboxDataset(opt.val_image_dir, class_names)
        elif opt.dataset == 'seals':
            self.db = SealsBboxDataset(opt.val_image_dir)
        elif opt.dataset == 'iwildcam':
          self.db = IWildCamBboxDataset(opt.iwildcam_image_root,opt.iwildcam_val_annotation)
        self.tsf = Transform(opt.min_size, opt.max_size)


    def __getitem__(self, idx):

        ori_img, bbox, label, difficult, image_id = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        return img.copy(), img.shape[1:], bbox.copy(), label.copy(), difficult, image_id


    def __len__(self):

        return len(self.db)
        
    
    def get_class_count(self):

        return self.db.get_class_count()
        
    
    def get_class_names(self):

        return self.db.get_class_names()
