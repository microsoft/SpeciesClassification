################
#
# data_loader.py
#
# Defines the loading of data sets (from .json files), models (from PyTorch model files), and images
# for training and inference.
#
################

import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import numpy as np
import torch
import shutil
import math
import numbers
import zipfile
from collections import OrderedDict
from torch.utils.data.sampler import Sampler
from torch.distributed import get_world_size, get_rank


def show_image(input, bbox=None):

    import matplotlib.pyplot as plt

    img = np.transpose(input.data.numpy(),[1, 2, 0])
    plt.imshow(img)

    if (bbox):

        x = [bbox[0], bbox[0], bbox[0]+bbox[2], bbox[0]+bbox[2], bbox[0]]
        y = [bbox[1], bbox[1]+bbox[3], bbox[1]+bbox[3], bbox[1], bbox[1]]

        plt.plot(x, y)

    plt.show()


def deploy_model(filein, fileout):
    """
    Loads a model from a checkpoint, then re-saves it in a format that is more practical
    for deployment for inference-only applications.
    """

    print("=> deploying checkpoint '{}'".format(filein))

    checkpoint = torch.load(filein, map_location=lambda storage, loc: storage)

    deploy_checkpoint = {
            'epoch' : checkpoint['epoch'],
            'state_dict': checkpoint['state_dict'],
            'classnames' : checkpoint['classnames'],
            'model_type' : checkpoint['model_type']}

    torch.save(deploy_checkpoint, fileout)


def save_model(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves a model to a checkpoint.
    """

    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_model(filename, useGPU=True):
    """
    Loads a model from a checkpoint.
    """

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))

        if useGPU:
            cuda_device = torch.cuda.current_device()
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage.cuda(cuda_device))
        else:
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint else 0
        best_prec3 = checkpoint['best_prec3'] if 'best_prec3' in checkpoint else 0
        best_prec5 = checkpoint['best_prec5'] if 'best_prec5' in checkpoint else 0

        state_dict = checkpoint['state_dict']
        classnames = checkpoint['classnames']
        model_type = checkpoint['model_type']

        print('Loaded %d classes' % len(classnames))

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            module = k[0:7] # check for 'module.' of dataparallel
            name = k[7:] # remove 'module.' of dataparallel            

            if k[:7] == 'module.':
                k = k[7:]
            if k[:2] == '1.':
                k = k[2:]
            if k[:6] == 'model.':
                k = k[6:]

            new_state_dict[k] = v

            #print("%s" % (k))

        model_dict = new_state_dict        
        optimizer_dict = checkpoint['optimizer'] if 'optimizer' in checkpoint else None

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(filename, start_epoch))

        data.best_prec1 = best_prec1
        data.best_prec3 = best_prec3
        data.best_prec5 = best_prec5
        data.start_epoch = start_epoch
        data.classnames = classnames
        data.model_dict = model_dict
        data.optimizer_dict = optimizer_dict
        data.model_type = model_type

        return data 

    else:
        print("=> no checkpoint found at '{}'".format(filename))

# ...def load_model(filename, useGPU=True)


class ImageLoader():

    def __init__(self, image_sizes):
        # The largest image size is used as target size in preprocessing
        # The scaling to the proper size should be done within the model 
        self.im_size = [max(image_sizes), max(image_sizes)] 
        self.mu_data = [0.5, 0.5, 0.5]
        self.std_data = [0.5, 0.5, 0.5]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.random_crop = transforms.RandomCrop((self.im_size[0], self.im_size[1]),pad_if_needed=True)
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.flip = transforms.RandomHorizontalFlip(1.0)
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)
        self.resize = transforms.Resize((self.im_size[0], self.im_size[1]))
        self.resize_for_crop = transforms.Resize((int(1.14 * self.im_size[0]), int(1.14 * self.im_size[1])))
        self.multi_crop = transforms.TenCrop((self.im_size[0], self.im_size[1]))

    def load_image(self, path):
        img =  Image.open(path).convert('RGB')
        return img

    def process_image(self, img, is_train, multi_crop = False, bboxes = None, showImage = False):
        if bboxes is None:
            bboxes = []
        # In training, random scaling, flipping, and color augmentation
        if is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
            img = self.tensor_aug(img)
            img = self.norm_aug(img)
            return img
        # In validation
        else:
            # We will collect all crops of the image in *imgs*
            min_size = min(img.size)
            scale_ratio = min(self.im_size) / min_size * 1.3
            resized_img = F.resize(img, (int(img.size[1]*scale_ratio), int(img.size[0]*scale_ratio)))
            imgs = [self.center_crop(resized_img)]

            # Add all bboxes and their flip
            for bbox in bboxes:
                bbox_shape = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
                padding = bbox_shape.max() * 0.1
                # Add offset to the shorter side to crop a square patch
                offset = (bbox_shape - np.min(bbox_shape))[::-1] // 2
                bbox_crop = img.crop((bbox[1] - padding - offset[1], 
                           bbox[0] - padding - offset[0],
                           bbox[3] + padding + offset[1], 
                           bbox[2] + padding + offset[0])) # (w - crop_w, h - crop_h, w, h))
                #img.save('crop{}.jpg'.format(np.random.randint(0,10)))
                bbox_crop = self.resize(bbox_crop)
                imgs.append(bbox_crop)
                imgs.append(self.flip(bbox_crop))

            # Add all crops 
            if multi_crop:
                imgs.append(self.flip(self.center_crop(resized_img)))
                imgs.extend(self.multi_crop(self.resize_for_crop(img)))

            # Convert everything to normalized tensor
            tensor_imgs = []
            for img in imgs:
                img = self.tensor_aug(img)
                img = self.norm_aug(img)
                tensor_imgs.append(img)
            return tensor_imgs

# ...class ImageLoader()


class DistributedBalancedSampler(Sampler):
    """Sampler for distributed training. It draws on average the same number
       of samples from each class even if the dataset itself is unbalanced.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.labels = dataset.get_labels()
        self.unique_labels, self.counts = np.unique(self.labels, return_counts=True)
        self.num_classes = len(self.unique_labels)
        self.num_samples_per_class = int(math.ceil(len(dataset) * 1.0 / self.num_classes / self.num_replicas))
        self.num_samples = self.num_samples_per_class * self.num_classes
        self.total_size = self.num_samples * self.num_replicas
        # we will create only len(dataset) indices per epoch to make a 
        # consistent experience for different samplers
        # Precompute a list of all images per class for speed reasons
        self.class_to_image_idx = {c:[] for c in self.unique_labels}
        for idx, label in enumerate(self.labels):
            self.class_to_image_idx[label].append(idx)
        # This will be a list of all images per class, from which we will draw without 
        # replacement and repopulate it once all images of a class have been drawn
        # We will add class images later and keep the list across epochs to make
        # sure that we always draw all images of a category before repeating images
        self.class_to_avail_images = {c:[] for c in self.unique_labels}


    def get_shuffled_class_images(self, label, generator):
        ''' Takes the list of images of class LABEL and returns 
          a shuffled copy of it '''
        tmp = self.class_to_image_idx[label].copy()
        image_perm = list(torch.randperm(len(tmp), generator=generator))
        return np.array(tmp)[image_perm].tolist()


    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # We iterate max_images_per_class times, and each iteration, we go over 
        # each class and pick a sample of it
        indices = []
        for i in range(self.num_samples_per_class):
            # replace this by torch
            class_perm = list(torch.randperm(self.num_classes, generator=g))
            for c_id in class_perm:
                cur_class = self.unique_labels[c_id]
                if not self.class_to_avail_images[cur_class]:
                    self.class_to_avail_images[cur_class] = self.get_shuffled_class_images(cur_class, g)
                indices.append(self.class_to_avail_images[cur_class].pop())

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) % self.num_replicas == 0

        # subsample
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

# ...class DistributedBalancedSampler(Sampler)


class JSONDataset(data.Dataset):
    '''
    Dataset class for loading JSON datasets, which use the COCO camera traps .json format.    
    '''
    def __init__(self, root, ann_file, image_sizes, is_train, dataFormat2017=False, multi_crop=False,
                 percentUse=100, bbox_predictions=None, label_smoothing=0, bg_classes = None):
        ''' Initializes a dataset class.
        Paramters:
            *root* Path to the root directory of the dataset. The training and validation json paths will be
            relative to this one
            *ann_file* Path to json file containing the annotations, relative to *root*
            *image_sizes* List of integers representing the sizes of the input images for each model in an ensemble. If
            only one model is used, pass a list with one element, e.g. [224]
            *is_train* boolean representing whether this is the training or validation dataset. This value determines
            the preprocessing applied to the images.
            *dataFormat2017* boolean representing whether this is the 2017 annotation format. If False, the 2018 format is
            assumed.
            *multi_crop* If true, we perform over-sampling on load. This is useful for evaluation.
            *percentUse* Integer representing the percentage of data to use. Useful for profiling.
            *bbox_predictions* Optional path to corresponding bounding box annotations for the images in this dataset.
            *label_smoothing* Value in [0.0, 1.0) representing the amount of smoothing applied to targets. If 0.0, we
            force the network to predict with a confidence of 1.0 to predict the class. For values > 0, the target will
            be smoothed considering the taxonomy of the classes.
            *bg_classes* This class allows to use certain classes as background images, which means the target is 0 probability
            for each output element. The output corresponding output elements of the classes selected by bg_classes
            will be unused.
        '''
        # load annotations
        # import pdb; pdb.set_trace()
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            im_to_cat = {aa['image_id']:aa['category_id'] for aa in ann_data['annotations']}
            self.classes = [im_to_cat[im_id] for im_id in self.ids]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        if (dataFormat2017):
            # self.tax_levels = ['id', 'name', 'supercategory']
            self.tax_levels = ['id', 'name']
        else:
            self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                               #8142, 4412,    1120,     273,     57,      25,       6

        self.taxonomy, self.classes_taxonomic, self.classnames = self.load_taxonomy(
                                             ann_data, self.tax_levels, self.classes)
        # Set targets 
        if label_smoothing > 0:
            # Build a list of the taxonomic assignment of each class
            # This is a list of list
            # tax_assignments[0] describes the assignment of each class at self.tax_levels[0]
            # tax_assignments[1] at the level of self.tax_levels[1], and so on
            tax_assignments = list(zip(*[[cc[tax_level] for tax_level in self.tax_levels] 
                                                        for cc in ann_data['categories']]))
            # Permute the class order to be 0, ..., num_classes-1 by assuming that 
            # tax_assignments[0] contains the class_ids as integer
            # First, compute how to permute the classes
            cat_permutation = np.argsort(tax_assignments[0])
            # Then apply the permutation to all lists
            # We cannot permute everything at once using numpy arrays as there are different dtype
            for tax_level in range(len(tax_assignments)):
                tax_assignments[tax_level] = np.array(tax_assignments[tax_level])[cat_permutation]
                # Also cut off the genus of the family name in 2017 format
            if dataFormat2017 and isinstance(tax_assignments[1][0], str):
                tax_assignments[1] = [ss.split(' ')[0] for ss in tax_assignments[1]]
                tax_assignments[1] = np.array(tax_assignments[1])
            else:
                raise Exception('Taxonomic label smoothing is not yet supported for the ' + \
                                '2018 annotation format.')

            # We create a matrix of realtionships of shape num_classes x num_classes
            # For now, we will store a number between 0 and len(self.tax_lavels)
            # with higher numbers denoting a closer relationship
            self.relationship = np.zeros((self.get_num_classes(), self.get_num_classes()))
            for tax_level, levelname in list(enumerate(self.tax_levels))[::-1]:
                assingm = tax_assignments[tax_level]
                self.relationship[assingm[:,None] == assingm[None,:]] = len(self.tax_levels) - tax_level

            # Compute the probability mass to be distributed for same genus, same family, etc
            # Start with the relative weights for each level: 2**(level)
            prob_per_tax_level = np.array([2**i for i in range(len(self.tax_levels) - 1)])
            # Distribute (1 - label_smoothing) according to these weights
            prob_per_tax_level = label_smoothing * prob_per_tax_level / np.sum(prob_per_tax_level)
            # Prob mass for unrelated classes as all probabilities have to be non-zero
            eps = prob_per_tax_level[0] * 0.1
            prob_per_tax_level[0] -= eps
            # Add probability mass for the same class predicition and unrelated class prediciton
            prob_per_tax_level = [eps] + prob_per_tax_level.tolist() + [1 - label_smoothing] 

            # Now convert the numbers in the matrix to probabilities
            self.targets = np.zeros_like(self.relationship, dtype=np.float64)
            # We will distribute prob_per_tax_level[LEVEL] across all entries in a row with entry LEVEL
            # Accumulate left over prob mass in case a class does not have any other classes with same 
            # family across the next level 
            # We will start from to most specific tax level and go more and more generic
            leftover_prob_mass = np.zeros((len(self.targets),))
            for tax_level in range(len(prob_per_tax_level))[::-1]:
                per_row_count = np.sum(self.relationship==tax_level, axis=1)
                prob_mass = prob_per_tax_level[tax_level] + leftover_prob_mass
                self.targets[self.relationship==tax_level] = np.repeat(prob_mass / per_row_count, per_row_count)
                leftover_prob_mass = prob_per_tax_level[tax_level] * (per_row_count==0)
            # Create a memory efficient target representation for each sample
            assert not np.any(np.isclose(0, self.targets))
            self.targets = self.targets.tolist()
            self.targets = [np.array(aa) for aa in self.targets]
            self.targets = [self.targets[cc].astype(np.float32) for cc in self.classes]
            print("The division-by-zero-error is handled properly, so don't worry.")

            self.bg_classes = bg_classes
            if isinstance(bg_classes, list):
                for idx in range(len(self.classes)):
                    if np.any(np.isclose(self.classes[idx], self.bg_classes)):
                        self.targets[idx][:] = 0
        else:
            self.targets = self.classes


        # print out some stats
        print ('\t' + str(len(self.imgs)) + ' images')
        print ('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = ImageLoader(image_sizes)
        self.multi_crop = multi_crop

        self.imagesInZip = (".zip" in self.root)
        if (self.imagesInZip):
            self.archive = zipfile.ZipFile(self.root, 'r')

        numToUse = int((len(self.imgs)*percentUse)/100)
        self.imgs = self.imgs[:numToUse]
        self.ids = self.ids[:numToUse]
        self.classes = self.classes[:numToUse]

        if bbox_predictions:
            img_id_to_idx = {image_id:idx for idx,image_id in enumerate(self.ids)}
            self.bboxes = [None for _ in range(len(self.ids))]
            self.bbox_labels = [None for _ in range(len(self.ids))]
            self.bbox_scores = [None for _ in range(len(self.ids))]
            loaded_dict = np.load(bbox_predictions)
            image_ids = loaded_dict['image_ids']
            for image_id, bbox, bbox_labels, bbox_scores in zip(image_ids,
                                                   loaded_dict['pred_bboxes'],
                                                   loaded_dict['pred_labels'],
                                                   loaded_dict['pred_scores']):
                if len(bbox) > 0:
                    assert image_id[0].tolist() in img_id_to_idx, 'Didn\'t find image for bounding box, ' + \
                                                                  'maybe it\'s the wrong json file?'
                    idx = img_id_to_idx[image_id[0].tolist()]
                    self.bboxes[idx] = bbox
                    self.bbox_labels[idx] = bbox_labels
                    self.bbox_scores[idx] = bbox_scores

        else:
            self.bboxes = None
            self.bbox_labels = None
            self.bbox_scores = None
        self.label_smoothing = label_smoothing


    def distanceMatrix(self):

        D = np.zeros((len(self.classes_taxonomic),len(self.classes_taxonomic)))

        norm = 1/len(self.tax_levels)

        for i in range(len(self.classes_taxonomic)):
            print("%d" % i, end=' ')
            for j in range(i, len(self.classes_taxonomic)):
                eq = np.equal(self.classes_taxonomic[i],self.classes_taxonomic[j])
                D[i,j] = np.argmax(eq)*norm
                D[j,i] = D[i,j]

        return D


    def load_taxonomy(self, ann_data, tax_levels, classes):

        # loads the taxonomy data and converts to ints
        taxonomy = {}
        classnames = {}
        if 'categories' in ann_data.keys():
            num_classes = len(ann_data['categories'])
            for tt in tax_levels:
                tax_data = [aa[tt] for aa in ann_data['categories']]
                _, tax_id = np.unique(tax_data, return_inverse=True)
                taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))

            for cat in ann_data['categories']:
                classnames[cat['id']] = cat['name']
        else:
            # set up dummy data
            for tt in tax_levels:
                taxonomy[tt] = dict(zip([0], [0]))

        # create a dictionary of lists containing taxonomic labels
        classes_taxonomic = {}
        for cc in np.unique(classes):
            tax_ids = [0]*len(tax_levels)
            for ii, tt in enumerate(tax_levels):
                tax_ids[ii] = taxonomy[tt][cc]
            classes_taxonomic[tax_ids[0]] = tax_ids
        return taxonomy, classes_taxonomic, classnames


    def __getitem__(self, index):
        im_id = self.ids[index]
        species_id = self.targets[index]

        if self.bboxes is not None and self.bboxes[index] is not None:
        #    bbox_scores = self.bbox_scores[index]
        #    most_confident_bbox = np.argmax(bbox_scores)
        #    bbox = self.bboxes[index][most_confident_bbox,:]
            bboxes = self.bboxes[index]
        else:
            bboxes = []

        if self.imagesInZip:
            path = archive.open(self.imgs[index])
        else:
            path = self.root + self.imgs[index]

        raw_image = self.loader.load_image(path)
        imgs = self.loader.process_image(raw_image, self.is_train, self.multi_crop, bboxes)
        return imgs, im_id, species_id


    def __len__(self):

        return len(self.imgs)


    def get_labels(self):

        return self.classes


    def get_num_classes(self):
        assert np.max(self.classes) < len(self.classnames)
        return len(self.classnames)

# ...class JSONDataset(data.Dataset)
