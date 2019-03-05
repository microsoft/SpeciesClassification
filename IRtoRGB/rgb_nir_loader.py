import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np
import torch
import shutil

import matplotlib.pyplot as plt

import numbers

import csv

import cv2

def show_image(input):
    img = np.transpose(input.data.numpy(),[1, 2, 0])
    plt.imshow(img)

    plt.show()

def show_images(inputA, inputB, inputC):

    inputA = inputA.cpu().data.numpy()
    inputB = inputB.cpu().data.numpy()
    inputC = inputC.cpu().data.numpy()

    #print(inputA.shape)

    for n in range(0, inputA.shape[0]):
        imgA = np.transpose(inputA[n,:,:],[1, 2, 0])
        imgB = np.transpose(inputB[n,:,:],[1, 2, 0])
        imgC = np.transpose(inputC[n,:,:],[1, 2, 0])

        if (imgA.shape[2] == 3):
            imgA = imgA[:,:,[2,1,0]]
        else:
            imgA = np.concatenate((imgA,imgA,imgA), axis=2)

        if (imgB.shape[2] == 3):
            imgB = imgB[:,:,[2,1,0]]
        else:
            imgB = np.concatenate((imgB,imgB,imgB), axis=2)

        if (imgC.shape[2] == 3):
            imgC = imgC[:,:,[2,1,0]]
        else:
            imgC = np.concatenate((imgC,imgC,imgC), axis=2)

        imgA = np.clip((imgA + 1)/2, 0, 1)
        imgB = np.clip((imgB + 1)/2, 0, 1)
        imgC = np.clip((imgC + 1)/2, 0, 1)

        imgAG = np.mean(imgA, 2, keepdims=True)
        imgAG = np.concatenate((imgAG,imgAG,imgAG), axis=2)

        cv2.imshow("", np.hstack((imgA, imgAG, imgB, imgC)))

        cv2.waitKey(200)

def save_model(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_model(resume, useGPU = True):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))

        if (useGPU):
            checkpoint = torch.load(resume)
        else:
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec'] if 'best_prec' in checkpoint else 0

        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            module = k[0:7] # check for 'module.' of dataparallel
            name = k[7:] # remove 'module.' of dataparallel            

            if (module == 'module.'):
                k = name

            if ( k[0:6] == 'model.'):
                k = k[6:]

            new_state_dict[k] = v

            #print("%s" % (k))

        model_dict = new_state_dict        
        optimizer_dict = checkpoint['optimizer']

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume, checkpoint['epoch']))

        data.best_prec = best_prec
        data.start_epoch = start_epoch
        data.model_dict = model_dict
        data.optimizer_dict = optimizer_dict

        return data 

    else:
        print("=> no checkpoint found at '{}'".format(resume))

class ImageLoader():
    def __init__(self, sz):
        # augmentation params
        self.im_size = [sz, sz]  # can change this to train on higher res
        self.mu_data = [0.5, 0.5, 0.5]
        self.std_data = [0.5, 0.5, 0.5]

        # augmentations
        self.center_crop = transforms.CenterCrop((2*self.im_size[0], 2*self.im_size[1]))                
        self.random_crop = transforms.RandomCrop((self.im_size[0], self.im_size[1]),pad_if_needed=True)
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def load_image(self, path):
        img =  Image.open(path).convert('RGB')
        return img

    def process_image(self, img, is_train, showImage=False):
        
        if(is_train):
            img = self.random_crop(img)
        else:
            div = 32

            #img = self.center_crop(img)

            paddedW = div*np.ceil(img.size[0]/div)
            paddedH = div*np.ceil(img.size[1]/div)

            padL = int((paddedW - img.size[0])/2)
            padR = int(paddedW - img.size[0] - padL)

            padT = int((paddedH - img.size[1])/2)
            padB = int(paddedH - img.size[1] - padT)

            self.pad = transforms.Pad((padL, padT, padR, padB)) 
            img = self.pad(img)

        img = self.tensor_aug(img)

        if(showImage):
            show_image(img)             

        #img[0:3,:,:,] = np.clip(img[0:3,:,:,].pow(2.2),0,1)
        #img[3:4,:,:,] = np.clip(img[3:4,:,:,]*1.5,0,1)

        img = 2*img-1

        #img = self.norm_aug(img)
        if(is_train):
            return img
        else:
            return img, (padL, padT, padR, padB)

class RGBNIR(data.Dataset):
    def __init__(self, root, list_file, sz, is_train=True):

        self.imgs = []

        # load annotations
        print('Loading image list from: ' + os.path.basename(list_file))
        with open(list_file) as data_file:
            image_data = csv.reader(data_file)

            for row in image_data:
                self.imgs.append(row)

        # print out some stats
        print ('\t' + str(len(self.imgs)) + ' images')

        self.root = root
        self.is_train = is_train
        self.loader = ImageLoader(sz)
        
    def __getitem__(self, index):
        
        nirpath = self.root + self.imgs[index][0]
        rgbpath = self.root + self.imgs[index][1]

        #print("%s\t%s" % (nirpath, rgbpath))

        nirimg = self.loader.load_image(nirpath)
        rgbimg = self.loader.load_image(rgbpath)

        n, n, n = nirimg.split()
        r, g, b = rgbimg.split()
        imgIn = Image.merge("RGBA", (r, g, b, n))

        img = self.loader.process_image(imgIn, self.is_train)

        #print(img.shape)

        return img[0:3,:,:,], img[3:4,:,:,]

    def __len__(self):
        return len(self.imgs)
