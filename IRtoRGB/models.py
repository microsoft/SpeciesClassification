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

import torch.nn as nn
import torch.nn.init as init

import pretrainedmodels
from torchvision import transforms

from rgb_nir_loader import *

class EncoderDecoder(nn.Module):

    def __init__(self, inputSize, numLayers, inputfeats, outputfeats, feats, imagenet=False, fcfeats=1024):
        super(EncoderDecoder, self).__init__()

        self.numLayers = numLayers
        self.feats = feats
        self.fcfeats = fcfeats
        self.imagenet = imagenet

        self.inputSizeEnd = [int(float(x)/pow(2,self.numLayers)) for x in inputSize]
        # print(self.inputSizeEnd)
        self.featsEnd = self.feats*pow(2,self.numLayers-1)

        if (imagenet):
            self.model = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained="imagenet")
        
            ct = 0
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False
                ct += 1

            self.model.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            self.model.last_linear = nn.Linear(self.model.last_linear.in_features, fcfeats)

            self.modelFeats = nn.Sequential(*list(self.model.children())[:-1])
        else:
            self.layersEnc = nn.ModuleList()

            for i in range(0, self.numLayers):     
                self.layersEnc.append(nn.Sequential(
                                    nn.Conv2d(inputfeats if i == 0 else feats*pow(2,i-1), feats*pow(2,i), 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(feats*pow(2,i), feats*pow(2,i), 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2,2)))

            #self.linearEnc = nn.Linear(self.featsEnd*self.inputSizeEnd[0]*self.inputSizeEnd[1], fcfeats)
            self.linearEnc = nn.Conv2d(self.featsEnd, fcfeats, self.inputSizeEnd[0])

        self.layersDec = nn.ModuleList()

        for i in range(self.numLayers-1, -1, -1):     
            self.layersDec.append(nn.Sequential(
                                nn.ConvTranspose2d(feats*pow(2,i), feats*pow(2,i), 3, padding=1, output_padding=1, stride=2),
                                nn.Tanh(),
                                nn.Conv2d(feats*pow(2,i), outputfeats if i == 0 else feats*pow(2,i-1), 3, padding=1),
                                nn.Tanh()))
            
        #self.linearDec = nn.Linear(fcfeats, self.featsEnd*self.inputSizeEnd[0]*self.inputSizeEnd[1])
        self.linearDec = nn.ConvTranspose2d(fcfeats, self.featsEnd, self.inputSizeEnd[0])

    def encode(self, x):
               
        if (self.imagenet):
            l = self.model.forward(x.repeat([1,3,1,1]))
        else:
            for i in range(0, self.numLayers):
                x = self.layersEnc[i](x)

            #x = x.view(-1, self.featsEnd*self.inputSizeEnd[0]*self.inputSizeEnd[1])

            l = self.linearEnc(x)

        return l

    def decode(self, x):
        
        x = self.linearDec(x)
        #x = l.view(-1, self.featsEnd, self.inputSizeEnd[0], self.inputSizeEnd[1])

        for i in range(0, self.numLayers):  
            x = self.layersDec[i](x)

        return x

    def forward(self, x):               
        out = self.decode(self.encode(x))

        #x = out
        #x = torch.mean(x, 1, keepdim=True) + out

        x = torch.sum(x*out[:,0:3,:,:] + out[:,3:6,:,:], 1, keepdim=True)

        return x

class IRtoRGBModel(nn.Module):

    def __init__(self, inputSize, numLayers, inputfeats, outputfeats, maxfeats):
        super(IRtoRGBModel, self).__init__()

        self.encdec = EncoderDecoder(numLayers, inputSize, inputfeats, outputfeats, maxfeats)

    def forward(self, x):
        x = self.encdec(x)
        return x

class IRtoRGB(nn.Module):
    def __init__(self, model_file, inputSize, numLayers, inputfeats, outputfeats, maxfeats, useGPU=True):
        super(IRtoRGB, self).__init__()

        self.useGPU = useGPU

        if(model_file):
            self.data = load_model(model_file, self.useGPU)
            self.init(numLayers, inputSize, inputfeats, outputfeats, maxfeats)

            try:
                self.model.load_state_dict(self.data.model_dict)
            except Exception as e:
                print(str(e))   
        else:
            self.init(numLayers, inputSize, inputfeats, outputfeats, maxfeats)

        if(useGPU):
            self.model = self.model.cuda()

    def loadOptimizer(self, optimizer):        
        optimizer.load_state_dict(self.data.optimizer_dict)
        
        print("prec: %f" % (self.data.best_prec))

        return self.data.best_prec, self.data.start_epoch

    def init(self, numLayers, inputSize, inputfeats, outputfeats, maxfeats):
        
        model = IRtoRGBModel(numLayers, inputSize, inputfeats, outputfeats, maxfeats)    

        self.model = model
        self.loader = ImageLoader(inputSize[0])        

    def forward(self, x):
        return self.model.forward(x)

    def predict_image(self, path):

        imgIn = self.loader.load_image(path)

        input, padding = self.loader.process_image(imgIn, False)

        output = self.inference(input)

        output = np.clip((output + 1)/2, 0, 1)

        output = np.squeeze(output[:,:,padding[1]:output.shape[2]-padding[3],padding[0]:output.shape[3]-padding[2]])

        return output

    def inference(self, input):

        with torch.no_grad():
            input = input.unsqueeze(0)
            if self.useGPU:
                input = input.cuda()
            input_var = torch.autograd.Variable(input, requires_grad=False)
            self.model.eval()
            output = self.model(input_var)
            return output.cpu().numpy().astype(np.float)
