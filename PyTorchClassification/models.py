################
#
# models.py
#
# Defines the architectures for models training for the species classification API.
#
# The ClassificationModel class also defines the interface that the API package uses to run inference.
#
# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
#
################


import os
import shutil
import time
import copy
from enum import Enum
import numpy as np
import pretrainedmodels
import torchvision.models
from data_loader import *
import pretrained.inceptionv3
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms


class ModelType(Enum):

    resnet50 = 0
    inceptionv4 = 1  
    inceptionresnetv2 = 2
    inceptionv4_resnet50 = 3
    inceptionv4_inceptionresnetv2 = 4
    inceptionv3 = 5
    resnext101 = 6
    inceptionv4_resnext101 = 7

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


class EnsembleAverage(nn.Module):

    def __init__(self, modelIncept, modelResnet, num_classes, input_sizes):

        super(EnsembleAverage, self).__init__()
        self.modelIncept = modelIncept
        self.modelResnet = modelResnet
        self.input_sizes = input_sizes
        assert len(input_sizes) == 2, 'Two input resolutions need to be specified for ensembles.'


    def forward(self, x):

        input_incept = F.interpolate(x, (self.input_sizes[0], self.input_sizes[0]), mode='bilinear') 
        input_resnet = F.interpolate(x, (self.input_sizes[1], self.input_sizes[1]), mode='bilinear') 
        return (self.modelIncept(input_incept) + self.modelResnet(input_resnet))/2


class EnsembleDoubleFC(nn.Module):

    def __init__(self, modelIncept, modelResnet, num_classes, intermediate_dim=None):

        super(EnsembleDoubleFC, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = modelIncept.last_linear.in_features + modelResnet.last_linear.in_features
        self.modelIncept = modelIncept
        self.modelResnet = modelResnet
        self.linear1 = nn.Linear(self.modelIncept.last_linear.in_features + self.modelResnet.last_linear.in_features, intermediate_dim)
        self.last_linear = nn.Linear(intermediate_dim, num_classes)
        self.last_linear.stddev = 0.001
        self.classifier = nn.Sequential(
                self.linear1,
                nn.ReLU(inplace = True),
                nn.BatchNorm1d(intermediate_dim),
                nn.Dropout(0.5),
                self.last_linear)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.modelInceptFeats = nn.Sequential(*list(modelIncept.children())[:-1])
        self.modelResnetFeats = nn.Sequential(*list(modelResnet.children())[:-1])


    def forward(self, x):

        x0 = self.modelInceptFeats(x)    
        x1 = self.modelResnetFeats(x)
        x0x1 = torch.cat((x0, x1), 1).squeeze()
        return self.classifier(x0x1)


class ClassificationModel(nn.Module):

    def __init__(self, model_file, image_sizes, useGPU=True, model_type=None, classes=None):

        super(ClassificationModel, self).__init__()

        if isinstance(model_file,str):
            model_file = [model_file]
        
        if (isinstance(image_sizes,int)):

            # If the model we're loading is an ensemble, we need multiple image sizes
            assert (len(model_file)==1), 'List of image sizes required for multiple models'
            image_sizes = [image_sizes]

        self.useGPU = useGPU
        self.image_sizes = image_sizes

        if model_file and len(model_file) > 1 and model_type!=None and classes!=None:
            self.initSubModelsFromFile(model_file, model_type, classes)
        elif model_file:            
            self.initFromFile(model_file[0], model_type, classes)
        else:
            self.init(model_type, classes, self.image_sizes)

        if(useGPU):
            self.model = self.model.cuda()


    def loadOptimizer(self, optimizer): 
        
        #optimizer.load_state_dict(self.data.optimizer_dict)
        print("prec 1,3,5: %f %f %f" % (self.data.best_prec1, self.data.best_prec3, self.data.best_prec5))

        return self.data.best_prec1, self.data.best_prec3, self.data.best_prec5, self.data.start_epoch


    def initFromFile(self, model_file, model_type=None, classnames=None):

        self.data = load_model(model_file, self.useGPU)
        assert self.data, 'Invalid checkpoint file'
        self.model_type = model_type if model_type else self.data.model_type
        print('Initializing a model of type {}'.format(str(self.model_type)))

        self.classnames = classnames if classnames else self.data.classnames
        num_classes = len(self.classnames)

        self.init(self.model_type, self.classnames, self.image_sizes, loadImagenetWeights=False)
        
        try:
            self.model.load_state_dict(self.data.model_dict)
        except Exception as e:
            print(str(e))


    def initSubModelsFromFile(self, model_file, model_type, classnames):

        self.model_type = model_type
        num_classes = len(classnames)

        self.init(self.model_type, classnames, self.image_sizes, loadImagenetWeights=False)
       
        print("Loading inception")
        dataIncept = load_model(model_file[0])
        try:
            self.model.modelIncept.load_state_dict(dataIncept.model_dict)
        except Exception as e:
            print(str(e))
           
        print("Loading resnet")
        dataResnet = load_model(model_file[1])
        try:
            self.model.modelResnet.load_state_dict(dataResnet.model_dict)
        except Exception as e:
            print(str(e))
        
        self.classnames = classnames


    def init(self, model_type, classnames, image_sizes, loadImagenetWeights=True):

        self.model_type = model_type
        num_classes = len(classnames)
        self.classnames = classnames
        pretrained = loadPretrained = "imagenet" if loadImagenetWeights else None

        if self.model_type == ModelType.inceptionv3:
            model = pretrained.inceptionv3.inception_v3(num_classes=num_classes, 
                               pretrained=loadPretrained, aux_logits=False)
            model.last_linear = model.fc

        elif (self.model_type == ModelType.inceptionv4):
            model = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained=loadPretrained)
            ct = 0
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = True
                ct += 1
            model.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)    

        elif (self.model_type == ModelType.inceptionresnetv2):

            model = pretrainedmodels.__dict__["inceptionresnetv2"](num_classes=1000, pretrained=loadPretrained)
        
            ct = 0
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = True
                ct += 1
            model.avgpool_1a = nn.AdaptiveAvgPool2d((1,1))
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)    

        elif (self.model_type == ModelType.resnext101):
            model = pretrainedmodels.__dict__["se_resnext101_32x4d"](num_classes=1000, pretrained=loadPretrained)
            model.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)    

        elif (self.model_type == ModelType.resnet50):
            model = models.resnet50(pretrained=True)
        
            ct = 0
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False
                ct += 1

            model.fc  = nn.Linear(model.fc.in_features, num_classes)

        elif (self.model_type == ModelType.inceptionv4_inceptionresnetv2):
            modelIncept = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained=loadPretrained)
            modelResnet = pretrainedmodels.__dict__["inceptionresnetv2"](num_classes=1000, pretrained=loadPretrained)

            '''
            ct = 0
            for child in modelIncept.children():
                #print("Child %d %s" % (ct, child))
                if (ct < 19):
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1

            ct = 0
            for child in modelResnet.children():
                #print("Child %d %s" % (ct, child))
                if (ct < 11):
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1
            '''
            modelIncept.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            modelResnet.avgpool_1a = nn.AdaptiveAvgPool2d((1,1))
            modelIncept.last_linear = nn.Linear(modelIncept.last_linear.in_features, num_classes)    
            modelResnet.last_linear = nn.Linear(modelResnet.last_linear.in_features, num_classes)   
            
            model = EnsembleAverage(modelIncept, modelResnet, num_classes, self.image_sizes)

        elif (self.model_type == ModelType.inceptionv4_resnext101):
            modelIncept = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained="imagenet")
            modelIncept.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            modelIncept.last_linear = nn.Linear(modelIncept.last_linear.in_features, num_classes)
    
            modelResnet = pretrainedmodels.__dict__["se_resnext101_32x4d"](num_classes=1000, pretrained="imagenet")
            modelResnet.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            modelResnet.last_linear = nn.Linear(modelResnet.last_linear.in_features, num_classes)    

            model = EnsembleAverage(modelIncept, modelResnet, num_classes, self.image_sizes)

        else: #if (self.model_type == Model.inceptionv4_resnet50):
            modelIncept = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained=loadPretrained)        
            modelResnet = models.resnet50(pretrained=True)

            ct = 0
            for child in modelIncept.children():
                #print("Child %d %s" % (ct, child))
                if (ct < 19):
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1

            ct = 0
            for child in modelResnet.children():
                #print("Child %d %s" % (ct, child))
                if (ct < 11):
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1

            modelIncept.last_linear = nn.Linear(modelIncept.last_linear.in_features, num_classes)    
            modelResnet.fc  = nn.Linear(modelResnet.fc.in_features, num_classes)

            model = EnsembleAverage(modelIncept, modelResnet, num_classes)

        self.model = model
        self.loader = ImageLoader(self.image_sizes)


    def forward(self, x):

        return self.model.forward(x)


    def predict_image(self, path, topK=1, multiCrop=False, bboxes=None, all=False):

        with torch.no_grad():
            imgIn = self.loader.load_image(path)
            return self.predict_from_image(imgIn, topK, multiCrop, bboxes, all)


    def predict_from_image(self, imgIn, topK=1, multiCrop=False, bboxes=None, all=False):

        with torch.no_grad():
            inputs = self.loader.process_image(imgIn, False, multiCrop, bboxes)
            numCrops = len(inputs)
            for i in range(0, numCrops):
                input = torch.Tensor(inputs[i])
                if (i >0):
                    output = output + self.inference(input)
                    #output = torch.max(output, self.inference(input))
                else:
                    output = self.inference(input)
            output /= numCrops

            ids, vals = self.get_preds(output, topK, all)
            classes = []
            for id in ids[0,:]:
                classes.append(self.classnames[id])
            return classes, vals[0,:]


    def get_preds(self, output, topK=1, all=False):

        with torch.no_grad():
            output = F.softmax(output, dim=1)
            if all: pred_vals, pred_inds = output, torch.arange(output.numel()).unsqueeze(0)
            else:   pred_vals, pred_inds = output.data.topk(topK)

            if (self.useGPU):
                pred_inds = pred_inds.cpu().numpy().astype(np.int)
                pred_vals = pred_vals.cpu().numpy().astype(np.float)
            else:
                pred_inds = pred_inds.numpy().astype(np.int)
                pred_vals = pred_vals.numpy().astype(np.float)

            return pred_inds, pred_vals


    def inference(self, input):

        with torch.no_grad():
            input = input.unsqueeze(0)
            if self.useGPU:
                input = input.cuda()
            input_var = torch.autograd.Variable(input, requires_grad=False)
            self.model.eval()
            output = self.model(input_var)
            return output

    def get_classnames(self):
        return self.classnames
