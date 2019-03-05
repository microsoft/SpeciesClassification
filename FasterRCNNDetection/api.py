################
#
# api.py
#
# Internal API for running the FasterRCNN framework for the wildlife classification project.
#
# The Detector class is the external entry point.
#
################

import os
import matplotlib
from tqdm import tqdm
import numpy as np
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.config import opt
import data.dataset
import data.util
import torch
from torch.autograd import Variable
from torch.utils import data as data_
import torchvision.transforms as transforms
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
import torch.utils.data
import torch
import PIL
import PIL.ImageDraw
import PIL.ImageFont


class Detector:

    def __init__(self, model_path, useGPU, n_fg_classes=1):
        ''' 
        Creates a new detection model using the weights 
        stored in the file MODEL_PATH and initializes the GPU 
        if USEGPU is set to true.
        MODEL_PATH: path to a trained detection model. 
        USEGPU: if true, the GPU will be used for faster computations.
        '''

        torch.set_num_threads(1)
        opt.load_path = model_path
        self.faster_rcnn = FasterRCNNVGG16(n_fg_class=n_fg_classes)
        self.trainer = FasterRCNNTrainer(self.faster_rcnn, n_fg_class=n_fg_classes)
        if useGPU:
            self.trainer = self.trainer.cuda()
        state_dict = torch.load(opt.load_path)
        self.trainer.load(state_dict)
        self.transforms = transforms.ToTensor()
        self.useGPU = useGPU


    def predict_image(self, img, topk=1):

        '''
        Detects objects in the provided testing images.
        IMG: PIL image fitting the input of the trained model
        TOPK: the number of bounding boxes to return. We return the
        most confident bounding boxes first. 

        RETURNs: (BBOXES, CONFS) where BBOXES is a n x 4 array,
        where each line corresponds to one bounding box. The 
        bounding box coordniates are stored in the format
        [x_min, y_min, x_max, y_max], where x corresponds to the width
        and y to the height. CONFS are the confidence values for 
        each bounding box and are a n x m array. Each row corresponds 
        to the bounding box in the same row of BBOXES and provides
        the scores for the m classes, that the model was trained to detect.
        '''

        pred_bboxes, pred_labels, pred_scores = self._run_prediction(img)
        return pred_bboxes[:topk, [1,0,3,2]], pred_scores[:topk]


    def annotate_image(self, img, topk):
        ''' 
        Detects objects in the provided testing images.
        IMG: PIL image fitting the input of the trained model
        TOPK: the number of bounding boxes to return. We return the
        most confident bounding boxes first.

        RETURNS: IMG: a PIL image with the detected bounding boxes 
        annotated as rectangles.
        '''

        pred_bboxes, pred_labels, pred_scores = self._run_prediction(img)
        draw = PIL.ImageDraw.Draw(img)
        colors = [(255,0,0),(0,255,0)]
        for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
            draw.rectangle(bbox[[1,0,3,2]], outline=colors[label])
            #font = PIL.ImageFont.truetype("sans-serif.ttf", 16)
            #draw.text(bbox[[1,0]],"Sample Text",colors[label])
        return img


    def _run_prediction(self, img, confidence_threshold=0.7):
        ''' 
        Prepare an input image for CNN processing. 
        IMG: PIL image

        RETURN: IMG as pytorch tensor in the format 1xCxHxW
        normalized according to data.dataset.caffe_normalize.
        '''

        img = img.convert('RGB')
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            img = img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            img = img.transpose((2, 0, 1))
        proc_img = data.dataset.caffe_normalize(img/255.)
        tensor_img = torch.from_numpy(proc_img).unsqueeze(0)
        if self.useGPU:
            tensor_img = tensor_img.cuda()
        
            # This preset filters bounding boxes with a score < *confidence_threshold*
        # and has to be set everytime before using predict()
        self.faster_rcnn.use_preset('visualize')
        pred_bboxes, pred_labels, pred_scores = self.faster_rcnn.predict(tensor_img, [(img.shape[1], img.shape[2])])
        box_filter = np.array(pred_scores[0]) > confidence_threshold
        return pred_bboxes[0][box_filter], pred_labels[0][box_filter], pred_scores[0][box_filter]

# ...class Detector


if __name__ == '__main__':

    det = Detector('checkpoints/fasterrcnn_07122125_0.5273599762268979', True)
    print('Loaded model.')
    image_path = 'misc/demo.jpg'
    test_image = PIL.Image.open(image_path)
    print('Working on image {}'.format(image_path))
    print(det.predict_image(test_image, 5))
    pred_bboxes, pred_scores = det.predict_image(test_image, 1000)
    pred_img = visdom_bbox(np.array(test_image.convert('RGB')).transpose((2, 0, 1)),
                    at.tonumpy(pred_bboxes[:,[1,0,3,2]]),
                    at.tonumpy([1 for _ in pred_bboxes]),
                    at.tonumpy(pred_scores),
                    label_names=['Animal', 'BG'])
    PIL.Image.fromarray((255*pred_img).transpose((1,2,0)).astype(np.uint8)).save('output.jpg')
    det.annotate_image(test_image, 5).save('output-annotate.jpg')
