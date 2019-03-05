#%% Constants and imports

# Note to self:
# conda install pytorch-nightly -c pytorch
# pip install future
# pip install opencv-python
# pip install onnx

# Useful links:
# https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb

import cv2
from caffe2.python import core, workspace
import onnx
import caffe2.python.onnx.backend
import numpy as np
import codecs
from operator import itemgetter
import argparse
import data_loader
import os
import random
import collections
import tqdm

parser = argparse.ArgumentParser(description='Evaluation of the accuracy of an exported ONNX model.')
parser.add_argument('--data_root', required=True,
                type=str, metavar='DATASET_ROOT', help='Path to the root directory of the dataset.')
parser.add_argument('--val_file', default='minival2017.json',
                type=str, metavar='VAL_FILE', help='Name of the json file containing the validation annotation ' + \
                '(default: minival2017.json). Should be located within the dataset directory.')
global args
args = parser.parse_args()


TOP_K = 5

IMAGE_FILENAME = '/data/images/lion.jpg'
MODEL_FILENAME = 'exported_model.onnx'
CLASSLIST_FILENAME = 'classlist.txt'

# Target mean / std; should match the values used at training time
MODEL_IMAGE_SIZE = 224
MODEL_RESIZE_SIZE = 256
OVERSIZE_FACTOR = 1.3

#%% Load model
model = onnx.load(MODEL_FILENAME)
prepared_backend = caffe2.python.onnx.backend.prepare(model, device='CUDA')

val_dataset = data_loader.JSONDataset(args.data_root,
         os.path.join(args.data_root, args.val_file),
         [MODEL_IMAGE_SIZE],
         is_train=False,
         dataFormat2017 = True)


def predict(im_path):
    #%% Load and prepare image (using cv2)

    # Load image
    imgIn = cv2.imread(im_path)

    # Convert from BGR to RGB (reverse the last channel of each pixel)
    imgRgb = imgIn[...,::-1]

    # Compute target size
    dims = imgRgb.shape
    h = dims[0]; w = dims[1]; nChannels = dims[2]
    assert nChannels == 3

    shortSideSize = min(w,h)

    ############################################
    ## CHANGE
    ############################################
    '''
    # Upsize images that are smaller than the target size
    if (shortSideSize < MODEL_IMAGE_SIZE):
        scaleFactor = MODEL_IMAGE_SIZE / shortSideSize
    else:
        scaleFactor = OVERSIZE_FACTOR
        
    targetShortSideSize = round(scaleFactor * MODEL_IMAGE_SIZE)
    targetShortSideSize = max(targetShortSideSize,MODEL_IMAGE_SIZE)

    if (h < w):
        ratio = targetShortSideSize / h
        targetH = targetShortSideSize
        targetW = round(ratio * w)
    else:
        ratio = targetShortSideSize / w
        targetW = targetShortSideSize
        targetH = round(ratio * h)
    '''

    M = max(w,h) * MODEL_RESIZE_SIZE
    targetW = round(M * 1.0 / h)
    targetH = round(M * 1.0 / w)
    ############################################
    ## END CHANGE
    ############################################
    # Resize short side to RESIZE_SCALE_FACTOR * TARGET_SIZE
    imgResized = cv2.resize(imgRgb, (targetW, targetH))

    # Center crop at TARGET_SIZE * TARGET_SIZE
    startCol = (targetW//2)-(MODEL_IMAGE_SIZE//2)
    endCol = (targetW//2)+(MODEL_IMAGE_SIZE//2)
    startRow = (targetH//2)-(MODEL_IMAGE_SIZE//2)
    endRow = (targetH//2)+(MODEL_IMAGE_SIZE//2)

    imgCropped = imgResized[startRow:endRow,startCol:endCol,:]
    s = imgCropped.shape; assert(s[0] == MODEL_IMAGE_SIZE and s[1] == MODEL_IMAGE_SIZE)

    # Convert to NCHW
    imgFinal = np.transpose(imgCropped, (2,0,1))
    imgExpanded = np.expand_dims(imgFinal,0).astype('float32')

    # cv2.imshow('image',imgCropped); cv2.waitKey(0); cv2.destroyAllWindows()


    #%% Run model

    # Run the ONNX model with Caffe2
    outputs = prepared_backend.run([imgExpanded])
    outputs_softmax = outputs[0][0].astype(np.float64)

    #%% Print top K class names

    # Load class names
    with codecs.open(CLASSLIST_FILENAME, "r",encoding='utf-8', errors='ignore') as f:
        classes = f.readlines()
    classes = [x.strip() for x in classes] 
    assert len(classes) == len(outputs_softmax)

    # Find top 5 probabilities
    topKIndices = np.argsort(outputs_softmax)[-TOP_K:].tolist()[::-1]
    topKValues = [outputs_softmax[k] for k in topKIndices]
    topKClassnames = itemgetter(*topKIndices)(classes)

    #for iClass,n in enumerate(topKClassnames):
    #    print('{:>6.2%}: {:>4} {}'.format(topKValues[iClass],topKIndices[iClass],n))
    return topKIndices[0]

correct = []
np.random.seed(0)
for idx in tqdm.tqdm(np.random.permutation(len(val_dataset.imgs))):
    im_path = val_dataset.imgs[idx]
    im_target = val_dataset.targets[idx]
    correct.append(predict(os.path.join(args.data_root, im_path)) == im_target)
    if idx%100 == 99:
        print('Currently at {:.2%} with {}'.format(np.mean(correct), str(collections.Counter(correct))))

print('Finished with {:.2%} and {}'.format(np.mean(correct), str(collections.Counter(correct))))
