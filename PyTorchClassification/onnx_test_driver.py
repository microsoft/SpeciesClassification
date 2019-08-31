###
#
# onnx_test_driver.py
#
# Example driver to run an image through the .onnx version of our species 
# classification model.
#
# Installation notes:
#
# conda install pytorch-nightly -c pytorch
# pip install future
# pip install opencv-python
# pip install onnx
# pip install caffe2
#
# Useful links:
#
# https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb
#
###

#%% Constants and imports

import cv2
import onnx
import caffe2.python.onnx.backend
import numpy as np
import codecs
from operator import itemgetter

TOP_K = 5

IMAGE_FILENAME = '/data/species_classification/190215-meer-full.jpg'
MODEL_FILENAME = '/data/species_classification/sc_all_extended_ensemble_resnext_inceptionV4_560_2019.08.27_model.onnx'
CLASSLIST_FILENAME = '/data/species_classification/sc_all_extended_ensemble_resnext_inceptionV4_560_2019.08.27_classes.txt'

MODEL_IMAGE_SIZE = 224
MODEL_RESIZE_SIZE = 256
OVERSIZE_FACTOR = 1.3


#%% Load and prepare image (using cv2)

# Load image
imgIn = cv2.imread(IMAGE_FILENAME)

# Convert from BGR to RGB (reverse the last channel of each pixel)
imgRgb = imgIn[...,::-1]

# Compute target size
dims = imgRgb.shape
h = dims[0]; w = dims[1]; nChannels = dims[2]
assert nChannels == 3

# Resize image such that smaller side is MODEL_RESIZE_SIZE long
M = max(w,h) * MODEL_RESIZE_SIZE
targetW = round(M * 1.0 / h)
targetH = round(M * 1.0 / w)

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


#%% Load model

model = onnx.load(MODEL_FILENAME)


#%% Run model

# Run the ONNX model with Caffe2
outputs = caffe2.python.onnx.backend.run_model(model, [imgExpanded])
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

for iClass,n in enumerate(topKClassnames):
    print('{:>6.2%}: {:>4} {}'.format(topKValues[iClass],topKIndices[iClass],n))
