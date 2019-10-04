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

import os
import cv2
import onnx
import caffe2.python.onnx.backend
import numpy as np
import codecs
from operator import itemgetter
import argparse 


TOP_K = 5

parser = argparse.ArgumentParser(description='Given a pointer to a frozen detection graph, runs inference on a single image and prints results.')
parser.add_argument('--frozen_graph', type=str,
                    help='Frozen graph of detection network as created by the export_inference_graph_definition.py ' + \
                                        ' freeze_graph.py script. The script assumes that the model already includes all necessary pre-processing.',
                                                           metavar='PATH_TO_CLASSIFIER_W_PREPROCESSING')
parser.add_argument('--classlist', type=str,
                    help='Path to text file containing the names of all possible categories.')
parser.add_argument('--image_path', type=str,
                    help='Path to image file.')
parser.add_argument('--image_list', type=str,
                    help='Path to image list.')
args = parser.parse_args()

IMAGE_FILENAME = args.image_path # '/home/meerkat/git/SpeciesClassification/PyTorchClassification/elephant.jpg'
MODEL_FILENAME = args.frozen_graph # '/ai4edevfs/models/iNat/iNat_all_extended/demosite-model-ensemble-resnext-inceptionV4-560-81.0/iNat_all_extended_ensemble_resnext_inceptionV4_560_81.9_model.onnx'
CLASSLIST_FILENAME = args.classlist # '/ai4edevfs/models/iNat/iNat_all_extended/demosite-model-ensemble-resnext-inceptionV4-560-81.0/iNat_all_extended_ensemble_resnext_inceptionV4_560_81.9_classes.txt'

# Target mean / std; should match the values used at training time
MODEL_IMAGE_SIZE = 560
MODEL_RESIZE_SIZE = 640
OVERSIZE_FACTOR = 1.3

model = onnx.load(MODEL_FILENAME)

if args.image_list:
    with open(args.image_list, 'rt') as fi:
        all_images = fi.read().splitlines()
else:
    all_images = [IMAGE_FILENAME]

for IMAGE_FILENAME in all_images:

    if not os.path.isfile(IMAGE_FILENAME):
        print("Did not find " + IMAGE_FILENAME)
        continue

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
