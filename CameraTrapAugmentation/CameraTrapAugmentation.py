#
# CameraTrapAugmentation.py
#
# Experiments around transforming handheld images to look like camera trap images.
#

import numpy as np
import skimage.io as io
import pylab
import cv2
import os
import os.path
import imutils
import scipy.ndimage
import time
import humanfriendly
import pickle

from data.coco_camera_traps_dataset import CocoCameraTrapsBboxDataset
from data.iwildcam_dataset import IWildCamBboxDataset
from multiprocessing.pool import ThreadPool 

import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')

import tf_detector
import matplotlib.pyplot as plt
import models
import rgb_nir_loader
import train

def showImage(img, bboxes=None, bboxesGT=None, label=None):

    plt.clf()
    plt.imshow(img)

    if (bboxes is not None and len(np.squeeze(bboxes))):

        for bbox in bboxes:

            x = [bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]]
            y = [bbox[0], bbox[0], bbox[2], bbox[2], bbox[0]]        

            plt.plot(x, y, 'r-')

    if (bboxesGT is not None and len(np.squeeze(bboxesGT))):

        for bbox in bboxesGT:

            x = [bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]]
            y = [bbox[0], bbox[0], bbox[2], bbox[2], bbox[0]]        

            plt.plot(x, y, 'g-')

    if (label):
        plt.title(label)

    plt.show(block=False)
    plt.pause(1)

    
def blurAnimals(img, bboxes):

    size = int(np.ceil(15*np.random.rand()))
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)

    kernel_motion_blur = imutils.rotate(kernel_motion_blur, np.random.rand()*180)
    kernel_motion_blur = np.clip(kernel_motion_blur, 0, 1)

    kernel_motion_blur = kernel_motion_blur / cv2.sumElems(kernel_motion_blur)[0]
    animalBlurred = cv2.filter2D(img, -1, kernel_motion_blur)

    #for bbox in bboxes:
    #    img[bbox[0]:bbox[2],bbox[1]:bbox[3],:] = animalBlurred[bbox[0]:bbox[2],bbox[1]:bbox[3],:]

    maskAll = np.zeros(img.shape)

    for bbox in bboxes:
        mask = np.zeros(img.shape)
        if (len(img.shape) == 3):
            mask[int(round((bbox[0]+bbox[2])/2)),int(round((bbox[1]+bbox[3])/2)),:] = 1
        else:
            mask[int(round((bbox[0]+bbox[2])/2)),int(round((bbox[1]+bbox[3])/2))] = 1

        bh = bbox[2]-bbox[0]
        bw = bbox[3]-bbox[1]
        #rad = 2*int(round(round(max(bh,bw)*1.5)/2)) + 1

        radw = 2*int(round(round(bw*1.5)/2)) + 1
        radh = 2*int(round(round(bh*1.5)/2)) + 1

        if (len(img.shape) == 3):
            mask = scipy.ndimage.filters.gaussian_filter(mask, (radh/4, radw/4, 0))
        else:
            mask = scipy.ndimage.filters.gaussian_filter(mask, (radh/4, radw/4))

        #mask = cv2.GaussianBlur(mask,(rad,rad),rad/6)
        mask = mask/np.amax(mask)

        maskAll = cv2.max(maskAll, mask)

    #showImage(maskAll, bboxes)

    img = img*(1-maskAll) + animalBlurred*maskAll

    return img

def makeDayImage(img, bboxes=None, blurAnimal=True):

    a = .85
    b = .15

    img = a*(img) + b

    img = cv2.GaussianBlur(img,(5,5),.5)

    if (bboxes is not None and len(bboxes) and blurAnimal):

        img = blurAnimals(img, bboxes)

    # noise
    noise = .01*np.random.randn(img.shape[0], img.shape[1], img.shape[2])
    vnoise = np.tile(.005*np.random.randn(1, img.shape[1], img.shape[2]), [img.shape[0], 1, 1])

    img = img + noise + vnoise

    img = np.clip(img, 0, 1)

    return img

def makeNightImage(img, bboxes=None, blurAnimal=True, img_file=None):

    if(img_file):
        img = model.predict_image(img_file)
    else:
        img = np.mean(img, 2)

    #a = .8
    #b = -.2

    a = .6
    b = .2

    img = a*(img) + b

    #img = cv2.GaussianBlur(img,(11,11),1)
    img = cv2.GaussianBlur(img,(5,5),.5)

    if (bboxes is not None and len(bboxes) and blurAnimal):

        img = blurAnimals(img, bboxes)

    # falloff
    x = np.linspace(-img.shape[1]/2, img.shape[1]/2, img.shape[1])
    y = np.linspace(-img.shape[0]/2, img.shape[0]/2, img.shape[0])
    xv, yv = np.meshgrid(x, y)

    maxR = np.sqrt((img.shape[0]/2)**2 + (img.shape[1]/2)**2)

    r = np.sqrt(xv**2 + yv**2)/maxR
    #falloff = (1 - r)**2
    falloff = np.clip(0.7607*r**2 -1.696*r + 1.2517, 0, 1)

    img = img*falloff

    # noise
    noise = .0134*np.random.randn(img.shape[0], img.shape[1])
    vnoise = np.tile(.008*np.random.randn(1, img.shape[1]), [img.shape[0], 1])

    img = img + noise + vnoise

    img = np.clip(img, 0, 1)

    img = np.tile(np.expand_dims(img,2), [1,1,3])   

    return img

def transformImage(iNat2017Data, ind, createNight, dirOut, displayImage = False):

    print(ind)

    img, bboxes, labels, difficulties, image_id, image_file = iNat2017Data.get_example(ind)        

    image_file_out = image_file.replace('/images/', dirOut)

    if (createNight):
        img = makeNightImage(img, bboxes, img_file=image_file)
    else:
        img = makeDayImage(img, bboxes)

    if (displayImage):
        showImage(img)
    else:
        paths = os.path.split(image_file_out)

        if not os.path.exists(paths[0]):
            os.makedirs(paths[0])

        cv2.imwrite(image_file_out, img[:,:,(2, 1, 0)]*255)

def iNatToCameraTrap(dataDir, annFile, dirOut, createNight = True, saveImages=False):

    displayImage = not saveImages
    multiThread = saveImages
    nThreads = 20
    
    iNat2017Data = INatBboxDataset(dataDir,annFile,loadBboxes=True)   

    filesInd, _ = iNat2017Data.find_files('Mammalia')

    print("found " + str(len(filesInd)) + " files")

    global model
    model_path = '../IRtoRGB/model_best.pth_kaist_256.tar'    
    args = train.Params()
    model = models.IRtoRGB(model_path, (args.sz, args.sz), args.num_layers, args.feats_in, args.feats_out, args.feats_max, useGPU=True)

    if (multiThread):
        pool = ThreadPool(nThreads)
        results = pool.map(lambda x: transformImage(iNat2017Data, x, createNight, dirOut, displayImage), filesInd)
    else:
        for i in range(0, len(filesInd)):
            transformImage(iNat2017Data, filesInd[i], createNight, dirOut, displayImage)

def iWildCamClasifyDayNight(iWildCamData):

    n = iWildCamData.__len__()

    isIRs = []
    maxVals = []

    for i in range(0, n):
        print(i)
        img, bboxes, labels, difficulties, image_id, _ = iWildCamData.get_example(i)

        m = np.mean(img, axis=(0,1))
        s = np.std(m)
        isIR = (s < 1e-4)

        maxVal = 0

        if (bboxes is not None and len(np.squeeze(bboxes))):

            for bbox in bboxes:

                animal = img[int(bbox[0]):int(bbox[2]),int(bbox[1]):int(bbox[3]),:]

                #minVal = np.min(animal)
                maxVal = np.max(animal)

                #showImage(animal, label=str(i) + 'min: ' + str(minVal)+ 'max: ' + str(maxVal))

        isIRs.append(isIR)
        maxVals.append(maxVal)

        #showImage(img, bboxesGT=bboxes, label=str(i) + ' ' + str(isIR))

    return isIRs, maxVals

def detect(MODEL_FILE, dir, iNat2017Data):

    filesInd, files = iNat2017Data.find_files('Mammalia')
    print("found " + str(len(filesInd)) + " files")

    # Load and run detector on target images
    detection_graph = tf_detector.load_model(MODEL_FILE)

    startTime = time.time()
    boxes,scores,classes,images = tf_detector.generate_detections(detection_graph, dir, files)
    elapsed = time.time() - startTime
    print("Done running detector on {} files in {}".format(len(images),humanfriendly.format_timespan(elapsed)))

    return boxes,scores

def detectAndShow(dir, iNat2017Data, boxes, scores):

    filesInd, files = iNat2017Data.find_files('Mammalia')
    print("found " + str(len(filesInd)) + " files")

    confidenceThreshold = 0.9

    for j in range(len(boxes)):

        img, bboxes, labels, difficulties, image_id, image_file = iNat2017Data.get_example(filesInd[j]) 

        bb = boxes[j][scores[j]>=confidenceThreshold]

        s = img.shape; imageHeight = s[0]; imageWidth = s[1]

        # top, left, bottom, right 
        # x,y origin is the upper-left
        bb[:,0] = bb[:,0] * imageHeight
        bb[:,1] = bb[:,1] * imageWidth
        bb[:,2] = bb[:,2] * imageHeight
        bb[:,3] = bb[:,3] * imageWidth
            
        showImage(img, bb.tolist(), bboxes)

def intersectionOverUnion(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[1], boxB[1])
	yA = max(boxA[0], boxB[0])
	xB = min(boxA[3], boxB[3])
	yB = min(boxA[2], boxB[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def intersectBoxes(bboxes, bboxesGT):

    iouMax = 0
    boxMax = None

    for bboxGT in bboxesGT:
        for bbox in bboxes:
            iou = intersectionOverUnion(bbox, bboxGT)
            iouMax = max(iouMax, iou)
            boxMax = bbox

    return iouMax, boxMax

def compareDetections(dir, iNat2017Data, boxes, scores, showImages=False):

    filesInd, files = iNat2017Data.find_files('Mammalia')
    print("found " + str(len(filesInd)) + " files")

    confidenceThreshold = 0.9

    intersections = []

    for j in range(len(boxes)):
        print(j)
        img, bboxes, labels, difficulties, image_id, image_file = iNat2017Data.get_example(filesInd[j]) 

        bb = boxes[j][scores[j]>=confidenceThreshold]

        s = img.shape; imageHeight = s[0]; imageWidth = s[1]

        # top, left, bottom, right 
        # x,y origin is the upper-left
        bb[:,0] = bb[:,0] * imageHeight
        bb[:,1] = bb[:,1] * imageWidth
        bb[:,2] = bb[:,2] * imageHeight
        bb[:,3] = bb[:,3] * imageWidth
            
        iouPred = 0
        boxPred = []

        if (len(np.squeeze(bb)) and len(np.squeeze(bboxes))):
            iouPred, boxPred = intersectBoxes(bb.tolist(), bboxes)

        bboxesN = len(bb)
        bboxesGTN = len(np.squeeze(bboxes))

        intersections.append([iouPred, bboxesN, bboxesGTN])

        if(showImages):
            showImage(img, [boxPred] if boxPred is not None else [], bboxes, str(j) + ' ' + str(iouPred))

    return intersections

def mainSynth():

    # iNat
    dataRoot = 'E:/Research/Images/iNat2017/'
    dataDir = dataRoot + 'images/'
    annFile = dataRoot + 'val_2017_bboxes.json'
    dirOut = '/imagesCTNight2/'
    createNight = True
    saveImages = True

    iNatToCameraTrap(dataDir, annFile, dirOut, createNight, saveImages)

def mainDetect():

    MODEL_FILE = r'.\train_on_ss_and_inat\frozen_inference_graph.pb'
    
    iouThresh = .5

    # iNat
    dataRoot = 'E:/Research/Images/iNat2017/'
    annFile = dataRoot + 'val_2017_bboxes.json'
    isiNat = True

    dataDir = dataRoot + 'imagesCTNight2/'        
    filePrefix = 'imagesCTNight2_train_on_ss_and_inat'
    
    ## iWildCam
    #dataRoot = 'E:/Research/Images/CaltechCameraTraps/'
    #annFile = dataRoot + 'eccv_18_annotation_files/trans_val_annotations.json'
    #dataDir = dataRoot + 'eccv_18_images_only/'
    #filePrefix = 'eccv_18_trans_val_train_on_ss'
    #isiNat = False

    # Run
    if (isiNat):
        data = INatBboxDataset(dataDir, annFile, loadBboxes = True)   
    else:
        data = IWildCamBboxDataset(dataDir, annFile)   

    predFile = filePrefix + '_predictions.dat'
    intersectionsFile = filePrefix + '_intersections.dat'

    if(not os.path.exists(predFile)) :

        boxes, scores = detect(MODEL_FILE, dataDir, data)

        with open(predFile, 'wb') as fp:
            pickle.dump(boxes, fp)
            pickle.dump(scores, fp)
    else:
        with open (predFile, 'rb') as fp:
            boxes = pickle.load(fp)
            scores = pickle.load(fp)

    if(not os.path.exists(intersectionsFile)) :
        intersections = compareDetections(dataDir, data, boxes, scores, showImages=False)

        with open(intersectionsFile, 'wb') as fp:
            pickle.dump(intersections, fp)
    else:
        with open (intersectionsFile, 'rb') as fp:
            intersections = pickle.load(fp)

    intersections = np.asarray(intersections)

    if (not isiNat):
        nightdayFile = filePrefix + '_nightday.dat'

        if(not os.path.exists(nightdayFile)) :
            isNight, maxVals = iWildCamClasifyDayNight(data)    

            with open(nightdayFile, 'wb') as fp:
                pickle.dump(isNight, fp)
                pickle.dump(maxVals, fp)

        else:
            with open (nightdayFile, 'rb') as fp:
                isNight = pickle.load(fp)
                maxVals = pickle.load(fp)

        maxVals = np.asarray(maxVals)
    
        brightThresh = .95

        tp = np.sum(np.logical_and(np.logical_and(intersections[:,0]>=iouThresh, intersections[:,2]>0), np.logical_not(isNight)))
        fp = np.sum(np.logical_and(np.logical_and(intersections[:,1]>0, intersections[:,2]==0), np.logical_not(isNight)))
        fn = np.sum(np.logical_and(np.logical_and(intersections[:,0]<iouThresh, intersections[:,2]>1), np.logical_not(isNight)))

        print('day percent precision ' + str((tp/(tp+fp))*100))
        print('day percent recall ' + str((tp/(tp+fn))*100))

        tp = np.sum(np.logical_and(np.logical_and(intersections[:,0]>=iouThresh, intersections[:,2]>0), isNight))
        fp = np.sum(np.logical_and(np.logical_and(intersections[:,1]>0, intersections[:,2]==0), isNight))
        fn = np.sum(np.logical_and(np.logical_and(intersections[:,0]<iouThresh, intersections[:,2]>1), isNight))

        print('night percent precision ' + str((tp/(tp+fp))*100))
        print('night percent recall ' + str((tp/(tp+fn))*100))

        # animals eyes
        tp = np.sum(np.logical_and(np.logical_and(np.logical_and(intersections[:,0]>=iouThresh, intersections[:,2]>0), isNight), maxVals>=brightThresh))
        fp = np.sum(np.logical_and(np.logical_and(np.logical_and(intersections[:,1]>0, intersections[:,2]==0), isNight), maxVals>=brightThresh))
        fn = np.sum(np.logical_and(np.logical_and(np.logical_and(intersections[:,0]<iouThresh, intersections[:,2]>1), isNight), maxVals>=brightThresh))

        n = np.sum(np.logical_and(isNight, maxVals>=brightThresh))

        print('night bright percent precision ' + str((tp/(tp+fp))*100))
        print('night bright percent recall ' + str((tp/(tp+fn))*100))
        print('n ' + str(n))

        tp = np.sum(np.logical_and(np.logical_and(np.logical_and(intersections[:,0]>=iouThresh, intersections[:,2]>0), isNight), maxVals<brightThresh))
        fp = np.sum(np.logical_and(np.logical_and(np.logical_and(intersections[:,1]>0, intersections[:,2]==0), isNight), maxVals<brightThresh))
        fn = np.sum(np.logical_and(np.logical_and(np.logical_and(intersections[:,0]<iouThresh, intersections[:,2]>1), isNight), maxVals<brightThresh))

        n = np.sum(np.logical_and(isNight, maxVals<brightThresh))

        print('night dark percent precision ' + str((tp/(tp+fp))*100))
        print('night dark percent recall ' + str((tp/(tp+fn))*100))        
        print('n ' + str(n))

    tp = np.sum(np.logical_and(intersections[:,0]>=iouThresh, intersections[:,2]>0))
    fp = np.sum(np.logical_and(intersections[:,1]>0, intersections[:,2]==0))
    fn = np.sum(np.logical_and(intersections[:,0]<iouThresh, intersections[:,2]>1))

    print('percent precision ' + str((tp/(tp+fp))*100))
    print('percent recall ' + str((tp/(tp+fn))*100))

    #detectAndShow(dataDir, iNat2017Data, boxes, scores)

def mainAvgCameraTraps():

    # iWildCam
    dataRoot = 'E:/Research/Images/CaltechCameraTraps/'
    annFile = dataRoot + 'eccv_18_annotation_files/cis_val_annotations.json'
    dataDir = dataRoot + 'eccv_18_images_only/'
    filePrefix = 'eccv_18_cis_val'

    data = IWildCamBboxDataset(dataDir, annFile)   

    nightdayFile = filePrefix + '_nightday.dat'

    if(not os.path.exists(nightdayFile)) :
        isNight = iWildCamClasifyDayNight(data)    

        with open(nightdayFile, 'wb') as fp:
            pickle.dump(isNight, fp)

    else:
        with open (nightdayFile, 'rb') as fp:
            isNight = pickle.load(fp)

    filesInd, files = data.find_files('Mammalia')
    print("found " + str(len(filesInd)) + " files")

    filesInd = np.asarray(filesInd)[isNight]

    avgImg = np.zeros((1494,2048,3))
    avgN = 0
    for j in range(len(filesInd)):
        print(j)
        img, bboxes, labels, difficulties, image_id, image_file = data.get_example(filesInd[j]) 

        minVal = np.percentile(img[50:,:,:],1)
        maxVal = np.percentile(img[50:,:,:],99)

        img = np.clip((img-minVal)/(maxVal-minVal), 0, 1)

        try:
            avgImg = avgImg + img
            avgN = avgN + 1
            #showImage(avgImg/avgN)
        except:
            continue

    avgImg = avgImg/avgN

    cv2.imwrite('avgNight.png', 255*avgImg)

def mainDisplayNightResults():

    dataRoot = ''
    annFile = dataRoot + 'annotations.json'
    dataDir = dataRoot + 'images'
    filePrefix = 'cameratraps'

    data = IWildCamBboxDataset(dataDir, annFile)   

    nightdayFile = filePrefix + '_nightday.dat'

    if(not os.path.exists(nightdayFile)) :
        isNight = iWildCamClasifyDayNight(data)    

        with open(nightdayFile, 'wb') as fp:
            pickle.dump(isNight, fp)

    else:
        with open (nightdayFile, 'rb') as fp:
            isNight = pickle.load(fp)

    filesInd, files = data.find_files('Mammalia')
    print("found " + str(len(filesInd)) + " files")

    filesInd = np.asarray(filesInd)[isNight]

    predFile = filePrefix + '_predictions.dat'
    intersectionsFile = filePrefix + '_intersections.dat'

    if(not os.path.exists(predFile)) :

        boxes, scores = detect(MODEL_FILE, dataDir, data)

        with open(predFile, 'wb') as fp:
            pickle.dump(boxes, fp)
            pickle.dump(scores, fp)
    else:
        with open (predFile, 'rb') as fp:
            boxes = pickle.load(fp)
            scores = pickle.load(fp)

    boxes = np.asarray(boxes)[isNight]
    scores = np.asarray(scores)[isNight]

    confidenceThreshold = 0.9

    for j in range(len(filesInd)):
        print(j)
        img, bboxes, labels, difficulties, image_id, image_file = data.get_example(filesInd[j]) 

        bb = boxes[j][scores[j]>=confidenceThreshold]

        s = img.shape; imageHeight = s[0]; imageWidth = s[1]

        # top, left, bottom, right 
        # x,y origin is the upper-left
        bb[:,0] = bb[:,0] * imageHeight
        bb[:,1] = bb[:,1] * imageWidth
        bb[:,2] = bb[:,2] * imageHeight
        bb[:,3] = bb[:,3] * imageWidth
            
        iouPred = 0
        boxPred = []

        if (len(np.squeeze(bb)) and len(np.squeeze(bboxes))):
            iouPred, boxPred = intersectBoxes(bb.tolist(), bboxes)

        showImage(img, [boxPred] if boxPred is not None else [], bboxes, str(j) + ' ' + str(iouPred))

        
if __name__ == '__main__':    

    #mainAvgCameraTraps()
    #mainSynth()
    mainDetect()
    #mainDisplayNightResults()
    