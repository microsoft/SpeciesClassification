#
# classify_night_day.py
#
# Simple heuristics for determining whether a camera trap image is 
# a day- or night-time image.
#

import os
import numpy as np
from PIL import Image
import json

from data.iwildcam_dataset import IWildCamBboxDataset

def clasifyDayNightImage(img):

    m = np.mean(img, axis=(0,1)) # take the mean over the 2d image
    s = np.std(m) # standard dev across RGB
    isIR = (s < 1e-4) # if it is low, it's a grayscale image

    return isIR

    
def readImage(path):

    f = Image.open(path)
    img = f.convert('RGB')
    f.close()
    img = np.asarray(img, dtype=np.float32)            
    img = img/255

    return img

    
def clasifyDayNight(imageDir, files):

    isIRs = {}

    for i in range(0, len(files)):
        img_file = os.path.join(imageDir, files[i])

        img = readImage(img_file)

        isIR = clasifyDayNightImage(img)

        isIRs[img_file] = int(isIR)

        print('%d:\t%d' % (i, isIR))

    return isIRs

if __name__ == '__main__':    
    
    dataRoot = ''
    annFile = dataRoot + ''
    dataDir = dataRoot + ''
    nightdayFile = ''

    data = IWildCamBboxDataset(dataDir, annFile)   
    files = data.get_files()

    # pass in list of files
    isNight = clasifyDayNight(dataDir, files)    

    with open(nightdayFile, 'w') as fp:
        json.dump(isNight, fp)
        