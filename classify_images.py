#######
#
# classify_images.py
#
# This is a test driver for running our classifiers and detectors trained on
# iNaturalist data.  The script classifies one or more hard-coded image files.
#
# Because the inference code has not been assembled into a formal package yet,
# you should define INAT_API_ROOT to point to the base of our iNat repo.  This
# will be added to your Python path later in the script.
#
# This script has two non-code dependencies:
#
# * a classification model file (and, optionally, a detection model model)
# * a taxonomy file, so the scientific names used in the training data can
#   be mapped to common names.
#
# We are currently testing against PyTorch 0.4.1 and Cuda 9.0, and we have tested on 
# both Linux and Windows.
#
####### 


#%% Constants and imports

import sys
import os
import pandas as pd

# Directory to which you sync'd the iNat repo.  Probably the same
# directory this file lives in, but for portability, this file is set up to only
# take dependencies on the iNat repo according to this constant.
INAT_API_ROOT = r'd:\git\ai4edev\iNat'

# Path to taxa.csv, for latin --> common mapping
#
# Set to None to disable latin --> common mapping
TAXONOMY_PATH = r'd:\temp\taxa.csv' # None

IMAGES_TO_CLASSIFY = [
        r"D:\temp\animals\African_Elephant\30651.ngsversion.1421960098780.jpg",
        r"D:\temp\animals\Alligator\Alligator_mississippiensis_01.JPG"
        ]

# CLASSIFICATION_MODEL_PATH = r'd:\temp\models\inc4-incres2-560-78.5\model_deploy.pth.tar'
CLASSIFICATION_MODEL_PATH = r"D:\temp\models\resnext-448-78.8\model_best.pth.tar"

# Detection (i.e., bounding box generation) is optional; set to None 
# to disable detection
DETECTION_MODEL_PATH = None

SUBDIRS_TO_IMPORT = ['iNatAPI','iNatEnsemble','iNatFasterRCNN']    
   
# This must be True if detection is enabled.  Classification can be run
# on the CPU or GPU.
USE_GPU = True

# List of image sizes to use, one per model in the ensemble.  Images will be resized 
# and reshaped to square images prior to classification.  
#
# We typically specify [560,560] if we're loading our Inception/InceptionResnet 
# ensemble. For ResNext, we typically specify [448].
#
# IMAGE_SIZES = [560, 560]
IMAGE_SIZES = [448]


#%% Path setup to import the classification code

if (not INAT_API_ROOT.lower() in map(str.lower,sys.path)):
    
    print("Adding {} to the python path".format(INAT_API_ROOT))
    sys.path.insert(0,INAT_API_ROOT)
    for s in SUBDIRS_TO_IMPORT:
        importPath = os.path.join(INAT_API_ROOT,s)
        print("Adding {} to the python path".format(INAT_API_ROOT))    
        sys.path.insert(0,importPath)    


#%% Import classification modules

import inatapi


#%% Build Latin --> common mapping

latinToCommon = {}

if TAXONOMY_PATH != None:
        
    print("Reading taxonomy file")
    
    # Read taxonomy file; takes ~1 minute
    df = pd.read_csv(TAXONOMY_PATH)
    df = df.fillna('')
    
    # Columns are:
    #
    # taxonID,scientificName,parentNameUsageID,taxonRank,vernacularName,wikipedia_url
    
    # Create dictionary by ID
    
    nRows = df.shape[0]
    
    for index, row in df.iterrows():
    
        latinName = row['scientificName']
        latinName = latinName.strip()
        if len(latinName)==0:
            print("Warning: invalid scientific name at {}".format(index))
            latinName = 'unknown'
        commonName = row['vernacularName']
        commonName = commonName.strip()
        latinName = latinName.lower()
        commonName = commonName.lower()
        latinToCommon[latinName] = commonName
    
    print("Finished reading taxonomy file")


#%% Define Latin-->common lookup

def doLatinToCommon(latinName):

    if len(latinToCommon) == 0:
        return latinName
    
    latinName = latinName.lower()
    if not latinName in latinToCommon:
        print("Warning: latin name {} not in lookup table".format(latinName))
        commonName = latinName
    else:
        commonName = latinToCommon[latinName]
        commonName = commonName.strip()
        
    if (len(commonName) == 0):
        print("Warning: empty result for latin name {}".format(latinName))
        commonName = latinName

    return commonName


#%% Create the model(s)

assert os.path.isfile(CLASSIFICATION_MODEL_PATH)
if DETECTION_MODEL_PATH != None:
    assert os.path.isfile(DETECTION_MODEL_PATH)

print("Loading model")
model = inatapi.iNatAPI(CLASSIFICATION_MODEL_PATH, DETECTION_MODEL_PATH, IMAGE_SIZES, USE_GPU)
print("Finished loading model")


#%% Classify images

nImages = len(IMAGES_TO_CLASSIFY)

for iImage,imageFileName in enumerate(IMAGES_TO_CLASSIFY):
    
    print("Processing image {} of {}".format(iImage,nImages))
    
    # def predict_image(self, image_path, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
    try:
        prediction = model.predict_image(imageFileName, topK=5, multiCrop=False, 
                                             predict_mode=inatapi.PredictMode.classifyOnly)
    except Exception as e:
        print("Error classifying image {} ({}): {}".format(iImage,imageFileName,str(e)))
        continue

    fn = os.path.splitext(imageFileName)[0]
    
    for i in range(0, len(prediction.species)):
        latinName = prediction.species[i]
        likelihood = prediction.species_scores[i]
        commonName = doLatinToCommon(latinName)
        print('"{}","{}","{}","{}","{}","{}"\n'.format(
                iImage,fn,i,latinName,commonName,likelihood))
        
print("Finished classifying {} images".format(nImages))


