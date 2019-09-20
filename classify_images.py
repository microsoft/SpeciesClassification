#######
#
# classify_images.py
#
# This is a test driver for running our species classifiers and detectors.
# The script classifies one or more hard-coded image files.
#
# Because the inference code has not been assembled into a formal package yet,
# you should define API_ROOT to point to the base of our repo.  This
# will be added to your Python path later in the script.
#
# This script has two non-code dependencies:
#
# * a classification model file (and, optionally, a detection model model)
# * a taxonomy file, so the scientific names used in the training data can
#   be mapped to common names.
#
####### 


#%% Constants and imports

import sys
import os
import pandas as pd

# Directory to which you sync'd the repo.  Probably the same
# directory this file lives in, but for portability, this file is set up to only
# take dependencies on the repo according to this constant.
API_ROOT = r'/home/coyote/git/speciesclassification'
SUBDIRS_TO_IMPORT = ['DetectionClassificationAPI','FasterRCNNDetection','PyTorchClassification']    
   
# Path to taxa.csv, for latin --> common mapping
#
# Set to None to disable latin --> common mapping
TAXONOMY_PATH = r'/data/species_classification/taxa.19.08.28.0536.csv' # None

# IMAGES_TO_CLASSIFY can be either an array of files or a single string; if
# it's a string, it's assumed to point to a .csv file, in which each row is
# [filename,description]
IMAGES_TO_CLASSIFY = [
        '/data/species_classification/coyote.jpg',
        '/data/species_classification/meerkat.jpg',
        '/data/species_classification/elephant.jpg'
        ]

IMAGES_TO_CLASSIFY = '/data/species_classification/animal_list.2018.10.23.12.58.16.csv'

# When IMAGES_TO_CLASSIFY points to a .csv file, this will be prepended to all
# filenames...
IMAGES_TO_CLASSIFY_BASE = '/data/species_classification/sample_animals'

# ...and classification results will be written here.
CLASSIFICATION_OUTPUT_FILE = '/data/species_classification/classifications.csv'

# The main .pytorch file
CLASSIFICATION_MODEL_PATH = '/data/species_classification/sc_all_extended_ensemble_resnext_inceptionV4_560_2019.09.19_model.pytorch'

# Detection (i.e., bounding box generation) is optional; set to None 
# to disable detection
DETECTION_MODEL_PATH = None

# This must be True if detection is enabled.  Classification can be run
# on the CPU or GPU.
USE_GPU = True

# List of image sizes to use, one per model in the ensemble.  Images will be resized 
# and reshaped to square images prior to classification.  
#
# We typically specify [560,560] if we're loading our Inception/InceptionResnet 
# ensemble. For ResNext, we typically specify [448].
#
IMAGE_SIZES = [560, 560]
# IMAGE_SIZES = [448]


#%% Path setup to import the classification code

if (not API_ROOT.lower() in map(str.lower,sys.path)):
    
    print("Adding {} to the python path".format(API_ROOT))
    sys.path.insert(0,API_ROOT)

for s in SUBDIRS_TO_IMPORT:
    if (not s.lower() in map(str.lower,sys.path)):
        importPath = os.path.join(API_ROOT,s)
        print("Adding {} to the python path".format(importPath))
        sys.path.insert(0,importPath)    


#%% Import classification modules

import api as speciesapi


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
model = speciesapi.DetectionClassificationAPI(CLASSIFICATION_MODEL_PATH, DETECTION_MODEL_PATH, IMAGE_SIZES, USE_GPU)
print("Finished loading model")


#%% Classify a hard-coded list of images

if isinstance(IMAGES_TO_CLASSIFY,list):
    
    nImages = len(IMAGES_TO_CLASSIFY)
    
    for iImage,imageFileName in enumerate(IMAGES_TO_CLASSIFY):
        
        print("Processing image {} of {}".format(iImage,nImages))
        
        # def predict_image(self, image_path, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
        try:
            prediction = model.predict_image(imageFileName, topK=5, multiCrop=False, 
                                                 predict_mode=speciesapi.PredictMode.classifyOnly)
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



#%% Classify a filed-specified list of images

else:
    
    #%%
    
    print("Reading image list file")
    
    # Read the image list file
    imageList = pd.read_csv(IMAGES_TO_CLASSIFY)
    nImages = len(imageList)
    
    print("Read {} image names".format(nImages))
    
    maxImages = -1
    
    nErrors = 0
    
    nImagesClassified = 0
    
    with open(CLASSIFICATION_OUTPUT_FILE,'w+') as f:
        
        for index, row in imageList.iterrows():
            
            print("Processing image {} of {}".format(index,nImages))
            filename = row[0].replace('\\','/')
            keyword = row[1]
            
            imagePath = os.path.join(IMAGES_TO_CLASSIFY_BASE,filename)
        
            # with torch.no_grad():
            # print('Clasifying image {}'.format(imagePath))
            # def predict_image(self, image_path, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
            try:
                prediction = model.predict_image(imagePath, topK=5, multiCrop=False, 
                                                     predict_mode=speciesapi.PredictMode.classifyOnly)
                nImagesClassified = nImagesClassified + 1
                
            except Exception as e:
                print("Error classifying image {} ({}): {}".format(index,imagePath,str(e)))
                nErrors = nErrors + 1
                continue
    
            # print('Classified image {}'.format(imagePath))
        
            # print('\nSpecies \n')
        
            for i in range(0, len(prediction.species)):
                latinName = prediction.species[i]
                likelihood = prediction.species_scores[i]
                commonName = doLatinToCommon(latinName)
                # print('%d) %s (%s) likelihood: %f' % (i+1, latinName, commonName, likelihood))
                f.write('"{}","{}","{}","{}","{}","{}","{}"\n'.format(
                        index,filename,keyword,i,latinName,commonName,likelihood))
                print('"{}","{}","{}","{}","{}","{}","{}"\n'.format(
                        index,filename,keyword,i,latinName,commonName,likelihood))
                
            if maxImages > 0 and index >= maxImages:
                break
    
    print("Finished classifying {} of {} images ({} errors)".format(nImagesClassified,nImages,nErrors))
    
    
