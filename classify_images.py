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
import glob

# Directory to which you sync'd the repo.  Probably the same
# directory this file lives in, but for portability, this file is set up to only
# take dependencies on the repo according to this constant.
API_ROOT = r'/home/coyote/git/speciesclassification'
SUBDIRS_TO_IMPORT = ['DetectionClassificationAPI','FasterRCNNDetection','PyTorchClassification']    
   
# Path to taxa.csv, for latin --> common mapping
#
# Set to None to disable latin --> common mapping
TAXONOMY_PATH = r'/data/species_classification/taxa.19.08.28.0536.csv' # None

# IMAGES_TO_CLASSIFY can be:
#
# an array of filenames
#
# a single string; if it's a string, it's assumed to point to a .csv file, in 
# which each row is [filename,description]
#
# a directory, which is recursively enumerated
if False:
    IMAGES_TO_CLASSIFY = [
            '/data/species_classification/coyote.jpg',
            '/data/species_classification/meerkat.jpg',
            '/data/species_classification/elephant.jpg'
            ]
    IMAGES_TO_CLASSIFY_BASE = None

if False:
    IMAGES_TO_CLASSIFY = '/data/species_classification/animal_list.2018.10.23.12.58.16.csv'
    IMAGES_TO_CLASSIFY_BASE = '/data/species_classification/sample_animals'

if True:
    IMAGES_TO_CLASSIFY = '/data/species_classification/sample_animals'
    IMAGES_TO_CLASSIFY_BASE = None

# ...and classification results will be written here.

CLASSIFICATION_OUTPUT_FILE = None

if False:
    CLASSIFICATION_MODEL_PATH = '/data/species_classification/sc_all_extended_ensemble_resnext_inceptionV4_560_2019.09.19_model.pytorch'
    CLASSIFICATION_OUTPUT_FILE = '/data/species_classification/classifications_sc_all_extended_ensemble_resnext_inceptionV4_560_2019.09.19_model.csv'

if True:
    CLASSIFICATION_MODEL_PATH = '/data/species_classification/inc4-incres2-560-78.5.model_deploy.pth.tar'
    CLASSIFICATION_OUTPUT_FILE = '/data/species_classification/classifications_inc4-incres2-560-78.5.csv'
    
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

MAX_K_TO_PRINT = 3
DEBUG_MAX_IMAGES = -1


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


#%% Latin-->common lookup

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


#%% Prepare the list of images and query names

queries = None

if isinstance(IMAGES_TO_CLASSIFY,str) and os.path.isdir(IMAGES_TO_CLASSIFY):
    
    images = glob.glob(os.path.join(IMAGES_TO_CLASSIFY,'**/*.*'), recursive=True)
    images = [fn for fn in images if os.path.isfile(fn)]
    queries = [os.path.basename(os.path.dirname(fn)) for fn in images]
    print('Loaded a folder of {} images'.format(len(images)))    
    
elif isinstance(IMAGES_TO_CLASSIFY,str) and os.path.isfile(IMAGES_TO_CLASSIFY):
    
    print("Reading image list file")
    df_images = pd.read_csv(IMAGES_TO_CLASSIFY,header=None)
    df_images.columns = ['filename','query_string']
    nImages = len(images)    
    print("Read {} image names".format(len(images)))
    images = list(df_images.filename)
    queries = list(df_images.query_string)
    assert(len(queries) == len(images))
    
else:
    
    assert isinstance(IMAGES_TO_CLASSIFY,list)
    images = IMAGES_TO_CLASSIFY
    queries = None
    print('Processing list of {} images'.format(len(images)))
    

#%% Classify images

nErrors = 0
nImagesClassified = 0
nImages = len(images)

if CLASSIFICATION_OUTPUT_FILE is not None:
    f = open(CLASSIFICATION_OUTPUT_FILE,'w+')

# i_fn = 1; fn = images[i_fn]    
for i_fn,fn in enumerate(images):
    
    print("Processing image {} of {}".format(i_fn,nImages))
    fn = fn.replace('\\','/')
    query = ''
    if queries is not None:
        query = queries[i_fn]
        
    if IMAGES_TO_CLASSIFY_BASE is not None and len(IMAGES_TO_CLASSIFY_BASE > 0):
        fn = os.path.join(IMAGES_TO_CLASSIFY_BASE,fn)

    # with torch.no_grad():
    # print('Clasifying image {}'.format(fn))
    # def predict_image(self, image_path, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
    try:
        prediction = model.predict_image(fn, topK=min(5,MAX_K_TO_PRINT), multiCrop=False, 
                                             predict_mode=speciesapi.PredictMode.classifyOnly)
        nImagesClassified = nImagesClassified + 1
        
    except Exception as e:
        print("Error classifying image {} ({}): {}".format(i_fn,fn,str(e)))
        nErrors = nErrors + 1
        continue

    # i_prediction = 0
    for i_prediction in range(0, min(len(prediction.species),MAX_K_TO_PRINT)):
        latinName = prediction.species[i_prediction]
        likelihood = prediction.species_scores[i_prediction]
        likelihood = '{0:0.3f}'.format(likelihood)
        commonName = doLatinToCommon(latinName)
        s = '"{}","{}","{}","{}","{}","{}","{}"'.format(
                i_fn,fn,query,i_prediction,latinName,commonName,likelihood)
        if CLASSIFICATION_OUTPUT_FILE is not None:
            f.write(s + '\n')
        print(s)
        
    if DEBUG_MAX_IMAGES > 0 and i_fn >= DEBUG_MAX_IMAGES:
        break

# ...for each image
        
if CLASSIFICATION_OUTPUT_FILE is not None:
    f.close()
    
print("Finished classifying {} of {} images ({} errors)".format(nImagesClassified,nImages,nErrors))
