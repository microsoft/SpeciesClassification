#######
#
# classify_images.py
#
# This is a test driver for running our species classifiers and detectors.
# The script classifies one or more hard-coded image files.
#
# Because the inference code has not been assembled into a formal package yet,
# you should define api_root to point to the base of our repo.  This
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
api_root = r'/home/coyote/git/speciesclassification'
subdirs_to_import = ['DetectionClassificationAPI','FasterRCNNDetection','PyTorchClassification']    
   
# Path to taxa.csv, for latin --> common mapping
#
# Set to None to disable latin --> common mapping
taxonomy_path = r'/data/species_classification/taxa.19.08.28.0536.csv' # None

job_name = ''
images_to_classify_base = None

# images_to_classify can be:
#
# an array of filenames
#
# a single string; if it's a string, it's assumed to point to a .csv file, in 
# which each row is [filename,description]
#
# a directory, which is recursively enumerated
if False:
    images_to_classify = [
            '/data/species_classification/coyote.jpg',
            '/data/species_classification/meerkat.jpg',
            '/data/species_classification/elephant.jpg'
            ]

# Pick images from a .csv file
if False:
    images_to_classify = '/data/species_classification/animal_list.2018.10.23.12.58.16.csv'
    images_to_classify_base = '/data/species_classification/sample_animals'
    
# Pick images from a folder
if True:
    images_to_classify = '/data/species_classification/images/sample_images.2019.12.28'
    job_name = 'sample_images.2019.12.28'
    
# Pick images from a folder
if False:
    images_to_classify = '/data/species_classification/elephants_and_hippos'
    
# Classification results will be written here
classification_output_file = None

model_base = '/data/species_classification/models'

if True:
    # 2019 fixed model
    classification_model_path = os.path.join(model_base,
                                             'iNat_all_extended/demosite-model-ensemble-resnext-inceptionV4-560-83.1/iNat_all_extended_ensemble_resnext_inceptionV4_560_83.1_model.2019.12.00.pytorch')

if False:
    # 2019 broken model
    classification_model_path = os.path.join(model_base,
                                             'iNat_all_extended_buggy/demosite-model-ensemble-resnext-inceptionV4-560-81.0/iNat_all_extended_ensemble_resnext_inceptionV4_560_81.9_model.2019.10.00.pytorch')

if False:
    # 2018 model    
    classification_model_path = os.path.join(model_base,
                                             'iNat_original/inc4-incres2-560-78.5/inc4-incres2-560-78.5.model_deploy.pth.tar')

assert(os.path.isfile(classification_model_path))

output_base = '/data/species_classification/output'
model_name = os.path.basename(classification_model_path)
classification_output_file = os.path.join(output_base,'classifications_{}_{}.csv'.format(job_name,model_name))
    
# Detection (i.e., bounding box generation) is optional; set to None 
# to disable detection
detection_model_path = None

# This must be True if detection is enabled.  Classification can be run
# on the CPU or GPU.
use_gpu = True

# List of image sizes to use, one per model in the ensemble.  Images will be resized 
# and reshaped to square images prior to classification.  
#
# We typically specify [560,560] if we're loading our Inception/InceptionResnet 
# ensemble. For ResNext, we typically specify [448].
#
image_sizes = [560, 560]
# image_sizes = [448]

mak_k_to_print = 3
debug_max_images = -1


#%% Path setup to import the classification code

if (not api_root.lower() in map(str.lower,sys.path)):
    
    print("Adding {} to the python path".format(api_root))
    sys.path.insert(0,api_root)

for s in subdirs_to_import:
    if (not s.lower() in map(str.lower,sys.path)):
        importPath = os.path.join(api_root,s)
        print("Adding {} to the python path".format(importPath))
        sys.path.insert(0,importPath)    


#%% Import classification modules

import api as speciesapi


#%% Build Latin --> common mapping

latinToCommon = {}

if taxonomy_path != None:
        
    print("Reading taxonomy file")
    
    # Read taxonomy file; takes ~1 minute
    df = pd.read_csv(taxonomy_path)
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

assert os.path.isfile(classification_model_path)
if detection_model_path != None:
    assert os.path.isfile(detection_model_path)

print("Loading model")
model = speciesapi.DetectionClassificationAPI(classification_model_path, 
                                              detection_model_path, image_sizes, use_gpu)
print("Finished loading model")


#%% Prepare the list of images and query names

queries = None

if isinstance(images_to_classify,str) and os.path.isdir(images_to_classify):
    
    images = glob.glob(os.path.join(images_to_classify,'**/*.*'), recursive=True)
    images = [fn for fn in images if os.path.isfile(fn)]
    queries = [os.path.basename(os.path.dirname(fn)) for fn in images]
    print('Loaded a folder of {} images'.format(len(images)))    
    
elif isinstance(images_to_classify,str) and os.path.isfile(images_to_classify):
    
    print("Reading image list file")
    df_images = pd.read_csv(images_to_classify,header=None)
    df_images.columns = ['filename','query_string']
    nImages = len(images)    
    print("Read {} image names".format(len(images)))
    images = list(df_images.filename)
    queries = list(df_images.query_string)
    assert(len(queries) == len(images))
    
else:
    
    assert isinstance(images_to_classify,list)
    images = images_to_classify
    queries = None
    print('Processing list of {} images'.format(len(images)))
    

#%% Classify images

nErrors = 0
nImagesClassified = 0
nImages = len(images)

if classification_output_file is not None:
    f = open(classification_output_file,'w+')

# i_fn = 1; fn = images[i_fn]    
for i_fn,fn in enumerate(images):
    
    print("Processing image {} of {}".format(i_fn,nImages))
    fn = fn.replace('\\','/')
    query = ''
    if queries is not None:
        query = queries[i_fn]
        
    if images_to_classify_base is not None and len(images_to_classify_base > 0):
        fn = os.path.join(images_to_classify_base,fn)

    # with torch.no_grad():
    # print('Clasifying image {}'.format(fn))
    # def predict_image(self, image_path, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
    try:
        prediction = model.predict_image(fn, topK=min(5,mak_k_to_print), multiCrop=False, 
                                             predict_mode=speciesapi.PredictMode.classifyOnly)
        nImagesClassified = nImagesClassified + 1
        
    except Exception as e:
        print("Error classifying image {} ({}): {}".format(i_fn,fn,str(e)))
        nErrors = nErrors + 1
        continue

    # i_prediction = 0
    for i_prediction in range(0, min(len(prediction.species),mak_k_to_print)):
        latinName = prediction.species[i_prediction]
        likelihood = prediction.species_scores[i_prediction]
        likelihood = '{0:0.3f}'.format(likelihood)
        commonName = doLatinToCommon(latinName)
        s = '"{}","{}","{}","{}","{}","{}","{}"'.format(
                i_fn,fn,query,i_prediction,latinName,commonName,likelihood)
        if classification_output_file is not None:
            f.write(s + '\n')
        print(s)
        
    if debug_max_images > 0 and i_fn >= debug_max_images:
        break

# ...for each image
        
if classification_output_file is not None:
    f.close()
    
print("Finished classifying {} of {} images ({} errors)".format(nImagesClassified,nImages,nErrors))
