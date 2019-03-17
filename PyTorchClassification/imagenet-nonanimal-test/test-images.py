import sys
sys.path.append('/code/DetectionClassificationAPI')
sys.path.append('/code/PyTorchClassification')
sys.path.append('/code/FasterRCNNDetection')
import numpy as np
import os
from models import *
import tqdm
import PIL
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import data_loader 
random.seed(0)

IMAGENET_TRAIN_DIR = '/data/imagenet/val/'
IMAGE_SIZE = 448
MODEL_PATH = \
'../models/resnext-448-bing-onevsall-78.2/model_best.pth.tar'
PLOT_OUTFILE = 'result/bing_resnext448_onevsall'
os.makedirs('result', exist_ok=True)
NONANIMALS_IMAGELIST_FILE = 'imagenet-nonanimal-val.txt' #'imagenet_nonAnimals_animalIntent.tsv'
IMAGE_ROOT_DIR = '/data/animals2/species_extended/'
NUM_IMAGE_SAMPLES = None

assert os.path.isfile(MODEL_PATH)
model = ClassificationModel(MODEL_PATH, image_sizes=[IMAGE_SIZE], useGPU=True)
def get_scores(model, pathlist):
    resultlist = []
    for impath in tqdm.tqdm(pathlist):
        test_image = PIL.Image.open(impath).convert('RGB')
        species, species_scores = model.predict_from_image(test_image, topK=1, multiCrop=False)
        resultlist.append(1/(1+np.exp(-species_scores)))
    return resultlist

# Load list of animal images, we assume that they are randomly sorted already
animals_val_dataset = data_loader.JSONDataset(IMAGE_ROOT_DIR, os.path.join(IMAGE_ROOT_DIR, 'minival_animals2017.json'), [448], False, dataFormat2017=True)
animal_images = [os.path.join(IMAGE_ROOT_DIR, impath) for impath in animals_val_dataset.imgs]
#animal_images = np.loadtxt(ANIMALS_IMAGELIST_FILE, dtype=str, delimiter=',')[:len(nonanimal_scores)]
if NUM_IMAGE_SAMPLES is not None:
    animal_scores = get_scores(model, random.sample(animal_images, NUM_IMAGE_SAMPLES))
else:
    animal_scores = get_scores(model, animal_images)
np.savetxt(PLOT_OUTFILE + 'animal_scores.txt', animal_scores, fmt='%.6f')

#Get non animal scores
fullcsv = np.loadtxt(NONANIMALS_IMAGELIST_FILE, dtype=str)
imagelist = fullcsv[:] #[:,0]
#animalscore = fullcsv[:,1].astype(float)
#img_selection = np.logical_and(animalscore < 0.733, animalscore > 0.058)
imagelist_abspath = [os.path.join(IMAGENET_TRAIN_DIR, impath.replace('\\','/')) for impath in imagelist] #[img_selection]]
# we sample 0.73/0.27 as many non-animal images because that is the ratio of non-animal / animal images in our application
nonanimal_scores = get_scores(model, random.sample(imagelist_abspath, int(len(animal_scores)*0.73/0.27)))
np.savetxt(PLOT_OUTFILE + 'nonanimal_scores.txt', nonanimal_scores, fmt='%.6f')

# Plot results
targets = np.hstack([np.ones((len(animal_scores),)), np.zeros((len(nonanimal_scores),))])
pred = np.vstack([animal_scores, nonanimal_scores])[:,0]

average_precision = average_precision_score(targets, pred)
precision, recall, thresholds = precision_recall_curve(targets, pred, 1)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(
          average_precision))

plt.savefig(PLOT_OUTFILE + '.pdf', format='pdf')
plt.savefig(PLOT_OUTFILE + '.png', format='png')


