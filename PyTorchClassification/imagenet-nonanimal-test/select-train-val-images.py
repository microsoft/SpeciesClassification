import sys
sys.path.append('/code/iNatAPI')
sys.path.append('/code/iNatEnsemble')
sys.path.append('/code/iNatFasterRCNN')
import numpy as np
import os
import tqdm
import PIL
import random
random.seed(0)

IMAGENET_TRAIN_DIR = '/datadrive/imagenet/train/'
IMAGE_SIZE = 448
NONANIMALS_IMAGELIST_FILE = 'imagenet_nonAnimals_animalIntent.tsv'
NONANIMALS_TRAIN_OUTFILE = 'imagenet-nonanimal-train.txt'
NONANIMALS_VAL_OUTFILE = 'imagenet-nonanimal-val.txt'
NUM_TRAIN_CLASSES = 20

#Get non animal scores
fullcsv = np.loadtxt(NONANIMALS_IMAGELIST_FILE, dtype=str)
imagelist = fullcsv[:,0]
animalscore = fullcsv[:,1].astype(float)
is_nonanimal_image = np.logical_and(animalscore < 0.733, animalscore > 0.058)
imagelist_filtered = imagelist[is_nonanimal_image]
# Randomly selected an image and use it's class if it's not in the train set already
# Compared to randomly selecting a class we are more likely to select a class
# with many images
train_classes = set()
while len(train_classes) < NUM_TRAIN_CLASSES:
    selected_image = random.choice(imagelist_filtered)
    # Get class from image path
    classname = selected_image[:selected_image.index('\\')]
    train_classes.add(classname)
# Write out all images of the train and val classes
def write_imagelist(outfile, classes, imagelist):
    relevant_images = [im.replace('\\','/') for im in imagelist if im[:im.index('\\')] in classes]
    np.savetxt(outfile, relevant_images, fmt='%s')
all_classes = set([im[:im.index('\\')] for im in imagelist])
write_imagelist(NONANIMALS_TRAIN_OUTFILE, train_classes, imagelist_filtered)

val_classes = all_classes - train_classes
# Alternative 1: get the images from the list as done with training images
#write_imagelist(NONANIMALS_VAL_OUTFILE, val_classes, imagelist_filtered)
# Alternative 2: get the validation images from the image val data
# Get class IDs from the classes.txt file
imagenet_classes_1 = open('/datadrive/imagenet/classes.txt', 'rt').read().splitlines()[1:]
imagenet_classes_2 = [im[im.index(',')+1:] for im in imagenet_classes_1]
imagenet_classes = [im[:im.index(',')] for im in imagenet_classes_2]
val_class_ids = set([imagenet_classes.index(cls) for cls in val_classes])
# Get val images of these classes
imagenet_val_images = np.loadtxt('/datadrive/imagenet/val.txt', delimiter=' ', dtype=str)
relevant_imagenet_val_images = [(int(im_cls) in val_class_ids) for im_cls in imagenet_val_images[:,1]]
relevant_val_paths = imagenet_val_images[relevant_imagenet_val_images, 0]
import pdb; pdb.set_trace()
np.savetxt(NONANIMALS_VAL_OUTFILE, relevant_val_paths, fmt='%s')
