# This script was used to compile a validation set for the Stanford dogs dataset 
# from the ImageNet validation images. The testing set of the Stanford dogs dataset
# itself cannot be used, because it is a subset of the ImageNet training data.

import numpy as np
import shutil
import random
import glob
import os
random.seed(0)

# Number of validation images per category in minival
AVG_VAL_IMAGES_PER_CAT = 95986.0/5089/10

valdata = np.loadtxt('./../imagenet/val.txt', delimiter=' ', dtype=str)
synmapping = open('./../imagenet/classes.txt','rt').read().splitlines()
classfolders = np.loadtxt('./species_extended/source/original_dog_names.txt',
                          dtype=str, delimiter=',')

for cf in classfolders:
    outdir = os.path.join('new_dogs_val', 'Canis familiaris | ' +
                          cf.lower()[10:].replace('_', ' '))
    os.makedirs(outdir)
    synset = cf[:cf.index('-')]
    cur_synmap = [s for s in synmapping if synset in s][0]
    class_id = int(cur_synmap[:cur_synmap.index(',')]) - 1
    cur_test_images = valdata[valdata[:,1].astype(int) == class_id, 0]
    val_prob = AVG_VAL_IMAGES_PER_CAT / len(cur_test_images)
    for impath in cur_test_images:
        if random.random() < val_prob:
            shutil.copy(os.path.join('../imagenet/val/',impath),
                        outdir + '/')

