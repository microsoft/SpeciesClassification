# This script was used to create the extended iNat dataset.

import json
import numpy as np
import PIL.Image
import random
import glob
import os
import tqdm
import sys
import shutil

# Number of validation images per category in minival of iNat 2017
AVG_VAL_IMAGES_PER_CAT = 95986.0/5089/10

with open('./trainval_animals2017.json', 'rt') as jsfile:
    train_json = json.load(jsfile)
with open('./minival_animals2017.json', 'rt') as jsfile:
    val_json = json.load(jsfile)

OUTPUT_DIR = 'train_val_images'
OUTPUT_TRAIN_JSON = 'tmp_train.json'
OUTPUT_VAL_JSON = 'tmp_val.json'

INPUT_ORIGINAL_INAT_IMG_DIR = 'source/inat/'
# Two files listing animal names and the corresponding animal class
# The animal name matches the folder name
ANIMAL_NAMES_FILE = 'source/animal_name.txt'
ANIMAL_CLASS_FILE = 'source/animal_class.txt'

def main():
    # Create output dir
    try:
        os.makedirs(OUTPUT_DIR)
        for dd in glob.glob(INPUT_ORIGINAL_INAT_IMG_DIR+'/*/*'):
            os.makedirs(os.path.join(OUTPUT_DIR, dd[len(INPUT_ORIGINAL_INAT_IMG_DIR):]))
    except:
        print('Output folder {} already exists.'.format(OUTPUT_DIR))

    with open(ANIMAL_NAMES_FILE, 'rt') as f:
        animal_names = f.read().splitlines()
    with open(ANIMAL_CLASS_FILE, 'rt') as f:
        animal_classes = f.read().splitlines()
    animal_to_class = {n:c for n,c in zip(animal_names, animal_classes)}

    add_new_categories(train_json, val_json, 'source/dans_extended_set/*', AVG_VAL_IMAGES_PER_CAT, animal_to_class)
    add_new_categories(train_json, val_json, 'source/dogs/*', 0, animal_to_class)
    add_new_categories(train_json, val_json, 'source/dogs_val/*', 999999, animal_to_class)
    #add_new_categories(train_json, val_json, 'train_val_images/nonanimal/*', 0)

    # Add all iNat Images
    inat_images = glob.glob(os.path.join(INPUT_ORIGINAL_INAT_IMG_DIR, '*', '*', '*'))
    for im in tqdm.tqdm(inat_images):
        source_file = os.path.abspath(im)
        target_file = os.path.join(OUTPUT_DIR, im[len(INPUT_ORIGINAL_INAT_IMG_DIR):])
        os.symlink(source_file, target_file)

    # Finished, writing json
    with open(OUTPUT_TRAIN_JSON, 'wt') as tj:
        json.dump(train_json, tj)
    with open(OUTPUT_VAL_JSON, 'wt') as tj:
        json.dump(val_json, tj)


def add_new_categories(train_js, val_js, folder, avg_val_images_per_cat, animal_to_class):
    # Adds new categories to the training and validation json
    # train_js and val_js are json structure as used in iNat2017
    # Both json files should be identical except for the field 'images'
    # folder should be a glob pattern, e.g. 'myfolder/*'
    # we will derive the classnames from all immediate subdirectories
    # and use all images located in these subdirectories as class images
    # avg_val_images_per_cat is the average number of images that we will
    # assign to the validation json, all other images will be added to the 
    # training json

    random.seed(0)
    print('Working on {}'.format(folder))
    # Add new license
    if 'unknown' in [lic['name'] for lic in train_js['licenses']]:
        lic_id = [lic['id'] for lic in train_js['licenses'] if lic['name'] == 'unknown'][0]
    else:
        lic_id = max([lic['id'] for lic in train_js['licenses']]) + 1
        train_js['licenses'].append(dict(url='unknown', id=lic_id, name='unknown'))
        val_js['licenses'].append(train_js['licenses'][-1])
    # Add new classes
    new_classes_dirs = sorted(glob.glob(folder))
    classnames = [os.path.split(longpath)[-1] for longpath in new_classes_dirs]
    next_cat_id = max([cat['id'] for cat in train_js['categories']]) + 1
    existing_classes = set([cat['name'] for cat in train_js['categories']])
    for new_cat_idx, new_cat_name in enumerate(classnames):
        if new_cat_name not in existing_classes:
            target_superclass = animal_to_class[new_cat_name]
            train_js['categories'].append({'id':next_cat_id,
                'name':new_cat_name,
                'supercategory':target_superclass})
            val_js['categories'].append(train_js['categories'][-1])
            next_cat_id = next_cat_id + 1

    # Add images
    # This function assumes that all annotations in train_js and val_js are
    # equivalent except for the field 'images'
    max_im_id = max([im['id'] for im in train_js['images']] + [im['id'] for im in val_js['images']])
    max_an_id = max([an['id'] for an in train_js['annotations']] + [an['id'] for an in val_js['annotations']])
    cat_to_id = {cat['name']:cat['id'] for cat in train_js['categories']}
    for classname, classfolder in tqdm.tqdm(list(zip(classnames, new_classes_dirs))):
        cat_id = cat_to_id[classname]
        classimages = sorted(glob.glob(os.path.join(classfolder, '*')))
        target_class = os.path.split(classfolder)[1]
        target_superclass = animal_to_class[target_class]
        target_folder = os.path.join(OUTPUT_DIR, target_superclass, target_class)
        os.makedirs(target_folder, exist_ok=True)
        for classimage in classimages:
            # This calculation is within the loop to handle the case of 0 images
            val_prob = avg_val_images_per_cat / len(classimages)
            try:
                width, height = PIL.Image.open(classimage).size
                if PIL.Image.open(classimage).mode != 'RGB':
                    PIL.Image.open(classimage).convert('RGB').save(classimage)
                # Reading the image should come first so a failed load does not
                # mess up the IDs
                max_im_id = max_im_id + 1
                max_an_id = max_an_id + 1
                next_im_id = max_im_id
                next_an_id = max_an_id
                if random.random() < val_prob:
                    js = val_js
                else:
                    js = train_js
                # Make link to the image in the output folder
                target_file = os.path.join(target_folder, os.path.split(classimage)[1])
                if not os.path.exists(target_file):
                    os.symlink(os.path.abspath(classimage), target_file)
                else:
                    print('File / Symlink already exists: '+ target_file)
                js['images'].append(dict(id=next_im_id,
                                         width=width,
                                         height=height,
                                         file_name=target_file,
                                         license=lic_id,
                                         rights_holder=''))
                js['annotations'].append(dict(id=next_an_id,
                                              image_id=next_im_id,
                                              category_id=cat_id))
            except IOError:
                print('Cannot read image {}'.format(classimage))
                os.remove(classimage)
            except:
                print('Cannot read image {}'.format(classimage))
                raise


if __name__ == '__main__': 
    main()
