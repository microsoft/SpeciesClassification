#
# convert_folders_to_coco_format.py
#
# Converts a data set in which class names are specified in folders to
# a COCO-style .json file.
#

import json
import numpy as np
import PIL.Image
import random
import glob
import os
import tqdm
import sys
import shutil
import argparse


def main():

    parser = argparse.ArgumentParser('Script for converting a folder-based classification dataset into COCO format. ' + \
                    'Your dataset should be organized by folder, e.g. you need a separate subfolder for each class and '+\
                    'within this folder should be all the images of the corresponding class.')
    parser.add_argument("input_directory", type=str, metavar='IMAGE_ROOT',
                        help='Path to the root of your image collection. Each subfolder is treated as a seprate class. ' + \
                        'For example, if you set this to "/path/to/images", then images of class "myfirstclass" should be in the ' + \
                        'folder "path/to/images/myfirstclass/", images of class "mysecondclass" should be in the folder ' + \
                        '"/path/to/images/mysecondclass" and so on')
    parser.add_argument("output_directory", type=str, metavar='OUTPUT_PATH',
                        help='The folder where you want to output be written to.')
    parser.add_argument("--test_proportion", type=float, default=0.3, metavar='0.3',
                        help='Proportion of images used for testing.')
    args = parser.parse_args()
    
    print('\n-----------------------')
    print('IMPORTANT: please make sure that there are only images in your folder tree.')
    print('-----------------------\n')

    OUTPUT_DIR = args.output_directory
    IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
    OUTPUT_TRAIN_JSON = os.path.join(OUTPUT_DIR, 'training.json')
    OUTPUT_VAL_JSON = os.path.join(OUTPUT_DIR, 'testing.json')

    # Create output dir
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        print('Output folder {} already exists.'.format(OUTPUT_DIR))
        
    train_json = dict(info='', images=[], categories=[], annotations=[], licenses=[])
    val_json = dict(info='', images=[], categories=[], annotations=[], licenses=[])

    add_new_categories(train_json, val_json, os.path.join(args.input_directory, '*'), args.test_proportion, OUTPUT_DIR)

    # Finished, writing json
    with open(OUTPUT_TRAIN_JSON, 'wt') as tj:
        json.dump(train_json, tj)
    with open(OUTPUT_VAL_JSON, 'wt') as tj:
        json.dump(val_json, tj)


def add_new_categories(train_js, val_js, folder, val_prob, OUTPUT_DIR):
    # Adds new categories to the training and validation json
    # train_js and val_js are json structure 
    #
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
        lic_id = max([-1] + [lic['id'] for lic in train_js['licenses']]) + 1
        train_js['licenses'].append(dict(url='unknown', id=lic_id, name='unknown'))
        val_js['licenses'].append(train_js['licenses'][-1])
    # Add new classes
    new_classes_dirs = sorted(glob.glob(folder))
    new_classes_dirs = list(filter(lambda x: os.path.isdir(x), new_classes_dirs))
    classnames = [os.path.split(longpath)[-1] for longpath in new_classes_dirs]
    next_cat_id = max([-1] + [cat['id'] for cat in train_js['categories']]) + 1
    existing_classes = set([cat['name'] for cat in train_js['categories']])
    for new_cat_idx, new_cat_name in enumerate(classnames):
        if new_cat_name not in existing_classes:
            target_superclass = 'Entity'
            train_js['categories'].append({'id':next_cat_id,
                'name':new_cat_name,
                'supercategory':target_superclass})
            val_js['categories'].append(train_js['categories'][-1])
            next_cat_id = next_cat_id + 1

    # Add images
    # This function assumes that all annotations in train_js and val_js are
    # equivalent except for the field 'images'
    max_im_id = max([-1] + [im['id'] for im in train_js['images']] + [im['id'] for im in val_js['images']])
    max_an_id = max([-1] + [an['id'] for an in train_js['annotations']] + [an['id'] for an in val_js['annotations']])
    cat_to_id = {cat['name']:cat['id'] for cat in train_js['categories']}
    for classname, classfolder in tqdm.tqdm(list(zip(classnames, new_classes_dirs))):
        cat_id = cat_to_id[classname]
        classimages = sorted(glob.glob(os.path.join(classfolder, '*')))
        target_class = os.path.split(classfolder)[1]
        target_superclass = 'Entity'
        target_folder = os.path.join(OUTPUT_DIR, target_superclass, target_class)
        os.makedirs(target_folder, exist_ok=True)
        for classimage in classimages:
            # This calculation is within the loop to handle the case of 0 images
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
                    shutil.copy(os.path.abspath(classimage), target_file)
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
