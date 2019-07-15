import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from random import random

DATA_ROOT = 'data'           # Name of data directory root, absolute or relative
IMAGE_FOLDER = 'train'                  # Name of the folder containing the images, relative to DATA_ROOT
SPLIT = 0.2                             # Train / validation split ratio
CLASS_MAPPER = 'class_idx_mapping.csv'   # class -> id mapper file

def folder2coco(data_root, image_folder, map_file, pct=0.2):
    '''
    Creates COCO data format to feed into the model.
    Inputs:
        folder - name of the folder containing the images
        map    - class to id mapper file
        pct    - train / valid split ratio
    Returns:
        train  - training dataset annotation in COCO format (dict)
        valid  - validation dataset annotation in COCO format (dict)
    '''
    # Create class_id to species mapping
    mapper = pd.read_csv(os.path.join(data_root,map_file))
    id2species = {
        idx: species for idx,
        species in zip(
            mapper['class_idx'].values,
            mapper['original_class'].values)}

    # Define the coco format
    train = defaultdict(list)
    valid = defaultdict(list)
    info = {
        'description': 'The 2019 Snake Species Identification Challenge',
        'url': 'https://www.aicrowd.com/challenges/snake-species-identification-challenge',
        'version': 1.0,
        'date_created': '2019-05-10'
    }
    train['info'] = info
    valid['info'] = info

    counter = 0
    # Enumerate across all the images and split them into train and valid set
    for idx, species_dir in enumerate(Path(os.path.join(data_root,image_folder)).iterdir()):
        train['categories'].append(
            {'id': idx, 'name': id2species[int(species_dir.stem.split('-')[-1])]})
        valid['categories'].append(
            {'id': idx, 'name': id2species[int(species_dir.stem.split('-')[-1])]})
        for image_path in species_dir.iterdir():
            try:
                coco = train if random() > pct else valid
                (w, h) = Image.open(image_path).size
                coco['images'].append(
                    {
                        'id': counter,
                        'file_name': f'/{image_folder}/{species_dir.name}/{image_path.name}',
                        'width': w,
                        'height': h})
                coco['annotations'].append(
                    {'id': counter, 'image_id': counter, 'category_id': idx})
                counter += 1
            except OSError:
                pass

    return train, valid


def main():
    '''Stores the COCO annotation inside the DATA_ROOT directory as train.json and valid.json.'''
    train, valid = folder2coco(DATA_ROOT, IMAGE_FOLDER, CLASS_MAPPER, SPLIT)

    json.dump(train, Path(os.path.join(DATA_ROOT,'train.json')).open('wt', encoding='utf-8'))
    json.dump(valid, Path(os.path.join(DATA_ROOT,'valid.json')).open('wt', encoding='utf-8'))


if __name__ == '__main__':
    main()
