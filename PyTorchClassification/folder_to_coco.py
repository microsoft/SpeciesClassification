import json
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from random import random


cwd = Path.cwd()
DATA = 'data'                           # Name of data directory
FOLDER = 'train'                        # Name of the folder containing the images
SPLIT = 0.2                             # Train / Validation split ratio
CLASS_MAPPER = 'class_id_mapping.csv'   # class -> id mapper file


def folder2coco(folder, map_file, pct=0.2):
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
    data_dir = cwd / DATA

    # Create class_id to species mapping
    mapper = pd.read_csv(data_dir / map_file)
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
    for idx, species_dir in enumerate((data_dir / folder).iterdir()):
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
                        'file_name': f'/{folder}/{species_dir.name}/{image_path.name}',
                        'width': w,
                        'height': h})
                coco['annotations'].append(
                    {'id': counter, 'image_id': counter, 'category_id': idx})
                counter += 1
            except OSError:
                pass

    return train, valid


def main():
    '''Stores the COCO annotation inside the DATA directory as train.json and valid.json.'''
    train, valid = folder2coco(FOLDER, CLASS_MAPPER, SPLIT)

    json.dump(train, (cwd / DATA / 'train.json').open('wt', encoding='utf-8'))
    json.dump(valid, (cwd / DATA / 'valid.json').open('wt', encoding='utf-8'))


if __name__ == '__main__':
    main()
