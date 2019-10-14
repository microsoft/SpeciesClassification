# Checks a COCO style classification JSON for bugs

#%% Imports
import json
import collections
import os


#%% Config
INPUT_JSON = '/path/to/file.json'
# If this is the JSON of iNat 2017, we can check if class names match with folder names
INAT_CLASSNAME_CHECK = True

#%% Load
with open(INPUT_JSON, 'rt') as fi:
    jfile = json.load(fi)

#%% Generate index
image_id_to_image = {im['id']:im for im in jfile['images']}

#%% Compute number of unique labels per folder
unique_labels = collections.defaultdict(lambda: set())
for ann in jfile['annotations']:
    im_path = image_id_to_image[ann['image_id']]['file_name']
    unique_labels[os.path.dirname(im_path)].add(ann['category_id'])

#%% Check if all images from the same folder have the same label
for k,v in unique_labels.items():
    assert len(v) == 1, 'Folder {} has multiple labels: {}'.format(k, str(v))


#%% Check if all ids are unique
all_image_ids = set([im['id'] for im in jfile['images']])
assert len(jfile['images']) == len(all_image_ids)
all_annotation_ids = set([an['id'] for an in jfile['annotations']])
assert len(jfile['annotations']) == len(all_annotation_ids)
all_cat_ids = set([c['id'] for c in jfile['categories']])
assert len(jfile['categories']) == len(all_cat_ids)

#%% Check if cross-referenced ids exist
for ann in jfile['annotations']:
    assert ann['image_id'] in all_image_ids, 'Annotation {} uses an invalid image ID of {}'.format(str(ann), ann['image_id'])
    assert ann['category_id'] in all_cat_ids, 'Annotation {} uses an invalid category ID of {}'.format(str(ann), ann['category_id'])

#%% iNat: check if class names match the folder name
cat_id_to_name = {cat['id']:cat['name'].replace(' × ', ' ').replace(' ×',' ') for cat in jfile['categories']}
img_id_to_path = {im['id']:im['file_name'].replace('  ', ' ') for im in jfile['images']}
for idx in range(len(jfile['annotations'])):
    assert cat_id_to_name[jfile['annotations'][idx]['category_id']] \
                    in img_id_to_path[jfile['annotations'][idx]['image_id']]


print('All seems good!')