# fix_iNatAll_json.py
#
# Script for fixing (and inspecting) the (trainval|minival)_iNatAllExtended2017.json files

import json
import os
import tqdm
import collections
import numpy as np


BASEDIR = '/data/animals2/iNat2017_extended/'

REFERENCE = os.path.join(BASEDIR, 'trainval2017.json')

FILES = {'trainval_iNatAllExtended2017.json' : 'trainval_iNatAllExtended2017_fixed.json',
        'minival_iNatAllExtended2017.json' : 'minival_iNatAllExtended2017_fixed.json'}

def load_json(path):
    with open(path, 'rt') as fi:
        j = json.load(fi)
    j['categories'] = sorted(j['categories'], key=lambda x:x['id'])
    return j

reference_json = load_json(os.path.join(BASEDIR, REFERENCE))
reference_cat_to_id = {cat['name']:cat['id'] for cat in reference_json['categories']}
#reference_cat_to_id = {cat['id']:cat['name'] for cat in reference_json['categories']}

for source, target in FILES.items():
    source_json = load_json(os.path.join(BASEDIR, source))
    #source_cat_to_ids = {cat['name']:cat['id'] for cat in source_json['categories']}
    #source_cat_id_to_reference = {source_cat_to_ids[cat['name']]:reference_cat_to_id[cat['name']] 
    #                                                        for cat in reference_json['categories']}
    #replacement = dict()
    # Replace cats with the ones from iNat
    for idx in range(len(reference_json['categories'])):
        source_json['categories'][idx]['name'] = reference_json['categories'][idx]['name']
        source_json['categories'][idx]['supercategory'] = reference_json['categories'][idx]['supercategory']

    # Fix annotation errors
    img_id_to_path = {im['id']:im['file_name'] for im in source_json['images']}
    cat_name_to_id = {cat['name'].replace(' × ', ' ').replace(' ×',' '):cat['id'] for cat in source_json['categories']}
    change_counter = 0
    for idx in range(len(source_json['annotations'])):
        cn = img_id_to_path[source_json['annotations'][idx]['image_id']].split('/')[2].replace('  ', ' ')
        if source_json['annotations'][idx]['category_id'] != cat_name_to_id[cn]:
            source_json['annotations'][idx]['category_id'] = cat_name_to_id[cn]
            change_counter = change_counter + 1

    # Check errors
    cat_id_to_name = {cat['id']:cat['name'] for cat in source_json['categories']}

    classnames = collections.defaultdict(lambda: [])
    proposed_c_id = collections.defaultdict(lambda: [])
    incorrect = []
    for idx in range(len(source_json['annotations'])):
        # old_id = source_json['annotations'][idx]['category_id']
        # if old_id in source_cat_id_to_reference:
        #     print('Replacing', old_id, source_cat_id_to_reference[old_id])
        #     source_json['annotations'][idx]['category_id'] = source_cat_id_to_reference[old_id]
        cn = img_id_to_path[source_json['annotations'][idx]['image_id']].split('/')[2].replace('  ', ' ')
        c_id = source_json['annotations'][idx]['category_id']
        source_cat_id = cat_name_to_id[cn]
        #assert classnames[c_id] is None or cn == classnames[c_id]
        classnames[c_id].append(cn)
        proposed_c_id[c_id].append(source_cat_id)
        #assert cat_id_to_name[source_json['annotations'][idx]['category_id']] \
        #                in img_id_to_path[source_json['annotations'][idx]['image_id']]:
        if cat_id_to_name[source_json['annotations'][idx]['category_id']].replace(' × ', ' ').replace(' ×',' ') \
                                not in img_id_to_path[source_json['annotations'][idx]['image_id']].replace('  ', ' '):
            incorrect.append(cn)
    assert len(incorrect) == 0

    with open(os.path.join(BASEDIR, target), 'wt') as fi:
        json.dump(source_json, fi, indent=1)