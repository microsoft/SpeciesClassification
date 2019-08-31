import json
import os

os.chdir('/path/to/dataset/root/')

def merge(input1, input2, output, old_new_mapping = [{}, {}]):
    print('mapping', len(old_new_mapping[0]), len(old_new_mapping[1]))
    # input 1 is the inat json
    # input 2 is the extension-only json
    # {659: 1221, 216: 375}
    input1_class_blacklist = [1221, 375]
    # First json
    with open(input1, 'rt') as fi:
        js1 = json.load(fi)
    # Second json
    with open(input2, 'rt') as fi:
        js2 = json.load(fi)

    # Delete duplicate classes from input1
    images_to_delete = set()
    for ann_idx in range(len(js1['annotations'])):
        if js1['annotations'][ann_idx]['category_id'] in input1_class_blacklist:
            images_to_delete.add(js1['annotations'][ann_idx]['image_id'])
    for k,v in js1.items():
        print(k,len(v))
    print('Going to delete {} images due to duplicate classes'.format(len(images_to_delete)))
    js1['images'] = [im for im in js1['images'] if im['id'] not in images_to_delete]
    js1['categories'] = [cat for cat in js1['categories'] if cat['id'] not in input1_class_blacklist]
    js1['annotations'] = [ann for ann in js1['annotations'] if ann['image_id'] not in images_to_delete]
    for k,v in js1.items():
        print(k,len(v))

    # Renumber classes in input1
    max_class_id = -1
    for old_id in set([ann['category_id'] for ann in js1['annotations']]):
        if old_id not in old_new_mapping[0].keys():
            old_new_mapping[0][old_id] = max_class_id + 1
            max_class_id += 1
    for cat_idx in range(len(js1['categories'])):
        js1['categories'][cat_idx]['id'] = old_new_mapping[0][js1['categories'][cat_idx]['id']]
    for ann_idx in range(len(js1['annotations'])):
        js1['annotations'][ann_idx]['category_id'] = old_new_mapping[0][js1['annotations'][ann_idx]['category_id']]

    # Renumber classes in input2
    max_class_id = max([cat['id'] for cat in js1['categories']])
    for new_id, old_id in enumerate(list(set([ann['category_id'] for ann in js2['annotations']]))):
        if old_id not in old_new_mapping[1].keys():
            old_new_mapping[1][old_id] = max_class_id + 1
            max_class_id += 1
    #assert len(set([cat['id'] for cat in js1['categories']]) & set(old_new_mapping[1].values())) == 0
    import ipdb; ipdb.set_trace()
    js2['categories'] = [cat for cat in js2['categories'] if cat['id'] in old_new_mapping[1].keys()]
    for cat_idx in range(len(js2['categories'])):
        js2['categories'][cat_idx]['id'] = old_new_mapping[1][js2['categories'][cat_idx]['id']]
    for ann_idx in range(len(js2['annotations'])):
        js2['annotations'][ann_idx]['category_id'] = old_new_mapping[1][js2['annotations'][ann_idx]['category_id']]
    import ipdb; ipdb.set_trace()

    js1['images'] += js2['images']
    js1['annotations'] += js2['annotations']
    js1['categories'] += js2['categories']
    # Write out
    with open(output, 'wt') as fi:
        json.dump(js1, open(output, 'wt'))
    print('mapping', len(old_new_mapping[0]), len(old_new_mapping[1]))
    return old_new_mapping

old_new_mapping = merge('trainval2017.json', 'trainval_animalsExtended2017_extensionOnly.json', 'trainval_iNatAllExtended2017.json')
merge('minival2017.json', 'minival_animalsExtended2017_extensionOnly.json', 'minival_iNatAllExtended2017.json', old_new_mapping)
