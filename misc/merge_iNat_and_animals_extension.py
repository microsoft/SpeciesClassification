import json
# Minival
# - Extension
js = json.load(open('./minival_animalsExtended2017.json'))
splitid = [idx for idx,im in enumerate(js['images']) if im['license'] == 9][0]
js['images'] = js['images'][splitid:]
js['annotations'] = js['annotations'][splitid:]
# - Original iNat
inat = json.load(open('./minival2017.json'))
import ipdb; ipdb.set_trace()
js['images'] = inat['images'] + js['images']
js['annotations'] = inat['annotations'] + js['annotations']
# Write out
json.dump(js, open('./minival_iNatAllExtended2017.json', 'wt'))


# Trainval
# - Extension
js = json.load(open('./trainval_animalsExtended2017.json'))
splitid = [idx for idx,im in enumerate(js['images']) if im['license'] == 9][0]
js['images'] = js['images'][splitid:]
js['annotations'] = js['annotations'][splitid:]
# - Original iNat
inat = json.load(open('./trainval2017.json'))
import ipdb; ipdb.set_trace()
js['images'] = inat['images'] + js['images']
js['annotations'] = inat['annotations'] + js['annotations']
# Write out
json.dump(js, open('./trainval_iNatAllExtended2017.json', 'wt'))

