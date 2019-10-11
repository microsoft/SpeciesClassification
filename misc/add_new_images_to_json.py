# Add new images to existing classes of a JSON dataset (COCO format)

#%% Imports
import json
import os
import glob
import PIL.Image


#%% Config

# The JSON file that will receive the new images
INPUT_FILE = '/path/to/trainval.json'
assert os.path.isfile(INPUT_FILE)
# The output file to write the resulting JSON
OUTPUT_FILE = '/path/to/trainval_V2.json'
assert not os.path.isfile(OUTPUT_FILE)
# file_name pattern used in the JSON files above
# The placeholder will be replaced by the relative path in *TARGET_TRAIN_VAL_DIR*
FILE_NAME_PATTERN = 'train_val_images/{}'
# Extension whitelist, not case sensitive, format is '.EXT' including the dot
EXT_WHITELIST = ['.jpg', '.jpeg']
# Resize larger side of images to match image dimensions/statistics of the remaining dataset
# If we don't do this, the classifier will learn that any high-res images belongs to these classes
# We only do downsizing to decrease image quality to the same level as all other images
RESIZE_LARGE_IMAGES_TO = 800


# Folder with the new images, that should be added
# This folder should have the same structure as the `train_val_images` dir, i.e.
# if `new_hippo_image1.jpg` is a new hippo image, then it should be located here:
#     $NEW_IMAGE_DIR/Mammalia/Hippopotamus amphibius/new_hippo_image1.jpg
NEW_IMAGE_DIR = '/path/to/new_image_dir'
assert os.path.isdir(NEW_IMAGE_DIR)
if not os.path.isabs(NEW_IMAGE_DIR):
    NEW_IMAGE_DIR = os.path.abspath(NEW_IMAGE_DIR)
# The output folder, i.e. the `train_val_images` dir 
TARGET_TRAIN_VAL_DIR = '/path/to/train_val_images'
assert os.path.isdir(TARGET_TRAIN_VAL_DIR)
if not os.path.isabs(TARGET_TRAIN_VAL_DIR):
    NEW_IMAGE_DIR = os.path.abspath(TARGET_TRAIN_VAL_DIR)


#%% Code
def main():
    #%% Load
    with open(INPUT_FILE, 'rt') as fi:
        jfile = json.load(fi)

    #%% Mapping
    cname_to_id = {c['name']:c['id'] for c in jfile['categories']}

    #%% Discover new images
    os.chdir(NEW_IMAGE_DIR)
    class_folders = [p for p in glob.glob('*/*') if os.path.isdir(p)]
    print('Found the following class folders in the input:')
    print(class_folders)

    # Make sure all class folders also exist in the output
    for cf in class_folders:
        assert os.path.isdir(os.path.join(TARGET_TRAIN_VAL_DIR, cf)), "Target folder {} does not exist".format(cf)

        #%% Collect all new images of this class
        input_dir = os.chdir(os.path.join(NEW_IMAGE_DIR, cf))
        new_images = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in EXT_WHITELIST]

        #%% Get class ID and next available image id
        class_id = cname_to_id[cf.split(os.path.sep)[1]]

        print('\nWe will add the following {} images to class {}\n'.format(len(new_images), cf))
        #%% For each image
        for im_file in new_images:

            target_file = os.path.join(TARGET_TRAIN_VAL_DIR, cf, im_file)
            assert not os.path.exists(target_file), 'The target file {} already exists'.format(target_file)

            # Load and resize image, if necessary
            im_object = PIL.Image.open(os.path.join(NEW_IMAGE_DIR, cf, im_file))
            scaling_ratio = RESIZE_LARGE_IMAGES_TO / max(im_object.size)
            # We only downsize large images, to reduce the quality to the same level as all other images in the dataset
            if scaling_ratio < 1:
                
                new_shape = (int(im_object.width * scaling_ratio), int(im_object.height * scaling_ratio))
                im_object = im_object.resize(new_shape, PIL.Image.ANTIALIAS)
            
            im_object.save(target_file)

            #%% Insert each image to the image list and annotations list
            last_image_id = max((im['id'] for im in jfile['images']))
            last_annotation_id = max((a['id'] for a in jfile['annotations']))
            jfile['images'].append({'id': last_image_id + 1,
                                            'width': im_object.width,
                                            'height': im_object.height,
                                            'file_name': FILE_NAME_PATTERN.format(os.path.join(cf, im_file)),
                                            'license': 9,
                                            'rights_holder': ''})
            jfile['annotations'].append({'id': last_annotation_id + 1, 
                                            'image_id': last_image_id + 1,
                                            'category_id': class_id})
            print(jfile['images'][-1]['file_name'])

    print('\n\nWriting output to file ' + OUTPUT_FILE)
    with open(OUTPUT_FILE, 'wt') as fi:
        json.dump(jfile, fi)


if __name__ == '__main__': 
    main()
