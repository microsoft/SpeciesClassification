from __future__ import division

import os
import shutil
import time
import csv
import pandas as pd
import common
import config
import constant as c
from common import*
from PIL import Image, ImageFile
import imghdr

static_dir = "static"

root = config.ROOT_PATH

data_dir = os.path.join(root, static_dir, "data")
animalimages_dir =  os.path.join(root, static_dir, "animals")
thumbnail_dir = os.path.join(root, static_dir, "thumbnails")

csv_path = os.path.join(data_dir, "updated_animal_list.csv")
 
ImageFile.LOAD_TRUNCATED_IMAGES = True

def update_animal_list():

    print("\nRename invalid directory names e.g replace chars +, %, emptyspace with '_'... ")
    rename_invalid_directory_names()

    print("\nRename invalid file names e.g replace chars +, %, emptyspace with '_'... ")
    rename_invalid_file_names()
  
    file_count = create_csv_animal_list(csv_path)

    if(file_count < 8):
       print("\nERROR: aborting, atleast 8 valid sample images are required (jpg, jpeg or png), found only: {} in directory {}\n"
             .format(str(file_count), animalimages_dir))
       df = pd.read_csv(csv_path)
       print("Images found:")
       print(df)

    else:
        print("\nCreate thumbnail images")
        create_thumbnail_images()


''' Replace characters +, %, space in directory names with _ '''
def rename_invalid_directory_names():
    for subdir, dirs, files in os.walk(animalimages_dir):
        for d in dirs:
            dir, is_invalid_str = remove_invalid_str(d)
         
            if(is_invalid_str):
                src_dir = animalimages_dir + "/" + d
                dst_dir = animalimages_dir + "/" + dir
                os.rename(src_dir, dst_dir)

''' Replace characters +, %, space in file names with _ '''        
def rename_invalid_file_names():
  for subdir, dirs, files in os.walk(animalimages_dir):
      for f in files:
          file_path, is_invalid_str = remove_invalid_str(f)
          
          if(is_invalid_str):
              src_direc = subdir + "/" + f
              dst_direc = subdir + "/" + file_path
             
              if not os.path.exists(dst_direc):
                  os.rename(src_direc, dst_direc)


def create_csv_animal_list(csv_path):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    count = 0
    with open(csv_path, 'wb') as writeFile:
        writer = csv.writer(writeFile, delimiter=',')
        print(animalimages_dir)
        writer.writerow(["Path", "Name"])
        
        for path, subdirs, files in os.walk(animalimages_dir):
            for file_name in files:
                file_ext = get_file_ext(file_name)
                
                if file_ext is not None and file_ext in ["jpg", "jpeg", "png"]:
                    animal_name = extract_animal_name_from_path(path)
                    file_path = os.path.basename(path) + "\\" + file_name
                    writer.writerow([file_path, animal_name])
                    count += 1

    return count

def extract_animal_name_from_path(path):
    animal_name = os.path.basename(path)
    
    lst = ["_", "-", "%"]
    for char in lst:
        if char in animal_name:
            animal_name = animal_name.replace(char, " ")
    return animal_name

def remove_invalid_str(path):
    not_allowed_str = [" ", "+", "%"]
    has_invalid_str = False
    
    for char in not_allowed_str:
        if char in path:
            has_invalid_str = True
            path = path.replace(char, "_")
    
    return path, has_invalid_str

def get_new_image_dimensions(image):
    maxWidth = 400
    maxHeight = 350

    width = image.size[0]
    height = image.size[1]

    ratio = maxWidth / width
    width = maxWidth
    
    height = height * ratio
        
    if(height < maxHeight):
        ratio = maxHeight / height
        width = maxWidth * ratio
        height = height * ratio

    return width, height

''' Create thumbnail images which are resized versions of images in the animals folder
    these images will be displayed on the website '''
def create_thumbnail_images():
    try:
        if os.path.exists(thumbnail_dir):
            print(thumbnail_dir)
            shutil.rmtree(thumbnail_dir)
        
        while os.path.exists(thumbnail_dir):
            sleep(1)

        os.mkdir(thumbnail_dir)

        for subdir, dirs, files in os.walk(animalimages_dir):
            for f in files: 
                file_ext = get_file_ext(f)
          
                if file_ext.lower() in ["jpg", "jpeg", "png"]:
                    file_path = os.path.join(subdir, f)
                    new_file_path = file_path.replace(animalimages_dir, thumbnail_dir)
                    save_image_new_size(file_path, new_file_path)
              
    except Exception as e:
      print(str(e))

def save_image_new_size(file_path, new_file_path):
    image = Image.open(file_path)
    width, height = get_new_image_dimensions(image)
    image.thumbnail((width, height), Image.ANTIALIAS)

    if not os.path.exists(os.path.dirname(new_file_path)):
        os.makedirs(os.path.dirname(new_file_path))
    
    image.save(new_file_path, format=image.format)
    print(new_file_path)
     

if __name__ == "__main__":
    update_animal_list()
