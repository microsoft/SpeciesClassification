import os
import shutil
import traceback
import pickle
import pandas as pd
import csv
import common
import time
import configparser
import constant as c
from common import*
from common_config import Common_config
from PIL import Image
from PIL import ImageFile
from distutils.dir_util import copy_tree

config = Common_config()
root_direc = config.get_root_path()

animalimages_direc = root_direc + "/static/animals"
thumbnail_direc = root_direc + "/static/thumbnails"
 
ImageFile.LOAD_TRUNCATED_IMAGES = True

def update_animal_list():

  #df = pd.read_csv("static/data/animal_list.csv")
  
  #rename_invalid_directories()
  #rename_invalid_files()
  #df = update_list(df)
  #print(df.head())
  ##print("rename_files")
  #df = remove_no_file_enteries(df)
  ##print("remove_no_file_enteries")
  #df = remove_gif_files(df)
  ##print("remove_gif_files")

  ##print(df["Path"].str.contains("piranha"))	
  #df.to_csv('./static/data/updated_animal_list.csv')

  #resize_thubnail_images()
  print(animalimages_direc)
  save_files_folders_lowercase(animalimages_direc)
  save_files_folders_lowercase(thumbnail_direc)

def update_list(df):
  
  lst = []

  for i, row in df.iterrows():
    
    path = row["Path"]
    path, is_invalid_path = remove_invalid_str(path)
    if(is_invalid_path):
      lst.append({"Path": path, "Name": row["Name"]})
  
  r = pd.DataFrame(lst, columns=["Path", "Name"])

  return r

def rename_invalid_directories():
  
  animalimages_direc = './static/animals'

  for subdir, dirs, files in os.walk(animalimages_direc):
    for d in dirs:
         direc, is_invalid_direc = remove_invalid_str(d)
         #print(is_invalid_direc)
         if(is_invalid_direc):
          #print("invalid")
          src_direc = animalimages_direc + "/" + d
          dst_direc = animalimages_direc + "/" + direc
          os.rename(src_direc, dst_direc)
         
def rename_invalid_files():
  
  for subdir, dirs, files in os.walk(animalimages_direc):
    for f in files:
         file, is_invalid_file = remove_invalid_str(f)
         if(is_invalid_file):
          src_direc = subdir + "/" + f
          dst_direc = subdir + "/" + file
          if not os.path.exists(dst_direc):
            os.rename(src_direc, dst_direc)
         
def remove_no_file_enteries(dataframe):
  
  df = dataframe
  lst = []

  for i, row in df.iterrows():
    
    file = animalimages_direc  + "/" + row["Path"]   
    if(os.path.exists(file)):
      lst.append({"Path": row["Path"], "Name": row["Name"]})
  
  r = pd.DataFrame(lst, columns=["Path", "Name"])

  return r

def rename_files(dataframe):

  df = dataframe
  
  for i, row in df.iterrows():
    
    path_parts = row["Path"].split("/")
    direc = path_parts[0]
    src_file_name = path_parts[1]
    src_path= animalimages_direc + "/" + row["Path"]

    dst_file_name, is_invalid_file = remove_invalid_str(src_file_name)

    file_exists = os.path.isfile("./static/animals/" + row["Path"])
    
    if(is_invalid_file):
    	df.iloc[i,0] = direc + "/" + dst_file_name

    if(is_invalid_file and file_exists):
    	dst_path = animalimages_direc + "/" + direc +"/" + dst_file_name
    	os.replace(src_path, dst_path)
    
  return df

def remove_invalid_str(path):
  
  not_allowed_str = [" ", "+", "%"]
  has_invalid_str = False
    
  for c in not_allowed_str:
    if c in path:
      has_invalid_str = True
      print(path)
      path = path.replace(c, "_")
      print(path)
    
  return path, has_invalid_str

def remove_gif_files(dataframe):
  
  df = dataframe
  lst = []

  for i, row in df.iterrows():
    
    file = animalimages_direc + "/" + row["Path"]  
    file_ext = os.path.splitext(file)[1] 
    
    if(file_ext.lower() != ".gif"):
      lst.append({"Path": row["Path"], "Name": row["Name"]})
  
  r = pd.DataFrame(lst, columns=["Path", "Name"])
  return r

def setup_thubnail_directory():
 
 #os.rmdir only works when directory is empty, therefore
 #deleting all files within the directory
 for root, dirs, files in os.walk(thumbnail_direc, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
 
 if(os.path.exists(thumbnail_direc)):
  os.rmdir(thumbnail_direc)

 while os.path.exists(thumbnail_direc):
  time.sleep(1)
 
 os.mkdir(thumbnail_direc)
    
 copy_tree(animalimages_direc, thumbnail_direc)
 print("copying directory complete...")

def get_new_image_dimensions(image):
  
  maxWidth = 400
  maxHeight = 350

  width = image.size[0]
  height = image.size[1]

  ratio = maxWidth / width
  print(ratio)
        
  width = maxWidth
  height = height * ratio
        
  if(height < maxHeight):
    ratio = maxHeight / height
    width = maxWidth * ratio
    height = height * ratio

  return width, height

def resize_thubnail_images():
    try:
      
      setup_thubnail_directory()

      for subdir, dirs, files in os.walk(thumbnail_direc):
        for f in files: 
          
          file_ext = get_file_ext(f)
          if(file_ext.lower() not in ["jpg", "jpeg", "png"]):
            continue
          
          file_path = subdir  + "/" + f.lower()

          image = Image.open(file_path)
          width, height = get_new_image_dimensions(image)
          
          save_image_new_size(image, file_path, width, height)
          print("Resized image: " +  file_path)

    except Exception as e:
      var = traceback.format_exc()
      print(str(var))

def save_image_new_size(image, file_path, width, height):
  
  image.thumbnail((width, height), Image.ANTIALIAS)
  image.save(file_path, format=image.format)

def save_files_folders_lowercase(direc):

   for subdir, dirs, files in os.walk(direc):
        for d in dirs: 
          src = subdir + "\\" +  d
          dst = subdir + "\\" +  d.lower()
          os.rename(src, dst)
        for f in files:
          src = subdir + "\\" +  f
          dst = subdir + "\\" +  f.lower()
          os.rename(src, dst)
 
          

if __name__ == "__main__":
  #rename_directories()
  update_animal_list()