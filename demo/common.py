import os
import sys
import uuid
import requests
import traceback
import numpy as np
import config
from PIL import Image
from urlparse import urlparse
from io import StringIO
from enums import*
from flask import Flask, render_template, request, jsonify, make_response, url_for, g

root_dir = config.ROOT_PATH
sample_dir = config.SAMPLE_PATH
upload_dir = config.UPLOAD_PATH

max_file_size = config.MAX_FILE_SIZE

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def delete_all_uploaded():
  directory = "." + upload_dir
  
  for root, dirs, files in os.walk(directory):
    for file in files:
      print(file)
      os.remove(file)

def get_file_ext(file_path):
  temp = file_path.split('.')
  length = len(temp)
  
  if(length > 0):
    return temp[length - 1]
  return "jpg"

def get_parsed_url(url):
    url_parsed = urlparse(url)
    url = url_parsed.geturl()

    return url.strip()

def get_img_filename_from_url(self):
    temp = request.args.get("url").split('\\')
    temp_len = len(temp)
    rel_path = "{}/{}".format(temp[temp_len - 2], temp[temp_len - 1])
    
    return rel_path
  
def is_largeimage_size(file):
    img_bytes = file.read()
    file_size = sys.getsizeof(img_bytes)

    if(file_size < max_file_size):
      return False
    return True

def has_large_dimensions(file):
    img = Image.open(file)
    width, height = img.size
    
    if(width > 1600):
      return True
    if(height > 1600):
      return True

def resize_image(img_path, img_full_path):
    try:
      image = Image.open(open(img_full_path, mode='rb'))

      width = image.size[0]
      height = image.size[1]
  
      file_size = None

      if(width < height and width > 1600):
        img_bytes, img_path, image = save_image_new_size(image, 1600, height)
        file_size = sys.getsizeof(img_bytes)
      
      elif(height < width and height > 1600):
        img_bytes, img_path, image = save_image_new_size(image, width, 1600)
        file_size = sys.getsizeof(img_bytes)
      
      if(not file_size is None):
        while (file_size > max_file_size):
          next_width = width - 150
          next_height = next_width
          
          img_bytes, img_path, image = save_image_new_size(image, next_width, 
                                       next_height)
          
          file_size = sys.getsizeof(img_bytes)
          width = next_width

      return img_path

    except Exception as e:
      print(str(e))
      return None

def save_image_new_size(image, width, height):
    img_path =  upload_dir + "/" +  str(uuid.uuid4()) + "." + image.format
    file_path = root_dir +"/" + img_path
  
    '''thumbnail will maintain aspect ratio'''
    image.thumbnail((width, height), Image.ANTIALIAS)
    image.save(file_path, format=image.format)
    img_bytes = open(file_path, mode='rb').read()
    
    return img_bytes, img_path, image

def save_url_img(url, file_content):
    file_ext  = get_file_ext(url)
    if(not file_ext.lower() in ["jpg", "jpeg", "png"]):
      file_ext  = "jpg"
    
    file_name = str(uuid.uuid4()) + "." + file_ext

    file_path = root_dir + "/" +  upload_dir + "/" + file_name
    
    file = open(file_path, 'wb')
    file.write(file_content)
    file.close()

    img_path = upload_dir + "/" + file_name
    return img_path

def save_posted_file(posted_file):
    file_ext = get_file_ext(posted_file.filename)
    file_name = str(uuid.uuid4()) + posted_file.filename
    file_path = root_dir + "/" +  upload_dir + "/" + file_name
    posted_file.save(file_path)
  
    img_path =  upload_dir + "/" + file_name

    return img_path

def get_image_paths(img_path):
    img_path = request.args.get("imgPath").replace("thumbnails", "animals").replace("\\", "/")
    img_full_path = root_dir + img_path

    return img_path, img_full_path

def check_if_valid_image(img_path):
    try:
      img = Image.open(img_path)
      img.verify() 
      
      return True, "no error"
    except (IOError, SyntaxError) as e:
      print('Bad file:', img_path)
      print(str(e))
      return False, "Sorry, Could not read the image"