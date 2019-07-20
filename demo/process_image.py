import os
import sys
import uuid
import common
import traceback
import numpy as np
from PIL import Image
from enums import*
from common import*
from common_config import Common_config
from common import str2bool, delete_all_uploaded

config = Common_config()
root_direc = config.get_root_path()

max_file_size = 3750000

class Process_Image:
  
  def __init__(self, type, file_path):
    
    self.file_path = file_path
    self.file = open(file_path, mode='rb')
    
   
  def get_processed_file(self):

    if(self.has_large_dimensions() or self.is_largeimage_size()):
      self.resize_image_bydimension()


  def resize_image(self):

    image = Image.open(file)

    width = image.size[0]
    height = image.size[1]
    
    file_size = None

    if(width < height and width > 1600):
      self.save_image(image, 1600, height)
      file_size = sys.getsizeof(img_bytes)
      
    elif(height < width and height > 1600):
      self.save_image(image, width, 1600)
      file_size = sys.getsizeof(img_bytes)
      
    if(not file_size is None):

      while (file_size > max_file_size):
          
          next_width = width - 150
          next_height = next_width
          img_bytes, file_path = self.save_image(image, next_width, next_height)
          file_size = sys.getsizeof(img_bytes)
          width = next_width
  

  def is_largeimage_size(self):
    
    img_bytes = self.file.read()
    file_size = sys.getsizeof(img_bytes)

    if(file_size < max_file_size):
      return False
    return True


  def has_large_dimensions(self):
    
    im = Image.open(self.file)
    width, height = im.size
    
    if(width > 1600):
      return True
    if(height > 1600):
      return True


  def save_image(self, image, width=0, height=0):

    file_path = os.path.join(root_direc, "static", "upload", 
                      str(uuid.uuid4()) + "." + image.format)
  
    '''thumbnail will maintain aspect ratio'''
    image.thumbnail((width, height), Image.ANTIALIAS)
    image.save(file_path, format=image.format)
    
    self.file = open(file_path, mode='rb')


  def is_largeimage_size(self,file):
    
    img_bytes = file.read()
    file_size = sys.getsizeof(img_bytes)

    if(file_size < max_file_size):
      return False
    return True


  def has_large_dimensions(self, file):
    
    im = Image.open(file)
    width, height = im.size
    
    if(width > 1600):
      return True
    if(height > 1600):
      return True
