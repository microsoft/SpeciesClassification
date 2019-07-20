import os
import future
import random
import pandas as pd
from common_config import Common_config

config = Common_config()
root_direc = config.get_root_path()

class Sample_images:

  def __init__(self):

    self.list = self.shuffle_animal_list()
    self.start_range = 0
  

  def shuffle_animal_list(self):
    
    d =  pd.read_csv(root_direc + "/static/data/updated_animal_list.csv")
    print(d)
    d = d.sample(frac=1)
    return d

  def get_images_data(self, add_more):
    
    end_range = self.start_range + 8
    if(not add_more):
      end_range = self.start_range + 12 

    img_list = []

    for i in range(self.start_range , end_range):
      img_data = self.list.iloc[i]
      
      img_list.append({
          'Path': img_data["Path"].lower(), 
          'Name' : img_data["Name"].title() 
        })
    
    self.start_range = end_range + 1
    
    return img_list

 

    
   
    
    