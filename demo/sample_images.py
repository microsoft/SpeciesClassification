import os
import random
import common
import config
import pandas as pd

root_dir = config.ROOT_PATH

animal_list = os.path.join(root_dir, "static/data/updated_animal_list.csv")

class Sample_images:

  def __init__(self):
    self.list = self.shuffle_animal_list()
    self.total_num_images = len(self.list)
    self.start_range = 0
  
  def shuffle_animal_list(self):
    d =  pd.read_csv(animal_list)
    d = d.sample(frac=1)
    return d

  def get_images_data(self, add_more):
    end_range = self.start_range + 8
    
    if(end_range > len(self.list)):
        if(add_more):
            return None
        
        self.start_range = 0
        end_range = self.start_range + 8 

    img_list = []

    for i in range(self.start_range , end_range):
      img_data = self.list.iloc[i]
      
      img_list.append({
          'Path': img_data["Path"], 
          'Name' : img_data["Name"].title() 
        })
    
    self.start_range = end_range + 1
    
    return img_list

 

    
   
    
    