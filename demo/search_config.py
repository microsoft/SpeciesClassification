import pandas as pd
import numpy as np
import configparser
import constant as c
import os

config = configparser.ConfigParser()
config.read('config.py')
config.sections()

class Search_config:
    
    #def __init__(self):
       #self.root_path = self.get_root_path()
    
    def get_index_col_names(self):

        col_names = [c.SPECIES_COMMON_NAME, 
                    c.SPECIES_SCIENTIFIC_NAME, 
                    c.PARENT_NAME_USAGE_ID, c.GENUS, 
                    c.GENUS_COMMON_NAME, 
                    c.PATH, 
                    c.SEARCH_RANKING]
                    
        return col_names

    def get_root_path(self):
        return config[c.COMMON][c.ROOT_PATH]

    def get_index_file_path(self):
        return config[c.SEARCH][c.SEARCH_INDEX_PATH]

    def get_image_location(self):
        return self.root_path + config[c.SEARCH][c.SEARCH_IMAGE_PATH]

    def get_taxa_file_path(self):
        return self.root_path + config[c.COMMON][c.TAXA_FILE_PATH]

    def get_img_rel_url(self):
        return config[c.SEARCH][c.SEARCH_IMG_REL_URL]



  
    