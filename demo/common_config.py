import pandas as pd
import numpy as np
import configparser
import constant as c
import os

config = configparser.ConfigParser()
config.read('config.py')
config.sections()

class Common_config:
    
    def get_root_path(self):
        return config["COMMON"]["ROOT_PATH"]

    def get_sample_img_path(self):
        return config["COMMON"]["SAMPLE_PATH"]

    def get_upload_path(self):
        return config["COMMON"]["UPlOAD_PATH"]

    def get_sample_img_url(self):
        return config["COMMON"]["SAMPLE_URL"]
    
    def show_bbox(self):
        return config["COMMON"]["SHOW_BBOX"]



    

  
    