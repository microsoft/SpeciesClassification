import os
import json
import traceback
import numpy as np
import pickle
import pandas as pd
import csv
import configparser
import constant as c
from search_config import Search_config
from taxa import Taxa

config = Search_config()
t = Taxa()

def create_taxa_mapping():
    
    try:
        taxa_info = t.get_data()
        d = taxa_info.sort_values(c.CSV_SCIENTIFIC_NAME,ascending=False)
        d = drop_cols_with_no_images(d)

        with open(config.get_index_file_path(), 'wb') as f:
          pickle.dump(d, f)

        print(d.head())

    except Exception as e:
        var = traceback.format_exc()
        print(str(var))
        return str(e) 

def drop_cols_with_no_images(data):
    lst = []
    for index, row in data.iterrows():
        
        t = row[c.CSV_TAXON_RANK].lower().strip()
         
        if(t not in ["species", "subspecies"]):
            continue

        scientific_name = row[c.CSV_SCIENTIFIC_NAME].replace(" ", "_")
        start_letter = scientific_name[0]
        direc = config.get_image_location() + "/" + start_letter

        if os.path.isdir(direc) == False :
            continue

        img_url = config.get_img_rel_url() 
        contains_images = False
        for file in os.listdir(direc):
            file_name = file.split("-")[0]
            if file_name == scientific_name:
                img_url = img_url + start_letter + "/" + file
                contains_images = True
                break
                
        if(contains_images):

            genus = ' '
            genus_common_name = ' '
            
            d = data[data[c.CSV_TAXON_ID] == row[c.CSV_PARENT_NAME_USAGE_ID]]
            
            if(len(d) > 0):
                genus = d[c.CSV_SCIENTIFIC_NAME].iloc[0]
                genus_common_name = d[c.CSV_VERNACULAR_NAME].iloc[0]

            lst.append({
                   c.SPECIES_COMMON_NAME :row[c.CSV_VERNACULAR_NAME], 
                    c.SPECIES_SCIENTIFIC_NAME : row[c.CSV_SCIENTIFIC_NAME], 
                    c.PARENT_NAME_USAGE_ID : row[c.CSV_PARENT_NAME_USAGE_ID], 
                    c.GENUS : genus,
                    c.GENUS_COMMON_NAME : genus_common_name,
                    c.PATH : img_url,
                    c.SEARCH_RANKING : 0})
            
    r = pd.DataFrame(lst, columns=config.get_index_col_names())
    return r


if __name__ == "__main__":
  create_taxa_mapping()

