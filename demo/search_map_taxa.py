import os
import json
import traceback
import numpy as np
import pickle
import pandas as pd
import csv

#Run this code file from console to create the pickle file
pklfile = "taxa_mapping.pkl"
root_path = "<add root path here>"
image_location = root_path + "result-img\\"
taxa_file_path = root_path + "\\data\\taxa.csv"
image_url = "/static/result-img/"

def create_taxa_mapping():
    
    try:
        taxa_info = pd.read_csv(taxa_file_path).dropna(subset=['vernacularName'])
        d = store_image_urls(taxa_info)
        print(d)
        
        with open(pklfile, 'wb') as f:
          pickle.dump(d, f)

        #return d
    except Exception as e:
        var = traceback.format_exc()
        print(str(var))
        return str(e) 

def store_image_urls(data):
    
    col_names = ['speciesCommonName', 'speciesScientificName', 
                'parentNameUsageID', 'genus' , 'genusCommonName', 
                'path', 'imageFound', 'searchRanking']

    lst = []
    
    for index, row in data.iterrows():
        
        s = row["scientificName"]
        s = s.replace(" ", "-")
        t = row["taxonRank"]
        image_found = False

        if(t.lower() == "species" or t.lower() == "subspecies"):

            d = data[data["taxonID"] == row["parentNameUsageID"]]
            
            path = image_url + "/image-not-available.jpg"

            if(os.path.isdir(image_location +  s)):
                files = os.listdir(image_location + s)
                
                if len(files) > 0 :
                    f = os.listdir(image_location + s)[0]
                    path = image_url + s + "/" + f,
                    image_found = True

            if(d is None or d.empty ):
                lst.append({'speciesCommonName' :row["vernacularName"], 
                        'speciesScientificName' : row["scientificName"], 
                        'parentNameUsageID' : row["parentNameUsageID"], 
                        'genus': ' ',
                        'genusCommonName': ' ',
                        'path': path,
                        'imageFound': image_found,
                        'searchRanking' : 0})  
            else:
                lst.append({'speciesCommonName' :row["vernacularName"], 
                            'speciesScientificName' : row["scientificName"], 
                            'parentNameUsageID' : row["parentNameUsageID"], 
                            'genus': d["scientificName"].iloc[0],
                            'genusCommonName': d["vernacularName"].iloc[0],
                            'path': path,
                            'imageFound': image_found,
                            'searchRanking' : 0})  
    
    r = pd.DataFrame(lst, columns=col_names)

    return r
    

if __name__ == "__main__":
  create_taxa_mapping()

