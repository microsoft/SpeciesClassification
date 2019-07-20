import os
import re
import pandas as pd
import numpy as np
import configparser
import constant as c

from common_config import Common_config
from search_config import Search_config

config = Common_config()
root_direc = config.get_root_path()

NAME = "Name"
SEARCH_RANKING	= "Search_ranking"

class Search:
  
    def __init__(self, list_path = root_direc + '/static/data/updated_animal_list.csv'):
        self.data = pd.read_csv(list_path)
        self.config = Search_config()
        self.result = None

    
    def do_search(self, search_string):
        
        s = search_string.lower()
        r = self.contains(s)
        r = self.add_ranking(s, r)
        
        self.result = r

        print(r.head())
        r = r.apply(lambda x: x.astype(str).str.lower())
        return r
    
    def contains_any_word(self, search_string):
        d = self.data
        regx_words_string = self.build_search_words(search_string)
        print(regx_words_string)
        r  = d[d[NAME].str.lower().str.contains(regx_words_string , case=False)]
        return r

    def contains(self, search_string):
        d = self.data
        s = search_string
        r  = d[d[NAME].str.lower().str.contains(s , case=False)]
        return r
     
    def add_ranking(self, search_string, search_result):
        
        s = search_string
        r = search_result
        r[SEARCH_RANKING] = 0
        words = s.split(' ')

        for index, row in r.iterrows():
            common_name = row[NAME].lower()
            ranking = int(row[SEARCH_RANKING])
            #print(common_name.find(s))

            if(common_name.find(s) != -1):
                ranking = ranking + 1
            for w in words:
                if(common_name.find(w) != -1):
                    ranking = ranking + 1
                word = "(^|\s)" + w.lower() + "($|\s)"
                if(bool(re.search(word, common_name))):
                    ranking = ranking + 1
                last_word = words[len(words) - 1]
                c_w = common_name.split(' ')
                common_name_last_word = c_w[len(c_w)-1]
                if(last_word == common_name_last_word):
                    ranking = ranking + 1
                if(common_name == s):
                    ranking = ranking + 1
            r.at[index, SEARCH_RANKING] = str(ranking)

        r = r.sort_values(SEARCH_RANKING,ascending=False)
        return r
     
    def build_search_words(self, search_string):
    
        words = search_string.split(" ")
        count = 1
        word_length = len(words)
    
        word_search_string = ""

        for w in words:
        
            #if(count > 1):
                #word_search_string += "|"

            word_search_string += "(\W*("
            word_search_string += w
            word_search_string += ")\W*)"
                
            count += 1
                    
        return word_search_string
