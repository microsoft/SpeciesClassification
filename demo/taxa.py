import pandas as pd
import numpy as np

class Taxa:
    def __init__(self, mapping_path = 'static/data/taxa.csv'):
        self.mapping = pd.read_csv(mapping_path)
        self.mapping['vernacularName'] = self.mapping['vernacularName'].fillna('')

    def get_data(self):
        return self.mapping
    
    def get_species_details(self, scientific_name):
        species = self.mapping[self.mapping['scientificName'].str.match(scientific_name,False)]

        details = dict()
        if len(species) < 1:
            return details
        
        taxa = species.iloc[0]
        while not np.isnan(taxa['parentNameUsageID']):
            details[taxa['taxonRank']] = taxa['scientificName']
            details[taxa['taxonRank'] + '_common'] = taxa['vernacularName']

            taxa = self.mapping.loc[taxa['parentNameUsageID'].astype('int64')]
        return details
