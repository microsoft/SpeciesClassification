################
#
# classify_image.py
#
# Test driver for running inference with the ClassificationModel class in models.py.
#
################

#%% Imports

from models import ClassificationModel
import argparse
import sys


#%% Functions

def run_classification_model(model,image_path):
    
    # [560, 560] for ensemble, [560] or [448] for single models like ResNeXt
    model_input_sizes = [560,560]

    if isinstance(model,str):
        model = ClassificationModel(model, model_input_sizes, useGPU=True)
        
    species, vals = model.predict_image(image_path, 3)

    for i in range(0, len(species)):
        print('%d) %s\tlikelihood: %f' % (i+1, species[i], vals[i]))
    
    return model


#%% Command-line driver
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Model filename')
    parser.add_argument('image_path', type=str, help='Image filename')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    run_classification_model(args.model_path,args.image_path)

    
if __name__ == '__main__':
    
    main()


#%% Interactive driveer
    
if False:

    #%%
    
    model = None
    
    #%%    
    
    model_path = '/data/species_classification/sc_all_extended_ensemble_resnext_inceptionV4_560_2019.09.19_model.pytorch'
    # image_path = '/data/species_classification/meerkat.jpg'
    # image_path = '/data/species_classification/coyote.jpg'
    image_path = '/data/species_classification/elephant.jpg'
    model = run_classification_model(model_path,image_path)
    