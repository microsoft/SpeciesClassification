################
#
# api.py
#
# The external entry point for classification using the species classifier and detector.
#
################

import PIL
from models import *
from api import *

class PredictMode(Enum):

    classifyOnly = 0         # Do not run the detector at all
    classifyAndDetect = 1    # Run the detector and output a bounding box, but don't use it for classification
    classifyUsingDetect = 2  # Run the detector, and - if it produces a bounding box - use that box for classification


class SpeciesResults():

    def __init__(self):

        super(SpeciesResults, self).__init__()

        self.species = None
        self.species_scores = None
        self.bboxes = None
        self.bboxes_scores = None
        

class DetectionClassificationAPI():
    
    def __init__(self, classification_model_path, detection_model_path, image_sizes, useGPU=True):
        """
        The external entry point for loading models.
        
        classification_model path (required) and detection_model_path (optional) specify paths to PyTorch
        model files.  During training, classification_model path may be a list of models from which to re-start
        training; for inference, it should be a single path.

        image_sizes is a list of short-side image sizes to which images will be resized prior to inference.  The
        length of this list corresponds to the number of models in an ensemble, so knowledge of exactly what's inside
        classification_model_path is necessary.  I.e., if you're loading an ensemble of two models, this should be 
        two elements long.  At some point, this will get cleaned up and moved inside the model file.

        useGPU must be true if detection is enabled.
        """

        super(DetectionClassificationAPI, self).__init__()

        self.useGPU = useGPU

        self.classification_model = ClassificationModel(classification_model_path, image_sizes, useGPU)
        if detection_model_path != None:
            assert useGPU, 'CPU-based detection is not currently supported'
            self.detection_model = Detector(detection_model_path, useGPU)
        else:
            self.detection_model = None
    
    def predict_image(self, image_path, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
        """
        The main external entry point for inference.  image_path is a single filename.  
        
        This function just loads an image and calls predict_from_image().
        """
        
        test_image = PIL.Image.open(image_path).convert('RGB')
        return self.predict_from_image(test_image, topK, multiCrop, predict_mode)


    def predict_from_image(self, test_image, topK=1, multiCrop=False, predict_mode=PredictMode.classifyUsingDetect):
        """
        Runs inference on a single PIL image.
        """
        
        ret = SpeciesResults()

        classification_bboxes = []
        
        # ...if we're supposed to be computing bounding boxes
        if (predict_mode != PredictMode.classifyOnly):

            assert self.detection_model != None, 'Can''t run detection, no detector model loaded'
            numBoxes = 1

            # get bounding boxes
            bboxes, bboxes_scores = self.detection_model.predict_image(test_image, topk=numBoxes)
            ret.bboxes = bboxes
            ret.bboxes_scores = bboxes_scores

            # if using boxes set classifcation boxes
            if (predict_mode==PredictMode.classifyUsingDetect):
                classification_bboxes = bboxes

        species, species_scores = self.classification_model.predict_from_image(test_image, topK, multiCrop, bboxes=classification_bboxes)
        ret.species = species
        ret.species_scores = species_scores

        return ret
