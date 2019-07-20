from enum import Enum


class PredictMode(Enum):
  
  classifyOnly = 0        # Do not run the detector at all
  classifyAndDetect = 1    # Run the detector and output a bounding box, but don't use it for classification
  classifyUsingDetect = 2 

class PredictType(Enum):
  
  sampleImages = 0
  uploadedFile = 1
  fromURL = 2
