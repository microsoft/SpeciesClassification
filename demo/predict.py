import json
import os
import sys
import uuid
import requests
import config
from PIL import Image
from bbox import Bbox
from flask import request
from search import Search
from io import StringIO
from flask import send_file
from enums import*
from common import*
from common import str2bool

try:
	from urllib.parse import urlparse
except ImportError:
	 from urlparse import urlparse

bbox = Bbox()

root_dir = config.ROOT_PATH
sample_dir = config.SAMPLE_PATH
upload_dir = config.UPLOAD_PATH
show_bbox = config.SHOW_BBOX

CONTENT_TYPE_KEY = config.CONTENT_TYPE_KEY
CONTENT_TYPE = config.CONTENT_TYPE 
SUBSCRIPTION_KEY = config.SUBSCRIPTION_KEY
AUTHORIZATION_HEADER = config.AUTHORIZATION_HEADER
BASE_URL = config.BASE_URL
API_VERSION = config.API_VERSION

classify_format = '{0}/species-recognition/v{1}/predict?topK={2}&predictMode={3}'

max_file_size = config.MAX_FILE_SIZE

class Predict:
  def build_classify_url(self, topK=5, base_url=BASE_URL, version=API_VERSION, 
                         predictMode="classifyOnly"):
    return classify_format.format(base_url, version, topK, predictMode)

  def get_api_headers(self, content_type):
    return { CONTENT_TYPE_KEY: content_type, AUTHORIZATION_HEADER: SUBSCRIPTION_KEY }

  def get_api_response(self, img, predictMode):
    url = self.build_classify_url(predictMode=predictMode)
    r = requests.post(url, headers= self.get_api_headers(CONTENT_TYPE), data=img) 
    
    '''Error'''
    if(r.status_code != 200):
      return r.json(), True
    
    return r.json(), False
   
  def get_prediction(self, type, img_full_path=None, img_path=None, showbboxUI=None):
    error_message = ''
   
    predictMode = PredictMode.classifyOnly
    data = None

    if(type == PredictType.sampleImages):
      data, img_path, has_error = self.get_prediction_sampleImg(
                                  img_path, img_full_path, predictMode)
           
    if(type == PredictType.uploadedFile):
      data, img_path, has_error, error_message  = self.get_prediction_uploadedFile(
                                                  img_path, img_full_path, predictMode)
    
    return data, img_path, has_error, error_message

  def get_prediction_sampleImg(self, img_path, img_full_path, predictMode):
    img = open(img_full_path, mode='rb').read()
    data, has_error = self.get_api_response(img, predictMode=predictMode.name)
    
    if(not has_error):
      if(predictMode == PredictMode.classifyAndDetect):
        img_path = bbox.draw_bbox(data, file=None, file_path=img_full_path)
        return data, img_url, has_error

    img_path = img_path.replace(root_dir, '')
    
    return data, img_path, has_error
  
  def get_prediction_uploadedFile(self, img_path, img_full_path, predictMode=PredictMode.classifyOnly):
    data = None
    has_error = False
    error_message = ''
    
    valid_img, error_message = check_if_valid_image(img_full_path)
    
    if not valid_img:
      has_error = True
      return data, img_path, has_error, error_message

    file = open(img_full_path, mode='rb')
    img = file.read()

    data, has_error = self.get_api_response(img, predictMode=predictMode.name)
 
    if(not has_error):
      if(predictMode != PredictMode.classifyOnly):
        img_path = bbox.draw_bbox(data, file_path=img_path)
        return data, img_path, has_error, error_message
    
    temp = img_path.split('/')
    file_part = temp[len(temp)-1].split("\\")
    img_path = upload_dir + "/" + file_part[len(file_part)-1]
    
    return data, img_path, has_error, error_message