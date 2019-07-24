import os
import traceback
import pickle
import numpy as np
import sys
import uuid
import requests
import config
import constant as c

from PIL import Image
from search import Search
from flask import json
from enums import*
from common import*
from io import StringIO
from flask import send_file
from predict import Predict
from sample_images import Sample_images

search = Search()
predict = Predict()
sample_images = Sample_images()

root_dir = config.ROOT_PATH
show_bbox = config.SHOW_BBOX

from flask import Flask, render_template, request, jsonify, make_response, url_for, g
app = Flask(__name__, static_url_path = "/static", static_folder = "static")


@app.route('/')
@app.route('/index.html')
def index():
    try: 
      return render_template('/explore.html')  
    except Exception as e:
        var = traceback.format_exc()
        return str(var)

@app.route('/showbbox')
def get_showbbox_option():
    try:          
        return show_bbox                                         
    except Exception as e:
        var = traceback.format_exc()
        return str(var) 

@app.route('/explore', methods=['GET'])
@app.route('/explore.html', methods=['GET'])
def explore():
    try:          
        return render_template('/explore.html')                                                                                                                                                                                                      
    except Exception as e:
        var = traceback.format_exc()
        return str(var)

@app.route('/about', methods=['GET'])
@app.route('/about.html', methods=['GET'])
def about():
    try:
        return render_template('/about.html')                                                                                                                                                                                                      
    except Exception as e:
        var = traceback.format_exc()
        return str(var)

def abspath(file):
    """Return abs path of folder."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        
def get_all_images():
    file_path = 'static/images/gallery'
    img_list = os.listdir(file_path)
    return jsonify(img_list)

def get_image_path():
  img_path = str(request.get_data())
  img_path = img_path.replace("b'", "")
  img_path = img_path.replace('b"', "")
  img_path = img_path.replace("/", "\\")
  img_path = img_path.replace("'", "")

  return img_path

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    img_path = save_posted_file(file)
    has_error = False
   
    return jsonify({'error': has_error, 'img_path': img_path})

@app.route('/upload_file_from_url', methods=['GET'])
def upload_file_from_url():
    try:
        url = request.args.get("url")
        url = get_parsed_url(url)  

        has_error = False
        
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'}

        r = requests.get(url, headers=headers, allow_redirects=True, verify=False)
            
        img_path = save_url_img(url, r.content) 

        return jsonify({'error': has_error, 'img_path': img_path})

    except Exception as e:
        var = traceback.format_exc()
        print(str(var))
        return jsonify({'data' : None, 'error': True, 'error_message': str(var)})

@app.route('/checkfile_size_dimensions', methods=['GET'])
def checkfile_size_dimensions():
    has_error = False

    img_path, img_full_path = get_image_paths(request.args.get("imgPath"))
    valid_img, error_message = check_if_valid_image(img_full_path)
    
    ok = True
    if not valid_img:
        ok = False
        has_error = True
        return jsonify({'ok' : ok, 'error': has_error, 'img_path': img_path, 'error_message': error_message})
    
    file = open(img_full_path, mode='rb')
    
    if(is_largeimage_size(file) or has_large_dimensions(file)):
        ok = False

    return jsonify({'ok' : ok, 'error': has_error, 'img_path': img_path})

@app.route('/resize_image_file', methods=['GET'])
def resize_image_file():
    img_path, img_full_path = get_image_paths(request.args.get("imgPath"))
    img_path = resize_image(img_path, img_full_path)

    has_error = False

    return jsonify({'error': has_error, 'img_path': img_path})

@app.route('/get_sample_image_prediction', methods=['GET'])
def get_sample_image_prediction():
    try:
        img_path, img_full_path = get_image_paths(request.args.get("imgPath"))
        show_bbox_UI = request.args.get("showbbox")

        data, img_path, has_error, error_message = predict.get_prediction(
                                                   PredictType.sampleImages, 
                                                   img_full_path, img_path, 
                                                   show_bbox_UI)
                                                                          
        return jsonify({'data' : data, 'error': has_error, 'img_path': img_path})
        
    except Exception as e:
      var = traceback.format_exc()
      print(str(var))
      return jsonify({'data' : None, 'error': True, 'error_message': str(var)})

@app.route('/get_image_prediction_uploadedfile', methods=['GET'])
def get_image_prediction_uploadedfile():
  try:
    img_path, img_full_path = get_image_paths(request.args.get("imgPath"))
    show_bbox_UI = request.args.get("showbbox")
    data, img_path, has_error, error_message = predict.get_prediction(
                                               PredictType.uploadedFile, 
                                               img_full_path, 
                                               img_path, 
                                               show_bbox)

    return jsonify({'data' : data, 'error': has_error, 
                    'img_path': img_path, 
                    'error_message': error_message})
   
  except Exception as e:
    var = traceback.format_exc()
    print(str(var))
    
    return jsonify({'data' : None, 'error': True, 
                   'error_message': str(var)})

@app.route('/get_image_prediction_url', methods=['GET'])
def get_image_prediction_url():
    try:
      data, img_path, has_error, error_message = predict.get_prediction(PredictType.fromURL)
      
      return jsonify({'data' : data, 'error': has_error, 'img_path': img_path})
    
    except Exception as e:
      var = traceback.format_exc()
      print(str(var))
      return jsonify({'data' : None, 
                      'error': True, 
                      'error_message': str(var)})

@app.route('/get_images', methods=['GET'])            
def get_images(): 
  try: 
      add_more = str2bool(request.args.get("addmore")) 
      
      if(sample_images.total_num_images < 7):
        print("ERROR there are only {} valid images in the animals folder, atleast 8 is required"
              .format(str(self.total_num_images)))
      
      img_data = sample_images.get_images_data(add_more)
      
      return jsonify(img_data)
      
  except Exception as e:
      var = traceback.format_exc()
      print (str(var)) 
      return str(var)     

@app.route('/get_search_results', methods=['GET'])     
def get_search_results():
    try:
        search_string = request.args.get("searchString")
        search.do_search(search_string)
        result = search.result.fillna(' ')

        row_count = result.shape[0]
        if(row_count >= 8):
          search_result = result.iloc[0:8].to_dict(orient='records')
        else:
          search_result = result.iloc[0:(row_count)].to_dict(orient='records')
        
        return jsonify(search_result)

    except Exception as e:
      print(str(e))
      return str(e)

@app.route('/get_more_search_images', methods=['GET'])
def get_more_search_images():
    try:
        start_range = int(request.args.get("startRange")) + 4
        end_range = start_range + 4
        
        result = search.result.fillna(' ')#.to_dict(orient='records')
        search_result = result.iloc[start_range:end_range].to_dict(orient='records')

        return jsonify({'data': search_result, 'end_range' : end_range})

    except Exception as e:
      print(str(e))
      return str(e)

@app.route('/check_image_url', methods=['GET'])
def check_image_url():
    error = ""

    try:
        from urllib.parse import urlparse
        import requests

        url = request.args.get('url').strip()
        parsed_url = urlparse(url)
        url = parsed_url.geturl()
        
        header = requests.head(url, allow_redirects=True)
        print(header.status_code)       
        if(h.status_code != 200):
          error = "The image could not be downloaded"
          return jsonify({'error' : True, 'error_message': error})

        content_type = header.headers['content-type']
        if(content_type.lower().find("image") == -1):
          error = "not a valid image URL"
          return jsonify({'error' : True, 'error_message': error})


        content_size = header.headers['content-length']
        if(int(content_size)  > 10000000):
          error = "Image size of " + content_size  + " bytes is > 10MB"
          return jsonify({'error' : True, 'error_message': error})
        
        header.raise_for_status()

        if(int(content_size)  > 4000000):
          return jsonify({'large_size': True, 'error' : False, 'size': content_size})          

    except requests.exceptions.HTTPError as errh:
      error = "Http Error:" + errh
      print(error)
    except requests.exceptions.ConnectionError as errc:
      error = "Error Connecting:" + errc
      print(error)
    except requests.exceptions.Timeout as errt:
      error = "Timeout Error:" + errt
      print(error)
    except requests.exceptions.RequestException as err:
      error = "An error occurred while retrieving image: " + err
      print(error)

    return jsonify({'error' : False, 'error_message': ""})


if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0")