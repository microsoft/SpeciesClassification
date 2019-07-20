import os
import uuid
from PIL import Image, ImageDraw
from common_config import Common_config
import base64
from io import BytesIO

config = Common_config()
root_direc = config.get_root_path()

class Bbox:
  
    def draw_bbox(self, data, file=None, file_path=None):     
        
        img_path = ""

        if(len(data['bboxes']) > 0):
            x0,y0,x1,y1 = self.get_bbox_coordinates(data)
        else:
            return img_path

        img = file
        if(img is None):
            img = Image.open(file_path)
        
        width, height = img.size
        initial_outline_width = 3
        outline_width = 0

        if(width or height > 100):
        	if(width > height):
        		outline_width = int(width / 100)
        	else:
        		outline_width = int(height / 100)
        if(outline_width < initial_outline_width):
        	outline_width = initial_outline_width
        
        img_format = img.format
        if(img_format == "gif"):
            img_format = "PNG"
        
        draw = ImageDraw.Draw(img)
        for i in range(outline_width):
        	draw.rectangle([(x0+i, y0+i),(x1+i,y1+i)], outline="red")

        new_file_name =  str(uuid.uuid4()) + "." + img_format
        img_path = root_direc + "/static/upload/" + new_file_name
        img.save(img_path, img_format)

        del draw

        img_url = "/static/upload/" + new_file_name
        return img_url

    def get_bbox_coordinates(self,data):

        x_min = data['bboxes'][0]['x_min']
        y_min = data['bboxes'][0]['y_min']
        x_max = data['bboxes'][0]['x_max']
        y_max = data['bboxes'][0]['y_max']  
        
        return x_min, y_min, x_max, y_max   
    
    
    

    
    