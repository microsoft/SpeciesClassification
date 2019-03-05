import os
import sys
sys.path.append('./../')
import api
import PIL
import tqdm
import glob
import uuid
import subprocess
import shutil

checkpoint = '/path/to/checkpoint'
infile = 'input_video.mp4'
outfile = os.path.splitext(infile)[0] + '_annotated.mp4'
tmp_dir = str(uuid.uuid4())

framerate = subprocess.check_output(['ffprobe', '-v', '0', '-of' ,'csv=p=0', '-select_streams', '0', \
                                     '-show_entries', 'stream=r_frame_rate', infile]).decode("utf-8") 
framerate = framerate.split('/')[0]
print('Input video has frame rate', framerate)

assert os.path.isfile(infile), 'Could not find video file ' + infile
starting_dir = os.getcwd()
os.makedirs(tmp_dir, exist_ok=True)

print('Extracting images from video...')
# For scaling, add  ['-vf', 'select=\'\',scale=800:-1']
subprocess.check_call(['ffmpeg', '-i', infile, '-qscale:v', '1', os.path.join(tmp_dir, 'frame%06d.jpg')])

print('Loading model...')
det = api.Detector(checkpoint, True, 2)
all_images = list(sorted(glob.glob(tmp_dir + '/*.jpg')))
print('Processing images...')
for img_file in tqdm.tqdm(all_images, total=len(all_images)):
    in_img = PIL.Image.open(img_file)
    out_img = det.annotate_image(in_img, 1000)
    out_img.save(os.path.join(tmp_dir, 'ann_' + os.path.basename(img_file)))

subprocess.check_call(['ffmpeg', '-framerate', '10', '-pattern_type', 'glob', '-i', os.path.join(tmp_dir, 'ann_*.jpg'), \
                             '-c:v', 'libx264', '-r', '30', '-pix_fmt', 'yuv420p', outfile])

shutil.rmtree(tmp_dir)
