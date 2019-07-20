import sys
import site
 
site.addsitedir('/anaconda/envs/py35/lib/python3.5/site-packages')
sys.path.append('/add_demo_path_here/')

from app import app as application
