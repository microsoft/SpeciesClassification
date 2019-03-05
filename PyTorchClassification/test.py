################
#
# test.py
#
# Test driver for running inference with the ClassificationModel class in models.py.
#
################

from models import *

def main():

    image_path = '/path/to/image.jpg'
    model_path = '/path/to/model.pth.tar'

    model = ClassificationModel(model_path, useGPU = True)
    species, vals = model.predict_image(image_path, 3)

    for i in range(0, len(species)):
        print('%d) %s\tlikelihood: %f' % (i+1, species[i], vals[i]))

if __name__ == '__main__':
    main()
