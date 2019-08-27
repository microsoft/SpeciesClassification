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
    # [560, 560] for ensemble, [560] or [448] for single models like ResNeXt
    model_input_sizes = [560]

    model = ClassificationModel(model_path, model_input_sizes, useGPU = True)
    species, vals = model.predict_image(image_path, 3)

    for i in range(0, len(species)):
        print('%d) %s\tlikelihood: %f' % (i+1, species[i], vals[i]))

if __name__ == '__main__':
    main()
