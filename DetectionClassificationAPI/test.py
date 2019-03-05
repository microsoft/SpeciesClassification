################
#
# test.py
#
# Test driver for inatapi.py
#
################

from inatapi import *

def main():

    image_path = '/path/to/image.jpg'

    classification_model_path = '/path/to/model.pth.tar'
    detection_model_path = '/path/to/model'
    useGPU = True    

    createLabeledImage = True

    model = DetectionClassificationAPI(classification_model_path, detection_model_path, useGPU)  

    print('Working on image {}'.format(image_path))

    ret = model.predict_image(image_path, topK=5)

    print('\nSpecies \n')

    for i in range(0, len(ret.species)):
        print('%d) %s likelihood: %f' % (i+1, ret.species[i], ret.species_scores[i]))

    if (ret.bboxes is not None):
        print('\nBounding box \n')

        for i in range(0, len(ret.bboxes)):
            print('%d) %f %f %f %f: conf %f' % 
                  (i+1, ret.bboxes[i][0], ret.bboxes[i][1], ret.bboxes[i][2], ret.bboxes[i][3], ret.bboxes_scores[i]))

        # if viz draw box on image
        if (createLabeledImage):
            test_image = PIL.Image.open(image_path).convert('RGB')
            labeled_image = visdom_bbox(np.array(test_image).transpose((2, 0, 1)),
                            at.tonumpy(ret.bboxes[:,[1,0,3,2]]),
                            at.tonumpy([1 for _ in ret.bboxes]),
                            at.tonumpy(ret.bboxes_scores),
                            label_names=['Animal', 'BG'])

            labeled_image = PIL.Image.fromarray((255*labeled_image).transpose((1,2,0)).astype(np.uint8))
            labeled_image.save('output.jpg')

    print('\n')

if __name__ == '__main__':
    main()
