
import matplotlib.pyplot as plt

from models import *

import rgb_nir_loader

import train

def main():
    args = train.Params()

    path = 'E:/Research/Images/iNat2017/images/train_val_images/Mammalia/Marmota flaviventris/03b2fc1c967a5965643e4066599e3752.jpg'
    model_path = './model_best.pth_128.tar'
    
    model = IRtoRGB(model_path, (args.sz, args.sz), args.num_layers, args.feats_in, args.feats_out, args.feats_max, False)
    img = model.predict_image(path)

    plt.imshow(img)
    plt.gray()
    plt.show()

if __name__ == '__main__':
    main()