import imghdr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch.utils.data as data
from data_loader import ImageLoader
from models import *


def get_model(model_path):
    return ClassificationModel(model_path, image_sizes=224, useGPU=True)


class TestDataset(data.Dataset):
    def __init__(self, folder, image_sizes):
        super().__init__()
        self.image_paths = [path for path in Path(folder).iterdir()
                            if imghdr.what(path)]
        self.loader = ImageLoader([image_sizes])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im_name = self.image_paths[idx].name

        raw_image = self.loader.load_image(self.image_paths[idx])
        imgs = self.loader.process_image(raw_image, is_train=False,
                                         multi_crop=True)
        return imgs, im_name


def main():
    # create the test loader
    test_folder = 'data/train/class-4'
    test_data = TestDataset(test_folder, 224)
    test_loader = data.DataLoader(test_data, batch_size=100, shuffle=False,
                                  num_workers=4, pin_memory=True)

    # load the model
    model_path = 'result/snakes/resnext_224_init/model_best.pth.tar'
    model = get_model(model_path)
    model.eval()

    classnames = model.classnames
    device = torch.device('cuda')

    result = []

    with torch.no_grad():
        for i, (inputIn, im_name) in tqdm(enumerate(test_loader),
                                          total=len(test_loader)):
            for j in range(0, len(inputIn)):
                input = inputIn[j].to(device)
                outputNew = model(input)
                if j == 0:
                    output = outputNew
                else:
                    output = output + outputNew
            output /= len(inputIn)
            output = output.detach().cpu().numpy()

            result.extend([[name] + scores.tolist()
                           for name, scores in zip(im_name, output)])

    test_df = pd.DataFrame(result,
                           columns=['filename'] +
                           list(map(lambda x: x[1], sorted(classnames.items(),
                                                           key=lambda x: x[0]))))
    test_df.to_csv('test_result.csv', index=False)


if __name__ == '__main__':
    main()
