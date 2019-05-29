import imghdr
import pandas as pd
import numpy as np
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


def sort_columns(filename):
    df = pd.read_csv(filename)
    cols = df.columns.tolist()
    cols = cols[:1] + sorted(cols[1:])
    df = df.reindex(columns=cols)
    df.to_csv(filename, index=False)
    

def fill_corrupted_files(filename, folder):
    df = pd.read_csv(filename)
    test_dir = Path(folder)
    test_imgs = [path.name for path in test_dir.iterdir()]
    corrupted_imgs = list(set(test_imgs) - set(df.filename))
    
    dummy_df = pd.DataFrame(np.ones((44, 45), dtype=np.float)/90, columns=df.columns[1:])
    dummy_df.insert(loc=0, column='filename', value=corrupted_imgs)
    
    df = pd.concat([df, dummy_df])
    df.to_csv(filename, index=False)


def main():
    # create the test loader
    test_folder = 'data/round1'
    test_data = TestDataset(test_folder, 224)
    test_loader = data.DataLoader(test_data, batch_size=300, shuffle=False,
                                  num_workers=8, pin_memory=True)

    # load the model
    model_path = 'result/snakes/resnext_224/model_best.pth.tar'
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
    # main()
    sort_columns('test_result.csv')
    fill_corrupted_files('test_result.csv', 'data/round1')
