import imghdr
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from pathlib import Path
from tqdm import tqdm

from data_loader import ImageLoader
from models import ClassificationModel

# Input Image size to your model
IMAGE_SIZES = 488
# Folder containing the test images
TEST_FOLDER = 'data/round1'
MODEL_PATH = 'result/snakes/inc4_488/model_best.pth.tar'   # Path to your best model
# File to store submission results
SAVE_TO = 'inc4_488_test_result.csv'


def get_model(model_path):
    '''Takes the model path and returns the model.'''
    return ClassificationModel(
        model_path, image_sizes=IMAGE_SIZES, useGPU=True)


class TestDataset(data.Dataset):
    '''
    Filters the corrupted images and applies the validation transformation before
    returning the images.
    Input:
        folder      - name of the test folder
        image_sizes - input image size to the model
    '''

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
    '''Sorts the column names inplace, in alphabetic order as required by AICrowd.'''
    df = pd.read_csv(filename)
    cols = df.columns.tolist()
    cols = cols[:1] + sorted(cols[1:])
    df = df.reindex(columns=cols)
    df.to_csv(filename, index=False)


def fill_corrupted_files(filename, folder):
    '''Adds the corrupted images to the submission file with random scores, as required by AICrowd.'''
    df = pd.read_csv(filename)
    test_dir = Path(folder)
    test_imgs = [path.name for path in test_dir.iterdir()]
    corrupted_imgs = list(set(test_imgs) - set(df.filename))

    dummy_df = pd.DataFrame(
        np.ones((44, 45), dtype=np.float) / 90, columns=df.columns[1:])
    dummy_df.insert(loc=0, column='filename', value=corrupted_imgs)

    df = pd.concat([df, dummy_df])
    df.to_csv(filename, index=False)


def main():
    '''Create the submission file for `Snake Species Classification Challenge`.'''
    # Create the test data loader
    test_folder = TEST_FOLDER
    test_data = TestDataset(test_folder, IMAGE_SIZES)
    test_loader = data.DataLoader(test_data, batch_size=180, shuffle=False,
                                  num_workers=4, pin_memory=True)

    # Load the model
    model_path = MODEL_PATH
    model = get_model(model_path)
    model.eval()

    classnames = model.classnames
    # Set the device to GPU
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
    test_df.to_csv(SAVE_TO, index=False)


if __name__ == '__main__':
    print('Run the model on test set...\n\n')
    main()
    print('Sorting the columns...\n\n')
    sort_columns(SAVE_TO)
    print('Filling in the corrupted images...\n\n')
    fill_corrupted_files(SAVE_TO, 'data/round1')
    print('Done!')
