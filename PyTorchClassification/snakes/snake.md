# Snake Species Identification Challenge

<p align="center"><img src="nbs/0ea412abd5014df4ecacc804d5907cb0.jpg" alt="Opheodrys vernalis"/></p>

## Overview

This repo was used to enter the [AIcrowd Snake Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge); an entry based on this repo placed first in the first qualifying round of the competition.  The competition aims to stimulate the development of a snake species identification application, so bite victims and health practitioners can prioritize care for potentially-harmful bites.  For the qualifying round, the competition provided ~82k images and ~18k test images covering 45 species.

## How this repo was used

The competition entry built on the core repo tools; new code was written to:

- Converted the competition data set to COCO format, for compatibility with the training code
- Call into the existing code to train both ResNeXt and Inceptionv4 networks (around 80 epochs)
- Aggregate results from the ResNeXt and Inceptionv4 models (post-hoc averaging)
- Run inference on the test data and prepare a submission for the competition

This approach yielded an F1 of `0.809` for `Inceptionv4` and `0.804` for `ResNext101`.  The averaged predictions achieved an F1 of `0.846`, which placed first in the qualifying round of the competition.

## Steps to replicate the results

- Follow the steps in [README.md](README.md) to create the required docker or conda environment.  
- Download the training and test data from [here](https://www.aicrowd.com/challenges/snake-species-identification-challenge/dataset_files)
- Unzip the training and test zipfiles into a folder called "data" in the PyTorchClassification directory, or symlink a directory called `data` to point to your data directory.  When you unzip the training data, images should end up in `data/train` (e.g. `data/train/[class]/[filename].jpg`).  Test data should end up in `data/round1/[filename].jpg`.


- Run the following commands:

```
# cd into the PyTorchClassification directory
# Make sure this directory is also on your Python Path
PYTHONPATH=${PYTHONPATH}$PWD:
cd snakes
python folder_to_coco.py          # Creates the COCO annotation format for the dataset
python run_snakes_training.py     # Trains the model for ResNext101 and Inceptionv4 architectures
python test_snakes.py             # Generates the prediction result on the test dataset
python merge_snakes_results.py    # Merges the results by different models
```