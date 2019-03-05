# PyTorch-based classification framework

This repository implements the training and testing of a PyTorch image classification model. It is based on the [ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) from the PyTorch codebase and adapted for multi-GPU and half precision training using the [Nvidia apex](https://github.com/NVIDIA/apex) library. 

Is also extends the original code with tensorboard-compatible logging, text-based logging, code backup, a meta-script for configuring the training of multiple models, export to the ONNX format, new loss functions, taxonomic label smoothing, and balanced sampling.

The code is developed for reading datasets with a JSON-based annotation in the format of the [iNaturalist competition dataset](https://github.com/visipedia/inat_comp/tree/master/2017).

## Installation
We need a few libraries set up. You can either use docker or anaconda.


### Variant A: Docker instructions

To build the docker image, execute in the root directory of this repository:

    cd docker/ && sudo docker build -t cls -f Dockerfile . && cd ..

Then run docker image, again in the root directory of this repository:

    nvidia-docker run -it -v /data:/data -v $(pwd):/code --ipc=host cls zsh

This maps the local folder /data to the folder /data in the docker container and the current work directory to /code. You might want to adjust these path such that your dataset is accessible in the docker container.

**If you get a build error…**

I had an image that didn't work for me on Ubuntu 16, so I followed the steps here:

    https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce

…to update docker.

### Variant B: Conda

The file `docker/Dockerfile` shows how to setup the conda environment. You need the following libraries to run the code:
- python 3.5+ 
- numpy
- pytorch
- pretrainedmodels
- pyyaml
- scipy
- tqdm
- tensorboardX
- scikit-learn
- https://github.com/pytorch/tnt
- https://www.github.com/nvidia/apex

If you want to see tensorboard visualizations, then you need to install tensorboard as well.

## Preparing the dataset

The code expects the dataset to have a JSON-based annotation in the format of the [iNaturalist competition dataset](https://github.com/visipedia/inat_comp/tree/master/2017). Please refer to this repository for a guideline. 

## Running the training

Once the data is prepared, we recommend using the meta-training script `run_training.py`. You probably want to edit the file to your needs. The `main()` function of the file basically collects and organizes all the command line parameters for your training. All possible command line parameters can be seen by running `python train.py -h`. The parameters are collected as list of strings. The variable `shared_params` contains all parameters, that are shared across all the models that you want to train. For example, you can add the path to your dataset here. Afterward, we collect the parameters for each training in the variable `params`. For each training run, you need to add a list of command line parameters to the `params`. At the same time, also add a string to the list `tags`, which describes what you are doing and which is used as name for the output folder. The complete usage is explained in detail in the comments of the file.

The script will execute `train.py` once for each element in `params`, create an output directory on completion and move the checkpoint, final model, and log directory to that output directory. 

You can also run `train.py` directly. A checkpoint, the best model, and the log directory will be created in the root directory of the project. However, they will not be moved upon completion. 
