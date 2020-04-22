
## Overview

This project contains the training code for the Microsoft AI for Earth Species Classification API, along with the code for our [API demo page](http://aka.ms/speciesclassification).  This API classifies handheld photos of around 5000 plant and animal species.  There is also a pipeline included for training detectors, and an API layer that simplifies running inference with an existing model, either on whole images or on detected crops.

The training data is not provided in this repo, so you can think of this repo as a set of tools for training fine-grained classifiers.  If you want lots of animal-related data to play around with, check out our open data repository at [lila.science](http://lila.science), including LILA's list of [other data sets](http://lila.science/otherdatasets) related to conservation.


## I don't want to train anything, I just want your model

No problem!  The model is publicly available:

* [PyTorch model file](https://lilablobssc.blob.core.windows.net/models/species_classification/species_classification.2019.12.00.pytorch)
* [ONNX model file] (https://lilablobssc.blob.core.windows.net/models/species_classification/species_classification.2019.12.00.onnx)
* [Class list](https://lilablobssc.blob.core.windows.net/models/species_classification/species_classification.2019.12.00.classes.txt)
* [Taxonomy file used for Latin &rarr; common mapping](https://lilablobssc.blob.core.windows.net/models/species_classification/species_classification.2019.12.00.taxa.csv)

Your one-stop-shop for learning how to run this model is the [classify_images.py](https://github.com/microsoft/SpeciesClassification/blob/master/classify_images.py) script in the root of this repo.


## Getting started with model training

See the [README](PyTorchClassification/README.md) in the `PyTorchClassification` directory to get started training your own classification models with this PyTorch-based framework.


## And if you love snakes...

This repo was also used as the basis for the winning entry in the first round of the [AIcrowd Snake Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge).  To replicate those results, see [snakes.md](PyTorchClassification/snakes/snakes.md).


## License

This repository is licensed with the [MIT license](https://github.com/Microsoft/dotnet/blob/master/LICENSE).


## Third-party components

The FasterRCNNDetection directory is based on [https://github.com/chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch).

The PyTorchClassification directory is based on the [ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) from the PyTorch codebase.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit [https://cla.microsoft.com](cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
