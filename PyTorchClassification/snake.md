<h1 align="center">Snake Species Identification Challenge</h1>

<p align="center">
  <img src="nbs/0ea412abd5014df4ecacc804d5907cb0.jpg" alt="Opheodrys vernalis"/>
</p>

Snakebite causes over 100,000 human deaths and 400,000 fatal injuries globally every year. Accurately Identifying the Genus & Species of the snakes will help in correct administration of anti-venoms. [AICrowd](https://www.aicrowd.com/) along with [Institute of Global Health](https://www.aicrowd.com/organizers/institute-of-global-health) hosted the [Snake Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge) in January, 2019.  
The goal is to build a Machine Learning model that could accurately classify the Genus & Species of snakes. For Phase 1, they shared 82,601 Images, spread across 45 different Species.

I started building on top of [SpeciesClassification](https://github.com/microsoft/SpeciesClassification/tree/master/PyTorchClassification) repository by Microsoft. The model was built for [iNaturalist-2018 challenge](https://www.kaggle.com/c/inaturalist-2018) and handles imbalanced classes very well.  
Since the model expects the input data in COCO format, I wrote a python script to convert the snakes' dataset into COCO format. I then trained the model for 2 different architectures, ResNext-101 and Inceptionv4. I trained each of them for around 80 Epochs at different Image Sizes [224, 488]. I then wrote a test_script that takes the best model from the above training, makes a prediction based on multiple splits(12) of the image. It stores the result in a CSV file suitable to submit in the challenge.

I got an F1-Score of `0.809` for `Inceptionv4` and `0.804` for `ResNext101`. I then averaged out the model predictions
by the two best models and was able to achieve an F1-Score of `0.846`.

> Steps to replicate the results

Follow the steps in the README.md to create the required docker or conda environment.  
Download the data from [here](https://www.aicrowd.com/challenges/snake-species-identification-challenge/dataset_files) and store it inside data repository in the root directory.  

```
cd {root}
python folder_to_coco.py          # Creates the COCO annotation format for the dataset
python run_snakes_training.py     # Trains the model for ResNext101 and Inceptionv4 architecture
python test_snakes.py             # Generates the prediction result on the test dataset
python merge_snakes_results.py    # Merges the results by different models
```