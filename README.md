# Cell Semantic Segmentation Repo
## Introduction
To help speed up research on cell microscopy images, this semantic segmentation program labels images on where the cell is segmentated.

## How to train model
Create conda environment using the environment.yml file and using the command: ```conda env create -f environment.yml```
If you don't have conda or conda isn't working run:
pip install -r requirements.txt

edit path in Train_model.py
make sure to change training and validate directory in Train_model.py

Then just run python Train_model.py

## How to test model
There are 3 testing files currently in use:

- Test_model.py - show image, label patch, predication patch
- Test_patched_predictions - show full size image, label, and prediction
- Test_full_image - show each prediction patch, as well as full size image, label, and prediction

Make sure to change root and test variables to the correct directories

* Windows doesn't work with ray. I can setup Windows support if there is a need.
