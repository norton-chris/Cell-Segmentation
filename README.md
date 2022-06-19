# Cell Semantic Segmentation Repo
## Introduction
To help speed up research on cell microscopy images, this semantic segmentation program labels images on where the cell is segmentated.

## How to train model
Create conda environment using the environment.yml file and using the command: ```conda env create -f environment.yml```
If you don't have conda or conda isn't working run:
```pip install -r requirements.txt```

edit path in Train_model.py
make sure to change training and validate directory in Train_model.py

Then just run python Train_model.py

## HYPER-Parameterization (I use this a lot)
I use ```specified_gpu_wandb.py``` the most. I just set the GPU at the bottom
of the file. 
```wandb_train.py``` runs with any available gpu.

## How to segment image
There are 3 testing files currently in use:
edit "root = "TrainingDataset/data_subset/output/test/"" with a path to
a directory of images. (line 41)

- *Test_patched_predictions - show full size image, label, and prediction (I use this one the most)
- Test_model.py - show image, label patch, predication patch
- Test_full_image - show each prediction patch, as well as full size image, label, and prediction

Make sure to change root and test variables to the correct directories

## How to segment timelapse
Segment_timelapse.py segments each image in a directory and saves to another specified directory.
Usage: ```Segment_timelapse.py -i "input_folder" -o "output_folder" --confidence 0.5 --see_confidence_mask False```
-i: path to input folder
-o: path to output folder
-c: confidence level between 0.0 and 1.0 (higher under segments, lower over segments)
-s: see confidence mask - if this is set to False you get a binary mask,
True overrides confidence level and produces a mask of confidence levels for each pixel.

* Windows doesn't work with ray. I can setup Windows support if there is a need.
