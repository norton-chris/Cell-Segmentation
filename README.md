# Cell Semantic Segmentation Repo
## Introduction
To help speed up research on cell microscopy images, this semantic segmentation program labels images on where the cell is segmentated.

## How to train model
Create conda environment using the environment.yml file and using the command: ```conda env create -f environment.yml```

edit path in Train_model.py
make sure to have folders names (shown below)

![alt text](https://github.com/norton-chris/Cell-Segmentation/change_dataset_path.png)

Then just run python Train_model.py


## How this repo is organized
```
├───.idea
│   └───inspectionProfiles
├───.ipynb_checkpoints
├───GCP_Notebook
│   └───.ipynb_checkpoints
├───h5 files
├───inference
│   ├───predictions
│   └───screenshots
├───logs
│   └───fit
│       ├───UNet_20220204-235403
│       │   └───train
│       │       └───plugins
│       │           └───profile
│       │               └───2022_02_05_04_54_09
│        U-Net_Specialist VGG19-U-Net_Generalist VGG19-U-Net
│   │   │   └───training_curves
│   │   │       ├───round1_cryptic_VGG19_dropout_round1_cryptic_VGG19_dropout_sm
│   │   │       ├───round1_generalist_unet_round1_generalist_VGG19_dropout
│   │   │       ├───round1_mDia_raw_unet_round1_mDia_raw_VGG19_dropout
│   │   │       ├───round1_paxillin_TIRF_normalize_cropped_unet_patience_10_round1_paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10
│   │   │       └───round1_unet_round1_VGG19_dropout
│   │   ├───subaxis
│   │   └───Violinplot-Matlab-master
│   │       └───test_cases
│   ├───generated_edge
│   │   ├───040119_PtK1_S01_01_phase
│   │   └───040119_PtK1_S01_01_phase_ROI2
│   ├───img_proc
│   ├───label_tool
│   │   ├───generated_edge
│   │   │   └───040119_PtK1_S01_01_phase_ROI2
│   │   └───__pycache__
│   ├───models
│   │   ├───results
│   │   │   ├───history_round1_Multi_VGG19D
│   │   │   ├───model_round1_Multi_VGG19D
│   │   │   └───predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256_patience_10
│   │   │       └───FNA_test
│   │   │           └───frame2_training_repeat0
│   │   │               └───bootstrapped_MTL_auto_reg_aut
│   │   └───__pycache__
│   ├───tests
│   ├───visualization
│   └───__pycache__
├───random-images
├───test_images
│   └───.ipynb_checkpoints
└───__pycache__
```