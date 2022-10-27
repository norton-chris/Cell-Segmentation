#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
This program will load model weights inputted and show
the output with a test image.
"""
# ---------------------------------------------------------------------------

# Built-in
import os
import sys

# 3rd Party Libs
import warnings
from datetime import datetime
import cv2
import numpy as np
import random
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import Models
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Union, cast
import argparse

# Owned
from Patcher import Patcher
import Batch_loader
from Random_patcher import Random_patcher
from Unpatcher import Unpatcher
import Scoring
__author__ = "Chris Norton"
__maintainer__ = "Chris Norton"
__email__ = "cnorton@mtu.edu"
__status__ = "Dev"
useLabels = False  # always have this set to False

# {code}
################################# CONFIGURATION ###############################
input_dir = "TrainingDataset/data_subset/output/test/"  # EDIT THIS LINE
output = "inference/predictions/"  # EDIT THIS LINE, change path to empty directory to save timelapse in
confidence_level = 0.5  # choose between 0.0 (lowest confidence) and 1.0 (highest confidence)
see_confidence_mask = False # choose if you want a binary mask (False) or a confidence mask with decimal predictions (True)
# useLabels can be useful for seeing the accuracy.
################################# EDIT THE LINE ABOVE ###############################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default=input_dir,
    help="input directory with timelapse pictures")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=output,
    help="output directory for segmented timelapse pictures")
parser.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=confidence_level,
    help="choose what confidence level to use. The higher the more under segmented the prediction will be,"
         + "the lower the more over segmented the prediction will be")
parser.add_argument(
    "-s",
    "--see_confidence_mask",
    type=bool,
    default=confidence_level,
    help="choose what confidence level to use. The higher the more under segmented the prediction will be,"
         + "the lower the more over segmented the prediction will be")

args = parser.parse_args()
input_dir = args.input
output = args.output
confidence_level = args.confidence
see_confidence_mask = args.see_confidence_mask

def normalize_image(input_block):
    block = input_block.copy()
    vol_max, vol_min = block.max(), block.min()
    if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
        for i in range(block.shape[-1]):
            block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
    return block

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
input_dir = input_dir + "Images/"  # uncomment if you images are in a folder called Images
dims = 256
step = 256
# Predict on patches
model = load_model('h5_files/model-best.h5',  # location of h5 file
                  custom_objects = { 'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                    'dice_scoring': Scoring.dice_scoring})

# load test patches
images = np.zeros((len(os.listdir(input_dir)), dims, dims, 1), dtype="float32")  # define the numpy array for the batch
masks = np.zeros((len(os.listdir(input_dir)), dims, dims, 1), dtype=bool)
resize =  np.zeros((1, dims, dims, 1), dtype=int)
i = 0
row = 0
col = 0
print("total image shape:", images.shape)
for path in os.listdir(input_dir):
    print("loop input", input_dir + path)
    print("loop output", output + path)
    img = cv2.imread(input_dir + path, -1).astype("float32")
    if useLabels:
        lab = cv2.imread(input_dir + "/Labels/" + path, -1)  # HERE'S THE LINE THAT READS THE LABELS

    batch_size = int(img.shape[0] / step) * int(img.shape[1] / step)
    if not useLabels:
        patcher_img = Patcher(img, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
        images, row, col = patcher_img.patch_image()
    else:
        patcher_img = Patcher(img, lab, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
        images, masks, row, col = patcher_img.patch_image()
    print("1 image shape:", images.shape)
    preds_test = model.predict(images, verbose=1)
    #pred_imgs = np.empty((i, dims, dims, 1), dtype=int)
    if not see_confidence_mask:
        preds_test = (preds_test > confidence_level) #.astype(np.uint8)
    #preds_full_image = (preds_full_image > 0.4).astype(np.uint8)
    print("Starting to segment images")

    # Stitch together patches into full sized image
    unpatcher = Unpatcher(img, preds_test, img_name=input_dir+path)
    full_pred_image = unpatcher.unpatch_image()
    int_img = np.array(full_pred_image, dtype="uint8")

    # Write images to specific directory
    cv2.imwrite(output + path, full_pred_image)
    print("Wrote image", i, "out of", len(os.listdir(input_dir)))
    i += 1
