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

# {code}
################################# EDIT THE LINE BELOW ###############################
test = "TrainingDataset/data_subset/output/test/" ## EDIT THIS LINE
useLabels = True # set to true if you have a folder called Labels inside test (the above variable)
# useLabels can be useful for seeing the accuracy.
################################# EDIT THE LINE ABOVE ###############################

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

#################### MAIN ************************
test = test + "Images/" # Uncomment if you have a folder inside called Images
dims = 512
step = 512
# Predict on patches
model = load_model('h5_files/model-best.h5',
                  custom_objects = { 'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                    'dice_scoring': Scoring.dice_scoring})

# load test patches
images = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype="float32")  # define the numpy array for the batch
masks = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype=bool)
resize =  np.zeros((1, dims, dims, 1), dtype=int)
i = 0
num_of_images = 0
image_name = ""
max_x_pixel = 512
max_y_pixel = 512
print("total image shape:", images.shape)
for path in os.listdir(test): # Loop over Images in Directory
    print("loop", test + path)
    img = cv2.imread(test + path, -1).astype("float32")
    if useLabels:
        lab = cv2.imread(test + "../Labels/" + path, -1) # HERE'S THE LINE THE READS THE LABELS

    batch_size = int(img.shape[0]/step) * int(img.shape[1]/step)
    if not useLabels:
        patcher_img = Patcher(img, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
    else:
        patcher_img = Patcher(img, lab, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
    images, masks, row, col = patcher_img.patch_image()
    print("1 image shape:", images.shape)
    preds_test = model.predict(images, verbose=1)

    # Predicting resized images
    #resized = cv2.resize(img, (dims, dims))
    #resize = resized.reshape(1, step, step, 1)

    # Predicting full sized images
    #preds_full_image = model.predict(resize)
    preds_test = (preds_test > 0.2) #.astype(np.uint8) # showing predictions with
    #preds_full_image = (preds_full_image > 0.4).astype(np.uint8)

    for i in range(0, len(preds_test)):
        # create figure
        fig = plt.figure(figsize=(20, 14))

        fig.add_subplot(1,3, 1)

        # showing image
        plt.imshow(img)
        plt.axis('off')
        plt.title("image")

        fig.add_subplot(1, 3, 2)

        # showing image
        if useLabels:
            plt.imshow(lab)
        plt.axis('off')
        plt.title("label")

        unpatcher = Unpatcher(img, preds_test, img_name=test+path)
        full_pred_image = unpatcher.unpatch_image()

        int_img = np.array(full_pred_image, dtype="uint8")
        grey = int_img[:,:,0]
        ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        remove_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        fig.add_subplot(1, 3, 3)
        plt.imshow(full_pred_image)
        #plt.imshow(preds_full_image)
        plt.axis('off')
        plt.title("prediction")

        plt.show()
        break
