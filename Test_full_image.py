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
root = "TrainingDataset/data_subset/output/train/"
test = root + "Images/"
dims = 512
step = 512
# Predict on patches
model = load_model('h5_files/model-best-unet512.h5',
                  custom_objects = { 'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                    'dice_scoring': Scoring.dice_scoring})

# load test patches
images = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype="float32")  # define the numpy array for the batch
masks = np.zeros((len(os.listdir(test)), dims, dims, 1), dtype=bool)
resize =  np.zeros((1, dims, dims, 1), dtype=int)
i = 0
num_of_images = 0
image_name = ""
print("total image shape:", images.shape)
for path in os.listdir(test):
    print("loop", test + path)
    img = cv2.imread(test + path, -1).astype("float32")
    lab = cv2.imread(root + "Labels/" + path, -1)

    batch_size = int(img.shape[0]/step) * int(img.shape[1]/step)
    patcher_img = Patcher(img, lab, batch_size=batch_size, input_shape=(dims, dims, 1), step=step)
    images, masks, row, col = patcher_img.patch_image()
    print("1 image shape:", images.shape)
    preds_test = model.predict(images, verbose=1)

    resized = cv2.resize(img, (dims, dims))
    resize = resized.reshape(1, step, step, 1)
    preds_full_image = model.predict(resize)
    #pred_imgs = np.empty((i, dims, dims, 1), dtype=int)
    #preds_test = (preds_test > 0.2) #.astype(np.uint8)
    #preds_full_image = (preds_full_image > 0.4).astype(np.uint8)
    for i in range(0, len(preds_test)):
        # create figure
        fig = plt.figure(figsize=(20, 14))

        for j in range(1, batch_size+1):
            # Adds a subplot at the 1st position
            fig.add_subplot(int(row/step) + 1, int(col/step), j)

            # showing image
            plt.imshow(preds_test[j-1])
            plt.axis('off')

            #cv2.imwrite("inference/predictions/images/Image[" + str(i) + "].tif", images[i])
            #cv2.imwrite("inference/predictions/predict/Prediction[" + str(i) + "].tif", preds_test[i])
        fig.add_subplot(int(row/step)+ 1, int(col/step), j+1)

        # showing image
        plt.imshow(img)
        plt.axis('off')
        plt.title("image")

        fig.add_subplot(int(row/step)+ 1, int(col/step), j + 2)

        # showing image
        plt.imshow(lab)
        plt.axis('off')
        plt.title("label")

        unpatcher = Unpatcher(img, preds_test, img_name=test+path)
        full_pred_image = unpatcher.unpatch_image()

        int_img = np.array(full_pred_image, dtype="uint8")
        #reshaped = int_img.reshape(lab.shape[0], lab.shape[1], 1)
        grey = int_img[:,:,0]
        ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        remove_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        fig.add_subplot(int(row/step)+ 1, int(col/step), j + 3)

    #     # showing image
    #     plt.imshow(full_pred_image)
    #     plt.axis('off')
    #     plt.title("full size prediction")
        plt.imshow(full_pred_image)
        #plt.imshow(preds_full_image)
        plt.axis('off')
        plt.title("prediction")

        intersection = np.logical_and(lab, full_pred_image)
        union = np.logical_or(lab, full_pred_image)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IOU:", iou_score)

        # fig.add_subplot(int(row / step) + 2, int(col / step) + 3, j + 4)
        # plt.imshow(remove_noise)
        # # plt.imshow(preds_full_image)
        # plt.axis('off')
        # plt.title("remove noise")

        plt.show()
        break
